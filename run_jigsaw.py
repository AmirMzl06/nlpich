"""NLP21 runner: unlabeled Jigsaw -> frozen trial embeddings -> character CER.

Place this file, nlp21_data.py, nlp21_decoders.py, jigsaw_net.py and
mobile_jigsaw.py beside nlpich/start_trainer.py (the existing utils/ is reused).
This replaces the PERICH runner only. The two uploaded encoder files are intact.

Example:
python -u run_jigsaw.py --datasetPath /data/hossein/mm_project/CORP_data_release \
  --arms order_reconstruct reconstruct_only random_encoder \
  --decoders gru_ctc linear_ctc attention_ce --epochs 60 --seeds 8
"""
import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from jigsaw_net import JigsawNet
from mobile_jigsaw import MobileJigsaw
from nlp21_data import (load_corp, feature_trials, feature_stats, corpus_metrics,
                       dump_json, N_CLASSES)
from nlp21_decoders import (DECODERS, seed_all, train_decoder, evaluate_decoder,
                           validate_ctc_lengths)


ARMS = {
    "order_reconstruct": {},
    "proposed": {},  # alias: current uploaded implementation, shuffle_tiles=True
    "reconstruct_only": dict(lambda_order=0.0, lambda_pair=0.0),
    "order_only": dict(lambda_reconstruct=0.0),
    "with_forecast": dict(lambda_forecast=1.0),
    "forecast_only": dict(lambda_order=0.0, lambda_pair=0.0,
                          lambda_reconstruct=0.0, lambda_forecast=1.0),
    "tiles_2": dict(n_tiles=2),
    "dim_256": dict(output_dimension=256),
    "mobile_v1": dict(_model="mobile", version="v1"),
    "mobile_v2": dict(_model="mobile", version="v2"),
    "mobile_v3": dict(_model="mobile", version="v3"),
    "mobile_stem_mix": dict(_model="mobile", stem="mix"),
    "mobile_alpha_035": dict(_model="mobile", width_multiplier=0.35),
    "random_encoder": dict(max_epochs=0),
    "mobile_random": dict(_model="mobile", max_epochs=0),
    "raw": dict(_model="raw"),
}


def arm_override(name):
    if name in ARMS:
        return dict(ARMS[name])
    # e.g. mobile_v3_random / mobile_stem_mix_random / dim_256_random
    if name.endswith("_random") and name[:-7] in ARMS:
        return dict(ARMS[name[:-7]], max_epochs=0)
    raise ValueError(f"Unknown arm {name!r}. Use --list.")


def model_settings(args, arm, seed):
    cfg = dict(window_size=args.window_size, n_tiles=args.n_tiles, tile_gap=tuple(args.tile_gap),
               output_dimension=args.output_dim, num_hidden_units=args.encoder_hidden,
               head_hidden_units=args.head_hidden, dropout=args.encoder_dropout,
               normalize=False, trunk_block="residual", lambda_order=args.lambda_order,
               lambda_pair=args.lambda_pair, lambda_reconstruct=args.lambda_reconstruct,
               lambda_forecast=args.lambda_forecast, forecast_levels=args.levels,
               order_grad_scale=1.0, tile_norm=args.tile_norm, shuffle_tiles=True,
               neuron_dropout=args.neuron_dropout, gain_jitter=args.gain_jitter,
               batch_size=args.jigsaw_batch_size, max_epochs=args.epochs,
               learning_rate=args.jigsaw_lr, weight_decay=args.jigsaw_weight_decay,
               device=args.device, random_state=seed, verbose=True,
               log_every=args.jigsaw_log_every)
    cfg.update(arm_override(arm))
    return cfg.pop("_model", "jigsaw"), cfg


def eligible_trials(trials, span, label):
    keep = [t for t in trials if len(t.x) >= span]
    skipped = [t.uid for t in trials if len(t.x) < span]
    if skipped:
        print(f"{label}: {len(skipped)} trials shorter than {span} bins omitted ONLY "
              "from puzzle sampling; all remain in decoder/CER.", flush=True)
    return keep, skipped


def pretext_metrics(model, trials, args):
    if model.lambda_order == 0 and model.lambda_pair == 0:
        return dict(applicable=False, reason="Order heads not trained in this arm")
    eligible, skipped = eligible_trials(trials, model.training_span, "pretext evaluation")
    if not eligible:
        return dict(applicable=False, reason="No trial long enough", omitted_trials=skipped)
    metrics = model.evaluate_pretext([t.x for t in eligible], max_spans=args.pretext_spans,
                                     batch_size=args.pretext_batch_size,
                                     repeats=args.pretext_repeats, random_state=args.eval_seed,
                                     verbose=True)
    metrics.update(applicable=True, omitted_trials=skipped,
                   trained_order_head=model.max_epochs > 0,
                   sampling_note="Overlapping spans are correlated; no binomial significance claimed")
    return metrics


def standardize_and_stride(groups, embeddings, args):
    mean = std = None
    if args.feature_norm == "train":
        mean, std = feature_stats(embeddings["train"], args.decoder_stride)
    features = {split: feature_trials(groups[split], zs, stride=args.decoder_stride,
                                       mean=mean, std=std) for split, zs in embeddings.items()}
    return features, mean, std


def decoder_config(args):
    return dict(hidden=args.decoder_hidden, layers=args.decoder_layers,
                dropout=args.decoder_dropout, bidirectional=not args.unidirectional,
                steps=args.decoder_steps, batch_size=args.decoder_batch_size,
                lr=args.decoder_lr, weight_decay=args.decoder_weight_decay,
                eval_every=args.eval_every, log_every=args.decoder_log_every,
                max_decode_chars=args.max_decode_chars)


def write_summary(root, rows):
    dump_json(root / "results.json", rows)
    if not rows:
        return
    fields = ["seed", "arm", "decoder", "epochs", "best_step", "dev_cer_percent",
              "online_cer_percent", "project_final_cer_percent", "online_no_recalibration_cer_percent",
              "online_recalibration_cer_percent", "decoder_parameters"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\nSUMMARY: lower CER is better; project_final includes the selection dev set.", flush=True)
    print(f"{'seed':>5} {'arm':<25} {'decoder':<14} {'dev CER%':>10} "
          f"{'online CER%':>12} {'project CER%':>13}", flush=True)
    for r in rows:
        print(f"{r['seed']:5} {r['arm']:<25} {r['decoder']:<14} "
              f"{r['dev_cer_percent']:10.2f} {r['online_cer_percent']:12.2f} "
              f"{r['project_final_cer_percent']:13.2f}", flush=True)
    pooled = []
    for arm, decoder in sorted({(r['arm'], r['decoder']) for r in rows}):
        chosen = [r for r in rows if r['arm'] == arm and r['decoder'] == decoder]
        values = np.array([r['online_cer_percent'] for r in chosen])
        pooled.append(dict(arm=arm, decoder=decoder, seeds=[r['seed'] for r in chosen],
                           mean_online_cer_percent=float(values.mean()),
                           sample_sd_online_cer_percent=float(values.std(ddof=1)) if len(values) > 1 else None))
    dump_json(root / "pooled.json", pooled)


def check_args(args):
    for key in ("decoder_steps", "decoder_stride", "decoder_batch_size", "decoder_hidden",
                "decoder_layers", "eval_every", "decoder_log_every", "max_decode_chars",
                "jigsaw_log_every", "transform_batch_size", "pretext_spans", "pretext_repeats",
                "pretext_batch_size", "cpu_threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.epochs < 0 or args.max_trials < 0 or args.gaussian_sigma < 0:
        raise ValueError("epochs, max-trials and gaussian-sigma must be nonnegative")
    if args.decoder_lr <= 0 or args.decoder_weight_decay < 0 or not 0 <= args.decoder_dropout < 1:
        raise ValueError("Invalid decoder learning rate, weight decay or dropout")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        raise ValueError("Seeds and arm names must be unique")
    if len(set(args.decoders)) != len(args.decoders):
        raise ValueError("Decoder names must be unique")
    if "proposed" in args.arms and "order_reconstruct" in args.arms:
        raise ValueError("proposed and order_reconstruct are aliases; select one")
    for name in args.arms:
        arm_override(name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasetPath", "--data-dir", dest="dataset_path", type=Path,
                   default=Path("/data/hossein/mm_project/CORP_data_release"))
    p.add_argument("--out-dir", "--out_dir", dest="out_dir", type=Path, default=Path("JIGSAW_NLP21_RESULTS"))
    p.add_argument("--arms", nargs="+", default=["order_reconstruct", "reconstruct_only", "random_encoder"])
    p.add_argument("--decoders", nargs="+", choices=DECODERS, default=list(DECODERS))
    p.add_argument("--seeds", nargs="+", type=int, default=[8])
    p.add_argument("--epochs", type=int, default=60, help="Full passes over all eligible puzzle starts, NOT optimizer steps")
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--n-tiles", type=int, default=4)
    p.add_argument("--tile-gap", type=int, nargs=2, default=(1, 8), metavar=("MIN", "MAX"))
    p.add_argument("--output-dim", type=int, default=64)
    p.add_argument("--encoder-hidden", type=int, default=64)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--encoder-dropout", type=float, default=0.0)
    p.add_argument("--lambda-order", type=float, default=1.0)
    p.add_argument("--lambda-pair", type=float, default=0.5)
    p.add_argument("--lambda-reconstruct", type=float, default=1.0)
    p.add_argument("--lambda-forecast", type=float, default=0.0)
    p.add_argument("--levels", type=int, default=8, help="Quantization levels shared by reconstruct AND forecast")
    p.add_argument("--tile-norm", choices=["mean", "none", "zscore", "global_mean"], default="mean")
    p.add_argument("--neuron-dropout", type=float, default=0.1)
    p.add_argument("--gain-jitter", type=float, default=0.1)
    p.add_argument("--jigsaw-batch-size", type=int, default=128)
    p.add_argument("--jigsaw-lr", type=float, default=1e-3)
    p.add_argument("--jigsaw-weight-decay", type=float, default=0.0)
    p.add_argument("--jigsaw-log-every", type=int, default=1)
    p.add_argument("--transform-batch-size", type=int, default=2048)
    p.add_argument("--gaussian-sigma", type=float, default=2.0, help="0 disables; otherwise trialwise project GaussianSmoothing(kernel=20)")
    p.add_argument("--unknown-chars", choices=["error", "drop"], default="drop",
                   help="Default reproduces the original charset's dropping, with counts logged")
    p.add_argument("--feature-norm", choices=["train", "none"], default="train")
    p.add_argument("--decoder-stride", type=int, default=4, help="Keep embeddings at bins 0,s,2s,... within each trial")
    p.add_argument("--decoder-steps", type=int, default=20000)
    p.add_argument("--decoder-batch-size", type=int, default=16)
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--decoder-dropout", type=float, default=0.3)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--decoder-weight-decay", type=float, default=1e-4)
    p.add_argument("--unidirectional", action="store_true", help="GRUs only; Jigsaw centered windows still use future bins")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--decoder-log-every", type=int, default=100)
    p.add_argument("--max-decode-chars", type=int, default=512, help="Attention free-generation limit; includes optional EOS step")
    p.add_argument("--pretext-spans", type=int, default=1024)
    p.add_argument("--pretext-repeats", type=int, default=2)
    p.add_argument("--pretext-batch-size", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=200008)
    p.add_argument("--max-trials", type=int, default=0, help="Smoke runs only: first N trials PER source group")
    p.add_argument("--device", default="cuda_if_available")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--reuse-encoders-from", type=Path, help="Previous run directory; checks data/config, skips SSL fit")
    p.add_argument("--list", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print("Arms:", ", ".join(ARMS))
        print("Matched zero-epoch controls: append _random to an arm (e.g. mobile_v3_random).")
        print("Decoders:", ", ".join(DECODERS))
        return
    check_args(args)
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda_if_available":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        # Set before CUDA initialization. Unsupported deterministic ops fail explicitly.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = args.out_dir.expanduser().resolve() / stamp
    root.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch_version=torch.__version__, numpy_version=np.__version__,
                  encoder_frozen=True, representation_labels_used=False,
                  evaluation="greedy character CER; no language model",
                  pad_transform=True, decoding_window="centered, offline/acausal")
    config["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("run_jigsaw.py", "jigsaw_net.py", "mobile_jigsaw.py",
                     "nlp21_data.py", "nlp21_decoders.py")}
    dump_json(root / "config.json", config)
    print(f"Output: {root}\nNLP21: {N_CLASSES} CTC classes, character CER (not PER).", flush=True)
    groups, manifest = load_corp(args.dataset_path, sigma=args.gaussian_sigma,
                                 unknown_chars=args.unknown_chars, max_trials=args.max_trials)
    # Include source file size/mtime for reuse checks; avoids a costly full data hash.
    source_stats = {f: [Path(f).stat().st_size, Path(f).stat().st_mtime_ns]
                    for fs in manifest["source_files"].values() for f in fs}
    fingerprint = hashlib.sha256(json.dumps([manifest, source_stats], sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    dump_json(root / "data_manifest.json", manifest)
    all_results = []
    cfg = decoder_config(args)
    for seed in args.seeds:
        for arm in args.arms:
            seed_all(seed)
            arm_dir = root / f"seed_{seed}" / arm
            arm_dir.mkdir(parents=True, exist_ok=False)
            family, settings = model_settings(args, arm, seed)
            print(f"\n[seed={seed}] {arm} family={family}", flush=True)
            started = time.monotonic()
            if family == "raw":
                embeddings = {k: [t.x for t in ts] for k, ts in groups.items()}
                encoder_info = dict(family="raw", epochs=0, parameters=0, pretext={})
            else:
                factory = MobileJigsaw if family == "mobile" else JigsawNet
                model = factory(**settings)
                train_trials, omitted = eligible_trials(groups["train"], model.training_span, "SSL train")
                if not train_trials:
                    raise ValueError(f"No train trial >= training span {model.training_span}")
                spans = sum(len(t.x) - model.training_span + 1 for t in train_trials)
                updates = (spans // model.batch_size + int(spans % model.batch_size >= 2)) * model.max_epochs
                print(f"SSL: {len(train_trials)} trials, {spans:,} eligible starts, "
                      f"{model.max_epochs} full epochs -> {updates:,} optimizer updates; "
                      f"input on device ~{sum(t.x.nbytes for t in train_trials) / 2**20:.1f} MiB.", flush=True)
                if args.reuse_encoders_from is not None:
                    old = args.reuse_encoders_from / f"seed_{seed}" / arm
                    info = json.loads((old / "encoder_info.json").read_text())
                    expected = json.loads(json.dumps(model.get_params()))
                    previous = dict(info["params"])
                    for ignore in ("device", "verbose", "log_every"):
                        expected.pop(ignore, None)
                        previous.pop(ignore, None)
                    if info["data_fingerprint"] != fingerprint or previous != expected:
                        raise ValueError("Saved encoder data/config mismatch; use its original SSL settings")
                    model = factory.load(old / "encoder.pt", device=args.device)
                    print(f"Reused encoder: {old / 'encoder.pt'}", flush=True)
                else:
                    # A list of actual trials. Original _spans() preserves every boundary.
                    model.fit([t.x for t in train_trials])
                model.save(arm_dir / "encoder.pt")
                pretext = {split: pretext_metrics(model, groups[split], args)
                           for split in ("dev", "online_test")}
                model.encoder_.eval()
                model.encoder_.requires_grad_(False)
                embeddings = {}
                for split, trials in groups.items():
                    print(f"Extracting {split} embeddings ({len(trials)} trials)", flush=True)
                    embeddings[split] = model.transform([t.x for t in trials], pad=True,
                                                        batch_size=args.transform_batch_size)
                encoder_info = dict(family=family, epochs=model.max_epochs,
                                     params=model.get_params(), data_fingerprint=fingerprint,
                                     parameters=sum(p.numel() for p in model.encoder_.parameters()),
                                     omitted_ssl_trials=omitted, n_spans=spans,
                                     optimizer_updates=updates, pretext=pretext)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            encoder_info["preparation_seconds"] = time.monotonic() - started
            dump_json(arm_dir / "encoder_info.json", encoder_info)
            features, mean, std = standardize_and_stride(groups, embeddings, args)
            np.savez(arm_dir / "feature_scaler.npz", mean=np.array([]) if mean is None else mean,
                     std=np.array([]) if std is None else std,
                     stride=args.decoder_stride, window_size=args.window_size)
            del embeddings
            # Check ALL splits before spending decoder training time.
            if any(d.endswith("ctc") for d in args.decoders):
                for trials in features.values():
                    validate_ctc_lengths(trials)
            for decoder in args.decoders:
                print(f"\n  Decoder: {decoder}; encoder frozen, no gradient to Jigsaw", flush=True)
                decoder_dir = arm_dir / decoder
                decoder_seed = seed + 100000 + DECODERS.index(decoder) * 1000
                dec, training = train_decoder(decoder, features["train"], features["dev"], cfg,
                                               decoder_dir, device, decoder_seed)
                dev_metrics, dev_rows = evaluate_decoder(dec, decoder, features["dev"], cfg, device)
                online_metrics, online_rows = evaluate_decoder(dec, decoder, features["online_test"], cfg, device)
                project_metrics = corpus_metrics(dev_rows + online_rows)
                subgroup = {g: corpus_metrics([r for r in online_rows if r["group"] == g])
                            for g in ("online_no_recalibration", "online_recalibration")}
                metrics = dict(dev=dev_metrics, online_test=online_metrics,
                               project_final=project_metrics, online_groups=subgroup, training=training,
                               note="project_final includes dev used for decoder selection")
                dump_json(decoder_dir / "metrics.json", metrics)
                dump_json(decoder_dir / "predictions_dev.json", dev_rows)
                dump_json(decoder_dir / "predictions_online.json", online_rows)
                result = dict(seed=seed, arm=arm, decoder=decoder, epochs=encoder_info["epochs"],
                              best_step=training["best_step"], decoder_parameters=training["decoder_parameters"],
                              dev_cer_percent=dev_metrics["cer_percent"],
                              online_cer_percent=online_metrics["cer_percent"],
                              project_final_cer_percent=project_metrics["cer_percent"],
                              online_no_recalibration_cer_percent=subgroup["online_no_recalibration"]["cer_percent"],
                              online_recalibration_cer_percent=subgroup["online_recalibration"]["cer_percent"],
                              paths=str(decoder_dir), metrics=metrics)
                all_results.append(result)
                write_summary(root, all_results)
                del dec
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del features
            gc.collect()
    print(f"\nSaved all results: {root}", flush=True)


if __name__ == "__main__":
    main()
