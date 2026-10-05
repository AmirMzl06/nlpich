"""Runner for the split-brain decoder on nlp21 (CORP handwriting data).

Trains, then evaluates like eval_single_model.py and prints the final CER
(greedy CTC, no LM) on seen days, unseen days, and the eval_single_model.py pool.

Examples
--------
# end-to-end, CTC + split-brain (activity + latent targets), no noise
python start_split_brain.py --out_dir sb_conv_M --mode e2e --aux M

# split-brain pretraining -> frozen encoders -> CTC decoder
python start_split_brain.py --out_dir sb_conv_M_freeze --mode pretrain_freeze --aux M

# Gaussian noise on the neural input / on the embedding fed to the CTC decoder
python start_split_brain.py --out_dir sb_conv_M_in08 --aux M --input_noise_sd 0.8
python start_split_brain.py --out_dir sb_conv_M_emb05 --aux M --emb_noise_sd 0.5

# evaluation only (like eval_single_model.py)
python start_split_brain.py --out_dir sb_conv_M --eval_only
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher

from utils.data_loader import get_input
from utils.dataset import HandwritingDataset, charset
from utils.split_brain_model import build_model

SEED_DIR = "seed_model_training_data/mat/"
NO_RECAL_DIR = "online_evaluation_data/no_recalibration/mat/"
RECAL_DIR = "online_evaluation_data/recalibration/mat/"


# =====================================================================
# arguments
# =====================================================================
def get_parser():
    p = argparse.ArgumentParser(description="Split-brain brain-to-text decoder (nlp21)")
    # paths / modes
    p.add_argument("--datasetPath", type=str, default="/data/hossein/mm_project/CORP_data_release")
    p.add_argument("--out_dir", type=str, default="split_brain_default")
    p.add_argument("--mode", type=str, default="e2e", choices=["e2e", "pretrain_freeze", "pretrain_finetune"],
                   help="e2e: CTC + split-brain jointly | pretrain_freeze: split-brain, then frozen encoders + CTC "
                        "| pretrain_finetune: split-brain, then everything fine-tuned with CTC")
    p.add_argument("--aux", type=str, default="M", choices=["none", "T", "L", "M"],
                   help="T: predict other view's activity | L: predict other view's latent | M: both")
    p.add_argument("--split", type=str, default="array", choices=["array", "random", "interleave", "none"],
                   help="array: channels 0-95 | 96-191; none: one encoder on all channels (baseline)")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--eval_only", action="store_true", help="load out_dir/modelWeights and only evaluate")
    p.add_argument("--no_resume", action="store_true", help="ignore out_dir/checkpoint.pt")
    p.add_argument("--device", type=str, default="cuda")
    # encoder
    p.add_argument("--arch", type=str, default="conv", choices=["conv", "gru", "transformer", "conformer"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=32, help="bottleneck per view (concat = 2x)")
    p.add_argument("--enc_layers", type=int, default=4, help="trunk depth (use ~2 for gru)")
    p.add_argument("--enc_dropout", type=float, default=0.1)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--conv_kernel", type=int, default=0, help="0 = auto (5 for conv, 15 for conformer)")
    p.add_argument("--kernel", type=int, default=32)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--smooth_sigma", type=float, default=2.0, help="Gaussian smoothing inside the model, 0 = off")
    # decoder (defaults = the project's command)
    p.add_argument("--decoder", type=str, default="gru", choices=["gru", "linear"])
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--no_bidir", action="store_true")
    # split-brain loss
    p.add_argument("--lambda_T", type=float, default=1.0)
    p.add_argument("--lambda_L", type=float, default=1.0)
    p.add_argument("--n_bins", type=int, default=16)
    p.add_argument("--target_window", type=str, default="center", choices=["center", "full"])
    p.add_argument("--head_hidden", type=int, default=256, help="0 = linear heads")
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument("--no_ema", action="store_true", help="L targets from the online encoder (stop-grad)")
    # noise (training only)
    p.add_argument("--input_noise_sd", type=float, default=0.0, help="Gaussian noise on the neural input")
    p.add_argument("--input_offset_sd", type=float, default=0.0, help="per-trial constant offset on the input")
    p.add_argument("--emb_noise_sd", type=float, default=0.0, help="Gaussian noise on the embedding fed to CTC")
    # optimisation
    p.add_argument("--batchSize", type=int, default=16)
    p.add_argument("--nBatch", type=int, default=20000, help="CTC steps (e2e / after pretraining)")
    p.add_argument("--pretrain_steps", type=int, default=10000)
    p.add_argument("--optim", type=str, default="auto", choices=["auto", "project", "adamw"],
                   help="project: Adam(eps=0.1) + linear decay as utils/trainer.py; adamw: AdamW + warmup + cosine; "
                        "auto: project for conv/gru, adamw for transformer/conformer")
    p.add_argument("--lrStart", type=float, default=0.02)
    p.add_argument("--lrEnd", type=float, default=0.002)
    p.add_argument("--adam_eps", type=float, default=0.1)
    p.add_argument("--l2_decay", type=float, default=1e-5)
    p.add_argument("--adamw_lr", type=float, default=1e-3)
    p.add_argument("--adamw_wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    # logging / eval
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    return p


def finalize_args(args):
    if args["conv_kernel"] <= 0:
        args["conv_kernel"] = 15 if args["arch"] == "conformer" else 5
    if args["optim"] == "auto":
        args["optim"] = "adamw" if args["arch"] in ("transformer", "conformer") else "project"
    if args["split"] == "none" and args["aux"] != "none":
        print("split=none -> no split-brain loss, setting aux=none")
        args["aux"] = "none"
    if args["mode"] != "e2e" and args["aux"] == "none":
        raise ValueError(f"mode={args['mode']} needs a split-brain loss (aux != none)")
    return args


# =====================================================================
# data
# =====================================================================
def ctc_collate(batch: List[Tuple[torch.Tensor, str, int]]):
    # same as utils/trainer.py
    xs, ys, ds = zip(*batch)
    B = len(xs)
    feat_dim = xs[0].shape[-1]
    input_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(input_lengths.max().item())
    x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        x_pad[i, T:] = x[-1:]
    target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    max_target_len = int(target_lengths.max()) if B > 0 else 0
    targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)
    for i, t in enumerate(target_seqs):
        targets_padded[i, :t.numel()] = t
    return x_pad, targets_padded, input_lengths, target_lengths, torch.tensor(ds, dtype=torch.long)


def session_names(path):
    # same ordering as utils.data_loader.get_input
    root = Path(path)
    return [str(f.relative_to(root).with_suffix("")) for f in sorted(root.rglob("*.mat"))]


def split_by_borders(items, borders):
    ends = borders[1:] + [len(items)]
    return [items[s:e] for s, e in zip(borders, ends)]


def load_train_items(dataset_path):
    return get_input(os.path.join(dataset_path, SEED_DIR), norm=True, gauss=False, train=True, gauss_sigma=2.0)


def load_eval_items(dataset_path):
    """Eval list in the exact order of utils/eval_utils.get_dataset_loader_nlp_21, plus a tag per trial.

    tags: (split, session_name) with split in {'seen', 'no_recal', 'recal'}
    """
    seed_path = os.path.join(dataset_path, SEED_DIR)
    seen, seen_b = get_input(seed_path, norm=True, gauss=False, train=False, valid=True,
                             gauss_sigma=2.0, return_borders=True)
    seen_tags = []
    for name, chunk in zip(session_names(seed_path), split_by_borders(seen, seen_b)):
        seen_tags += [("seen", name)] * len(chunk)

    nr_path, rc_path = os.path.join(dataset_path, NO_RECAL_DIR), os.path.join(dataset_path, RECAL_DIR)
    nr, nr_b = get_input(nr_path, norm=True, gauss=False, train=False, valid=False,
                         gauss_sigma=2.0, return_borders=True)
    rc, rc_b = get_input(rc_path, norm=True, gauss=False, train=False, valid=False,
                         gauss_sigma=2.0, return_borders=True)
    nr_s = list(zip(session_names(nr_path), split_by_borders(nr, nr_b)))
    rc_s = list(zip(session_names(rc_path), split_by_borders(rc, rc_b)))
    if len(nr_s) != len(rc_s):
        print(f"WARNING: {len(nr_s)} no_recal vs {len(rc_s)} recal sessions; "
              f"eval_single_model.py would fail its assert here, extra sessions are appended at the end")

    unseen, unseen_tags = [], []
    for i in range(max(len(nr_s), len(rc_s))):  # same interleaving as merge_by_borders
        for split, sessions in (("no_recal", nr_s), ("recal", rc_s)):
            if i < len(sessions):
                name, chunk = sessions[i]
                unseen += chunk
                unseen_tags += [(split, name)] * len(chunk)
    return seen + unseen, seen_tags + unseen_tags


def make_loader(items, batch_size, shuffle, num_workers=0):
    return DataLoader(HandwritingDataset(items), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, collate_fn=ctc_collate,
                      persistent_workers=num_workers > 0, drop_last=False)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# =====================================================================
# evaluation (same decoding / CER as utils/eval_utils.eval_model)
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    model.eval()
    all_loss, per_trial = [], []
    for X, y, X_len, y_len, _ in loader:
        X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred, lengths = model(X, X_len)
        pred = pred.float()
        loss = ctc(pred.log_softmax(2).permute(1, 0, 2), y, lengths, y_len)
        all_loss.append(loss.item())
        for i in range(pred.shape[0]):
            dec = torch.argmax(pred[i, :lengths[i], :], dim=-1)
            dec = torch.unique_consecutive(dec, dim=-1).cpu().numpy()
            dec = [int(c) for c in dec if c != 0]
            true = y[i, :y_len[i]].cpu().numpy().tolist()
            per_trial.append((SequenceMatcher(a=true, b=dec).distance(), len(true)))
    return per_trial, float(np.sum(all_loss) / max(len(loader), 1))


def cer_of(pairs):
    n = sum(l for _, l in pairs)
    return sum(d for d, _ in pairs) / n if n > 0 else float("nan")


def summarize(per_trial, tags):
    groups = {"seen": [], "no_recal": [], "recal": []}
    sessions = {}
    for (d, l), (split, name) in zip(per_trial, tags):
        groups[split].append((d, l))
        sessions.setdefault((split, name), []).append((d, l))
    out = {k: cer_of(v) for k, v in groups.items()}
    out["unseen_all"] = cer_of(groups["no_recal"] + groups["recal"])
    out["eval_single_pool"] = cer_of(per_trial)
    out["n_trials"] = {k: len(v) for k, v in groups.items()}
    out["per_session"] = [{"split": s, "session": n, "cer": cer_of(v), "n_trials": len(v)}
                          for (s, n), v in sessions.items()]
    return out


def print_final(summary, loss):
    n = summary["n_trials"]
    print("\n" + "=" * 64)
    print("FINAL CER (greedy CTC, no LM)")
    print("=" * 64)
    print(f"seen days   (held-out block of training sessions) : {summary['seen']:.4f}   ({n['seen']} trials)")
    print(f"unseen days, no_recalibration                      : {summary['no_recal']:.4f}   ({n['no_recal']} trials)")
    print(f"unseen days, recalibration                         : {summary['recal']:.4f}   ({n['recal']} trials)")
    print(f"unseen days, all  (= utils/trainer.py test set)    : {summary['unseen_all']:.4f}")
    print(f"eval_single_model.py pool (seen + unseen)          : {summary['eval_single_pool']:.4f}")
    print("-" * 64)
    print("per session:")
    for s in summary["per_session"]:
        print(f"  {s['split']:>8s}  {s['session']:<40s}  CER {s['cer']:.4f}  ({s['n_trials']} trials)")
    print("-" * 64)
    print(f"CER: {summary['eval_single_pool']:.4f}, Loss: {loss:.4f}   <- same numbers as eval_single_model.py")


def final_eval(model, args, device, eval_items, eval_tags):
    loader = make_loader(eval_items, args["eval_batch_size"], shuffle=False)
    per_trial, loss = evaluate(model, loader, device, amp=not args["no_amp"])
    summary = summarize(per_trial, eval_tags)
    summary["eval_single_loss"] = loss
    print_final(summary, loss)
    with open(os.path.join(args["out_dir"], "evalStats.pkl"), "wb") as f:
        pickle.dump(per_trial, f)  # same content as eval_single_model.py's evalStats.pkl
    with open(os.path.join(args["out_dir"], "final_cer.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args['out_dir']}/final_cer.json and evalStats.pkl")
    return summary


# =====================================================================
# diagnostics
# =====================================================================
@torch.no_grad()
def channel_correlation_report(model, items, device, max_trials=100):
    """Mean |corr| (smoothed input) within / between the two views; within >> between supports the array split."""
    if model.split == "none":
        return
    x = torch.cat([model.smoother(it[0][None].float().to(device))[0] for it in items[:max_trials]], 0)
    c = torch.corrcoef(x.T).nan_to_num(0.0).abs().cpu()
    a, b = model.idx_A.cpu(), model.idx_B.cpu()

    def off_diag_mean(m):
        n = m.shape[0]
        return ((m.sum() - m.diagonal().sum()) / (n * n - n)).item()

    print(f"[split check] mean |corr| within A: {off_diag_mean(c[a][:, a]):.4f} | "
          f"within B: {off_diag_mean(c[b][:, b]):.4f} | between A-B: {c[a][:, b].mean().item():.4f}")


# =====================================================================
# training
# =====================================================================
def build_phases(args):
    """List of (name, n_steps, use_ctc, use_aux, trainable)."""
    use_aux = args["aux"] != "none"
    if args["mode"] == "e2e":
        return [("e2e", args["nBatch"], True, use_aux, "all")]
    pre = ("pretrain", args["pretrain_steps"], False, True, "encoder")
    if args["mode"] == "pretrain_freeze":
        return [pre, ("ctc_frozen", args["nBatch"], True, False, "decoder")]
    return [pre, ("finetune", args["nBatch"], True, False, "all")]


def make_optimizer(params, args, n_steps):
    if args["optim"] == "project":
        opt = torch.optim.Adam(params, lr=args["lrStart"], betas=(0.9, 0.999), eps=args["adam_eps"],
                               weight_decay=args["l2_decay"])
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                                  end_factor=args["lrEnd"] / args["lrStart"], total_iters=n_steps)
    else:
        opt = torch.optim.AdamW(params, lr=args["adamw_lr"], weight_decay=args["adamw_wd"])
        warm = max(args["warmup"], 1)

        def f(s):
            prog = min(s, n_steps) / max(n_steps, 1)
            return min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    return opt, sched


def set_phase(model, trainable):
    for p in model.parameters():
        p.requires_grad_(False)
    if trainable in ("all", "encoder"):
        for p in model.encoder_side_parameters():
            p.requires_grad_(True)
    if trainable in ("all", "decoder"):
        for p in model.decoder_parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def train_mode(model, trainable):
    model.train()
    if trainable == "decoder":
        model.encoders.eval()  # frozen encoders: no dropout


def save_checkpoint(path, model, opt, sched, phase_idx, step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "phase_idx": phase_idx, "step": step}, path)


def train(args, model, device, train_items, eval_items, eval_tags):
    out_dir = args["out_dir"]
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    amp = not args["no_amp"]
    ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    train_loader = make_loader(train_items, args["batchSize"], shuffle=True, num_workers=args["num_workers"])
    eval_loader = make_loader(eval_items, args["eval_batch_size"], shuffle=False)
    model.emb_noise_sd = args["emb_noise_sd"]

    resume = None
    if os.path.exists(ckpt_path) and not args["no_resume"]:
        resume = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(resume["model"])
        print(f"resuming from phase {resume['phase_idx']} step {resume['step']}")

    stats = {"step": [], "phase": [], "cer_seen": [], "cer_unseen": []}
    stats_path = os.path.join(out_dir, "trainingStats")
    if resume is not None and os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

    batches = infinite(train_loader)
    bad_streak = 0
    for phase_idx, (name, n_steps, use_ctc, use_aux, trainable) in enumerate(build_phases(args)):
        if resume is not None and phase_idx < resume["phase_idx"]:
            continue
        params = set_phase(model, trainable)
        opt, sched = make_optimizer(params, args, n_steps)
        start = 0
        if resume is not None and phase_idx == resume["phase_idx"]:
            opt.load_state_dict(resume["opt"])
            sched.load_state_dict(resume["sched"])
            start = resume["step"] + 1
        print(f"\n=== phase '{name}': {n_steps} steps | ctc={use_ctc} aux={use_aux and args['aux']} "
              f"| trainable={trainable} ({sum(p.numel() for p in params) / 1e6:.2f}M params) ===")

        running = {}
        t0 = time.time()
        for step in range(start, n_steps):
            train_mode(model, trainable)
            X, y, X_len, y_len, _ = next(batches)
            X, y, X_len, y_len = X.to(device), y.to(device), X_len.to(device), y_len.to(device)
            X_clean = X
            noisy = args["input_noise_sd"] > 0 or args["input_offset_sd"] > 0
            if args["input_noise_sd"] > 0:
                X = X + torch.randn_like(X) * args["input_noise_sd"]
            if args["input_offset_sd"] > 0:
                X = X + torch.randn(X.shape[0], 1, X.shape[2], device=device) * args["input_offset_sd"]

            logs = {}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if use_ctc:
                    pred, out_len = model(X, X_len)
                else:
                    _, out_len = model.encode(X, X_len)
                loss = 0.0
                if use_ctc:
                    l_ctc = ctc(pred.float().log_softmax(2).permute(1, 0, 2), y, out_len, y_len)
                    loss = loss + l_ctc
                    logs["ctc"] = l_ctc.item()
                if use_aux:
                    aux = model.aux_losses(X_clean if noisy else None)
                    if "T" in aux:
                        loss = loss + args["lambda_T"] * aux["T"]
                    if "L" in aux:
                        loss = loss + args["lambda_L"] * aux["L"]
                    logs.update({k: float(v) for k, v in aux.items()})

            if torch.isfinite(loss):
                bad_streak = 0
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=args["grad_clip"])
                opt.step()
                if use_aux and model.use_L:
                    model.update_ema(args["ema_decay"])
                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                running["_n"] = running.get("_n", 0) + 1
            else:  # skip the update, keep the schedule / eval / checkpoint cadence
                bad_streak += 1
                print(f"[{name}] non-finite loss at step {step + 1}, update skipped ({bad_streak} in a row)")
                if bad_streak >= 10:
                    raise RuntimeError("10 non-finite losses in a row, stopping")
            sched.step()

            if (step + 1) % args["log_every"] == 0:
                n = running.pop("_n", 0)
                if n > 0:
                    msg = " ".join(f"{k}={v / n:.4f}" for k, v in running.items())
                    print(f"[{name}] step {step + 1}/{n_steps} {msg} lr={sched.get_last_lr()[0]:.2e} "
                          f"({time.time() - t0:.0f}s)")
                running = {}

            last = step == n_steps - 1
            if use_ctc and ((step + 1) % args["eval_every"] == 0 or last):
                per_trial, _ = evaluate(model, eval_loader, device, amp)
                s = summarize(per_trial, eval_tags)
                print(f"[{name}] step {step + 1}: CER seen {s['seen']:.4f} | unseen {s['unseen_all']:.4f} "
                      f"(no_recal {s['no_recal']:.4f}, recal {s['recal']:.4f})")
                for k, v in (("step", step + 1), ("phase", name), ("cer_seen", s["seen"]),
                             ("cer_unseen", s["unseen_all"])):
                    stats[k].append(v)
                with open(stats_path, "wb") as f:
                    pickle.dump(stats, f)
            if (step + 1) % args["eval_every"] == 0 or last:
                torch.save(model.state_dict(), os.path.join(out_dir, "modelWeights"))
                save_checkpoint(ckpt_path, model, opt, sched, phase_idx, step)
        resume = None


def main():
    try:  # show logs immediately when stdout is redirected (nohup / > log)
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = finalize_args(vars(get_parser().parse_args()))
    device = args["device"]
    os.makedirs(args["out_dir"], exist_ok=True)

    if args["eval_only"]:
        with open(os.path.join(args["out_dir"], "args"), "rb") as f:
            model_args = pickle.load(f)
        model_args.update({k: args[k] for k in ("datasetPath", "out_dir", "eval_batch_size", "no_amp", "device")})
        model = build_model(model_args).to(device)
        model.load_state_dict(torch.load(os.path.join(args["out_dir"], "modelWeights"), map_location=device))
        eval_items, eval_tags = load_eval_items(model_args["datasetPath"])
        final_eval(model, model_args, device, eval_items, eval_tags)
        return

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    with open(os.path.join(args["out_dir"], "args"), "wb") as f:
        pickle.dump(args, f)
    print(json.dumps(args, indent=1))

    model = build_model(args).to(device)
    n_enc = sum(p.numel() for p in model.encoders.parameters())
    n_dec = sum(p.numel() for p in model.decoder_parameters())
    print(f"model: split={args['split']} arch={args['arch']} aux={args['aux']} mode={args['mode']} | "
          f"encoders {n_enc / 1e6:.2f}M, decoder {n_dec / 1e6:.2f}M params")

    train_items = load_train_items(args["datasetPath"])
    eval_items, eval_tags = load_eval_items(args["datasetPath"])
    print(f"train trials: {len(train_items)} | eval trials: {len(eval_items)}")
    channel_correlation_report(model, train_items, device)

    ckpt_exists = os.path.exists(os.path.join(args["out_dir"], "checkpoint.pt")) and not args["no_resume"]
    if model.use_T and not ckpt_exists:
        n, n_const = model.fit_bins([it[0] for it in train_items], seed=args["seed"])
        print(f"fitted {args['n_bins']} quantile bins per channel on {n} windows "
              f"({n_const} constant channels excluded from the T loss)")

    train(args, model, device, train_items, eval_items, eval_tags)
    final_eval(model, args, device, eval_items, eval_tags)


if __name__ == "__main__":
    main()
