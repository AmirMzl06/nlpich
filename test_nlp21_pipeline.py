"""Small CPU integration/contract tests; no CORP data or GPU required.

Run beside nlpich's utils/: python test_nlp21_pipeline.py
Does not assert a research result or a particular random-data CER.
"""
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import savemat
import torch

from jigsaw_net import JigsawNet
from mobile_jigsaw import MobileJigsaw
from nlp21_data import (load_corp, feature_trials, feature_stats, edit_distance,
                       ctc_collapse, corpus_metrics, BOS, EOS, CHAR_TO_ID, N_CLASSES)
from nlp21_decoders import (DECODERS, build_decoder, collate_trials, training_loss,
                           evaluate_decoder, validate_ctc_lengths, load_torch)


def write_fixture(root):
    rng = np.random.default_rng(17)
    paths = ["seed_model_training_data/mat", "online_evaluation_data/no_recalibration/mat",
             "online_evaluation_data/recalibration/mat"]
    for group, folder in enumerate(paths):
        path = Path(root) / folder
        path.mkdir(parents=True)
        n = 6 if group == 0 else 3
        neural, sentences, blocks = (np.empty((1, n), dtype=object) for _ in range(3))
        for i in range(n):
            # Different offsets per block exercise block-local normalization.
            neural[0, i] = rng.normal(2 + i // 2, 1, (70 + 3 * i, 8)).astype(np.float32)
            sentences[0, i] = ["aa", "ab", "a>b"][i % 3]
            blocks[0, i] = np.array([[i // 2 + 1]]) if group == 0 else np.array([[1]])
        savemat(path / "day_01.mat", dict(tx_feats=neural, sentences=sentences, blocks=blocks))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    assert len(CHAR_TO_ID) == 31 and N_CLASSES == 32
    assert ctc_collapse([1, 1, 0, 1, 2, 2, 0]) == [1, 1, 2]
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("a" * 300, "") == 300
    pooled = corpus_metrics([dict(edits=1, reference_length=1), dict(edits=0, reference_length=9)])
    assert pooled["cer"] == 0.1  # micro, not average sentence CER (=0.5)
    print("OK alphabet, CTC collapse, long edit distance and corpus weighting")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixture(root / "corp")
        groups, manifest = load_corp(root / "corp", sigma=0)
        assert len(groups["train"]) == 4 and len(groups["dev"]) == 2
        assert len(groups["online_test"]) == 6
        # Each pair of seed train trials forms a block: verify block moments.
        for i in (0, 2):
            x = np.concatenate([t.x for t in groups["train"][i:i+2]])
            np.testing.assert_allclose(x.mean(0), 0, atol=2e-6)
            np.testing.assert_allclose(x.std(0), 1, atol=2e-6)
        # Same smoother implementation supports inferred feature count.
        smooth, _ = load_corp(root / "corp", sigma=2)
        assert smooth["train"][0].x.shape == groups["train"][0].x.shape
        print("OK MATLAB loading, max-block split, online groups, normalization, smoothing")

        shared = dict(window_size=4, n_tiles=2, tile_gap=(1, 2), output_dimension=8,
                      num_hidden_units=8, head_hidden_units=8, max_epochs=1,
                      batch_size=64, device="cpu", verbose=False)
        model = JigsawNet(**shared)
        model._build(8)
        _, starts, _ = model._spans([t.x for t in groups["train"]])
        ends = np.cumsum([len(t.x) for t in groups["train"]])
        for start in starts:
            trial_id = np.searchsorted(ends, start, side="right")
            assert start + model.training_span <= ends[trial_id]
        model.fit([t.x for t in groups["train"]])
        assert len(model.history_) == 1 and np.isfinite(model.history_[0]["total"])
        model.save(root / "encoder.pt")
        restored = JigsawNet.load(root / "encoder.pt", device="cpu")
        arrays = [t.x for t in groups["dev"]]
        for a, b in zip(model.transform(arrays), restored.transform(arrays)):
            np.testing.assert_allclose(a, b)
        print("OK trial boundaries, real Jigsaw optimization and checkpoint roundtrip")

        # Verify each decoder sees a fixed encoder, no autograd path to SSL weights.
        model.encoder_.requires_grad_(False)
        before = {k: v.clone() for k, v in model.encoder_.state_dict().items()}
        embeddings = {k: model.transform([t.x for t in ts], pad=True) for k, ts in groups.items()}
        mean, std = feature_stats(embeddings["train"], 2)
        features = {k: feature_trials(groups[k], z, stride=2, mean=mean, std=std)
                    for k, z in embeddings.items()}
        validate_ctc_lengths(features["train"])
        bad = features["train"][0]
        from dataclasses import replace
        try:
            validate_ctc_lengths([replace(bad, x=bad.x[:2], targets=(1, 1))])
            raise AssertionError("Adjacent-repeat infeasibility not detected")
        except ValueError:
            pass
        cfg = dict(hidden=8, layers=1, dropout=0., bidirectional=True, batch_size=2,
                   max_decode_chars=5)
        x, lengths, y, y_lengths, _ = collate_trials(features["train"][:2])
        for kind in DECODERS:
            decoder = build_decoder(kind, 8, cfg)
            loss = training_loss(decoder, kind, x, lengths, y, y_lengths)
            assert torch.isfinite(loss)
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
            metrics, rows = evaluate_decoder(decoder, kind, features["dev"], cfg, "cpu")
            assert np.isfinite(metrics["cer"]) and len(rows) == 2
            torch.save(decoder.state_dict(), root / "decoder.pt")
            other = build_decoder(kind, 8, cfg)
            other.load_state_dict(load_torch(root / "decoder.pt", "cpu"))
            _, rows2 = evaluate_decoder(other, kind, features["dev"], cfg, "cpu")
            assert rows == rows2
            print(f"OK {kind}: finite loss/gradients, free decoding, serialization")
        for k, v in model.encoder_.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        attention = build_decoder("attention_ce", 8, cfg).eval()
        # CER is independent of reference labels except its edit-distance scoring.
        _, rows_a = evaluate_decoder(attention, "attention_ce", features["dev"], cfg, "cpu")
        changed = [replace(t, targets=(1,) * 12) for t in features["dev"]]
        _, rows_b = evaluate_decoder(attention, "attention_ce", changed, cfg, "cpu")
        assert [r['prediction'] for r in rows_a] == [r['prediction'] for r in rows_b]
        # Packed recurrent lengths prevent other trials' padding from changing a prediction.
        gru = build_decoder("gru_ctc", 8, cfg).eval()
        with torch.no_grad():
            batched = gru(x, lengths)
            n = int(lengths[0])
            alone = gru(x[:1, :n], lengths[:1])
        torch.testing.assert_close(batched[0, :n], alone[0], atol=1e-6, rtol=1e-5)
        print("OK frozen encoder, attention without reference leakage, recurrent padding isolation")

        for version in ("v2", "v3"):
            mobile = MobileJigsaw(version=version, **shared)
            mobile.fit([t.x for t in groups["train"]])
            out = mobile.transform([groups["dev"][0].x])
            assert out[0].shape == (len(groups["dev"][0].x), 8)
            print(f"OK MobileJigsaw {version}: list-of-trials fit and transform")
    print("All NLP21 pipeline contract tests passed. No dataset performance claim.")


if __name__ == "__main__":
    main()
