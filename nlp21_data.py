"""NLP21 CORP loader and character metrics; imports no project model.

Block normalization and seed/online trial selection reuse utils.data_loader.
The new runner uses held-out seed blocks as development data. Online trials
are evaluated only after decoder checkpoint selection.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import torch


# Exactly the NLP21 alphabet/order in nlpich/utils/dataset.py.
CHARS = list(">,?~'abcdefghijklmnopqrstuvwxyz")
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}
CTC_BLANK = PAD = 0
N_CLASSES = len(CHARS) + 1  # 32: blank + 31 characters
BOS, EOS = N_CLASSES, N_CLASSES + 1
AR_CLASSES = N_CLASSES + 2


@dataclass
class Trial:
    x: np.ndarray
    targets: tuple
    transcript: str
    session: int
    uid: str
    group: str


def decode_ids(ids):
    return "".join(CHARS[int(i) - 1] for i in ids if 1 <= int(i) < N_CLASSES)


def edit_distance(reference, hypothesis):
    """Exact Levenshtein distance, integer arithmetic (no uint8 overflow)."""
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(reference) + 1))
    for i, token in enumerate(hypothesis, 1):
        current = [i]
        for j, ref in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != token)))
        previous = current
    return previous[-1]


def ctc_collapse(path):
    """Merge consecutive repeats FIRST, then remove blank."""
    result, previous = [], None
    for token in path:
        token = int(token)
        if token != previous and token != CTC_BLANK:
            result.append(token)
        previous = token
    return result


def corpus_metrics(rows):
    total_edits = sum(r["edits"] for r in rows)
    total_chars = sum(r["reference_length"] for r in rows)
    if not total_chars:
        raise ValueError("Evaluation set has no reference characters.")
    return dict(cer=total_edits / total_chars,
                cer_percent=100 * total_edits / total_chars,
                edits=total_edits, reference_characters=total_chars,
                n_trials=len(rows),
                exact_sentence_percent=100 * sum(r["edits"] == 0 for r in rows) / len(rows),
                generation_limit_hits=sum(r.get("generation_limit_hit", False) for r in rows))


def _read_group(path, group, *, train, valid, sigma, unknown_chars):
    # Import just the loader/augmentation, never utils.trainer or utils.model.
    from utils.data_loader import get_input
    from utils.augmentation import GaussianSmoothing

    path = Path(path)
    files = sorted(path.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files in {path}. Check --datasetPath.")
    items = get_input(str(path), norm=True, gauss=False, train=train, valid=valid)
    if not items:
        raise ValueError(f"No trials selected for {group} in {path}.")
    smoother = None
    result, dropped = [], {}
    for index, (features, text, session) in enumerate(items):
        if not isinstance(text, str):
            raise TypeError(f"{group}/{index}: sentence must be a string, got {type(text)}")
        unknown = set(text) - set(CHAR_TO_ID)
        if unknown and unknown_chars == "error":
            raise ValueError(
                f"{group}/{index}: unsupported characters {sorted(unknown)!r}. "
                "Use --unknown-chars drop only to reproduce the original charset's "
                "silent dropping. Spaces are not automatically converted to '>'.")
        for c in unknown:
            dropped[c] = dropped.get(c, 0) + text.count(c)
        targets = tuple(CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID)
        if not targets:
            raise ValueError(f"{group}/{index}: empty target after alphabet conversion.")
        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
            raise ValueError(f"{group}/{index}: expected finite nonempty (T,F) input.")
        if sigma > 0:
            if smoother is None:
                smoother = GaussianSmoothing(features.shape[1], 20, sigma, dim=1)
            with torch.no_grad():
                features = smoother(features[None])[0]
        result.append(Trial(np.ascontiguousarray(features.numpy()), targets, text,
                            int(session), f"{group}/{index:06d}", group))
    if dropped:
        print(f"{group}: unsupported characters DROPPED: {dropped}", flush=True)
    print(f"{group}: {len(result)} trials, {sum(len(t.x) for t in result):,} bins, "
          f"{result[0].x.shape[1]} features", flush=True)
    return result, [str(f.resolve()) for f in files]


def load_corp(root, *, sigma=2.0, unknown_chars="drop", max_trials=0):
    root = Path(root).expanduser().resolve()
    specs = (
        ("train", "seed_model_training_data/mat", True, False),
        ("dev", "seed_model_training_data/mat", False, True),
        ("online_no_recalibration", "online_evaluation_data/no_recalibration/mat", False, False),
        ("online_recalibration", "online_evaluation_data/recalibration/mat", False, False),
    )
    groups, sources = {}, {}
    for name, relative, train, valid in specs:
        trials, files = _read_group(root / relative, name, train=train, valid=valid,
                                    sigma=sigma, unknown_chars=unknown_chars)
        groups[name] = trials[:max_trials] if max_trials else trials
        sources[name] = files
    if len({t.x.shape[1] for ts in groups.values() for t in ts}) != 1:
        raise ValueError("All train/dev/online trials must have the same feature dimension.")
    groups["online_test"] = (groups.pop("online_no_recalibration")
                             + groups.pop("online_recalibration"))
    manifest = dict(root=str(root), source_files=sources,
                    normalization="per-file per-block mean/std from that block, including held-out blocks",
                    gaussian_sigma=sigma, gaussian_kernel=20 if sigma else 0,
                    gaussian_scope="each trial separately, zero same-padding (project GaussianSmoothing)",
                    unknown_characters=unknown_chars, alphabet=CHARS,
                    smoke_limit_per_source=max_trials,
                    selection="dev=max block ID per seed MAT; online excluded from checkpoint selection",
                    project_final_definition="dev + both online groups; includes decoder-selection dev",
                    trials={k: [dict(uid=t.uid, group=t.group, session=t.session,
                                      bins=len(t.x), chars=len(t.targets)) for t in ts]
                            for k, ts in groups.items()})
    return groups, manifest


def feature_trials(trials, embeddings, *, stride, mean=None, std=None):
    if len(trials) != len(embeddings):
        raise ValueError("Encoder dropped a trial.")
    output = []
    for trial, z in zip(trials, embeddings):
        if len(z) != len(trial.x):
            raise ValueError("Expected pad=True: one embedding per original input bin.")
        z = np.asarray(z[::stride], dtype=np.float32)
        if mean is not None:
            z = (z - mean) / std
        if not np.isfinite(z).all():
            raise FloatingPointError(f"Nonfinite decoder features in {trial.uid}")
        output.append(replace(trial, x=np.ascontiguousarray(z)))
    return output


def feature_stats(embeddings, stride):
    """Streaming stable, time-weighted moments, fitted ONLY on train features."""
    count, mean, m2 = 0, None, None
    for z in embeddings:
        x = np.asarray(z[::stride], dtype=np.float64)
        n = len(x)
        local_mean = x.mean(0)
        local_m2 = np.square(x - local_mean).sum(0)
        if count == 0:
            count, mean, m2 = n, local_mean, local_m2
        else:
            delta = local_mean - mean
            total = count + n
            m2 += local_m2 + delta * delta * (count * n / total)
            mean += delta * (n / total)
            count = total
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-6).astype(np.float32)


def dump_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
