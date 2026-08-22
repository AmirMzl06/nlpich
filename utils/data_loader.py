import torch
import numpy as np
from pathlib import Path
import scipy.io as sio
from utils.constants import NUM_TRAIN_DAYS
from utils.augmentation import GaussianSmoothing

def _to_scalar(x):
    x = np.array(x).squeeze()
    return x.item() if x.shape == () else x


def _block_stats_chan(neural, block_ids, dtype=np.float64):
    stats = {}
    uniq = np.unique(block_ids)
    for b in uniq:
        n_total = 0
        mu = None
        M2 = None
        for i, bb in enumerate(block_ids):
            if bb != b:
                continue
            x = np.asarray(neural[i], dtype=dtype)
            if x.ndim == 1:
                x = x[:, None]
            n_i = x.shape[0]
            mu_i = x.mean(axis=0, dtype=dtype)
            diff = x - mu_i
            M2_i = (diff * diff).sum(axis=0, dtype=dtype)
            if n_total == 0:
                n_total = n_i
                mu = mu_i
                M2 = M2_i
            else:
                delta = mu_i - mu
                n = n_total + n_i
                mu = mu + delta * (n_i / n)
                M2 = M2 + M2_i + (delta * delta) * (n_total * n_i / n)
                n_total = n
        var = M2 / max(n_total, 1)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var, dtype=dtype)
        stats[b] = (mu, std)
    return stats


def get_input(path, norm=False, gauss=False, train=False, eps=0,
              stats_dtype=np.float64, output_dtype=torch.float32,
              valid=False,
              return_borders=False,
              gauss_sigma=2.0
              ):
    root = Path(path)
    if gauss:
        smoother = GaussianSmoothing(192, 20, gauss_sigma, dim=1)
    files_recursive = sorted(root.rglob("*.mat"))
    print(f"Found {len(files_recursive)} .mat files")


    per_file_block_stats = {}
    if norm:
        for fi, file in enumerate(files_recursive):
            mat = sio.loadmat(file)
            neural = mat["tx_feats"][0]
            blocks = mat["blocks"][0]
            block_ids = np.array([_to_scalar(b) for b in blocks])
            per_file_block_stats[fi] = _block_stats_chan(
                neural, block_ids, dtype=stats_dtype
            )

    out = []
    borders = []
    last_border = 0
    all_trials = 0
    train_trials = 0
    all_blocks_counter = 0

    for fi, file in enumerate(files_recursive):
        mat = sio.loadmat(file)
        neural = mat["tx_feats"][0]
        sentences = mat["sentences"][0]
        blocks = mat["blocks"][0]

        block_ids = np.array([_to_scalar(b) for b in blocks])
        majority_block = block_ids.max()

        if train:
            idxs = [i for i, b in enumerate(block_ids) if b != majority_block]
        elif valid: idxs = [i for i, b in enumerate(block_ids) if b == majority_block]
        else: idxs = [i for i, b in enumerate(block_ids)]
        train_idxs = [i for i,b in enumerate(block_ids) if b != majority_block]
        # print(train_trials, all_trials)
        train_trials += len(train_idxs)
        # print(set(block_ids))
        all_blocks = set(block_ids.tolist())
        block_ids_un = {}
        for i, block in enumerate(sorted(list(all_blocks))):
            block_ids_un[block] = (i + all_blocks_counter) if block != majority_block else (i + all_blocks_counter - 1)
        all_blocks_counter += len(all_blocks) - 1
        # print(block_ids_un)


        samples = []

        if norm:
                stats_this_file = per_file_block_stats.get(fi, {})
        else:
                stats_this_file = {}
        for i in idxs:
                x = np.asarray(neural[i], dtype=np.float64)
                if x.ndim == 1:
                    x = x[:, None]

                if norm:
                    mu, std = stats_this_file[block_ids[i]]
                    std = np.maximum(std, eps+1e-6)
                    x = (x - mu) / std

                x_t = torch.as_tensor(x, dtype=torch.float32)
                if gauss:
                    with torch.no_grad():
                        x_t = smoother(x_t.unsqueeze(0)).squeeze(0)

                
                y = _safe_sentence(sentences[i])
                # samples.append((x_t, y, all_trials if train else (train_trials - 1)))
                samples.append((x_t, y, fi if train or valid else (NUM_TRAIN_DAYS - 1)))
                # samples.append((x_t, y, block_ids_un[block_ids[i]]))
                all_trials += 1
        borders.append(last_border)
        last_border += len(samples)
        out += samples
    if return_borders: return out, borders
    return out


def _safe_sentence(s):
    if isinstance(s, np.ndarray) and s.size == 1:
        s = s.item()
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return s