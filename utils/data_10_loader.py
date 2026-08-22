import numpy as np
import torch
import pickle
from utils.augmentation import GaussianSmoothing

def load_data_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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

def _safe_sentence(s):
    if isinstance(s, np.ndarray) and s.size == 1:
        s = s.item()
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except:
            pass
    return s

def slow_time_2x(x):
    return_numpy = False
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
        return_numpy = True

    T = x.shape[0]

    if T % 2:
        x = x[:-1]

    x = x.reshape(x.shape[0] // 2, 2, x.shape[1]).sum(dim=1)
    if return_numpy:
        x = x.numpy()
    return x

def get_input(file_path, norm=False, train=False, eps=0,
                        gauss = False, valid=False,
                        stats_dtype=np.float64, output_dtype=torch.float32, days = range(10),
                        gauss_sigma = 2.0
                        ):
    data = load_data_dict(file_path)
    if gauss:
        smoother = GaussianSmoothing(192, 20, gauss_sigma, dim=1)

    out = []
    for day_key in days:
        day_list = data[day_key]
        for d in day_list:
            d['trial'] = slow_time_2x(d['trial'])

        blocks = sorted({item["block"] for item in day_list})
        last_block = blocks[-1]
        stats_this_day = {}
        if norm:
            stats_this_day = _block_stats_chan([d["trial"] for d in day_list],
                                               np.array([d["block"] for d in day_list]),
                                               dtype=stats_dtype)
        samples = []
        for item in day_list:
            blk = item["block"]
            if train and blk == last_block:
                continue
            if valid and blk != last_block:
                continue
            x = np.asarray(item["trial"], dtype=np.float64)
            if x.ndim == 1:
                x = x[:, None]
            if norm:
                mu, std = stats_this_day[blk]
                std = np.maximum(std, eps+1e-6)
                x = (x - mu) / std
            
            x_t = torch.tensor(x, dtype=torch.float32)
            if gauss:
                with torch.no_grad():
                    x_t = smoother(x_t.unsqueeze(0)).squeeze(0)

            y = _safe_sentence(item["sentence"])
            day = 1
            samples.append((x_t, y, day))
        out += samples
    return out
