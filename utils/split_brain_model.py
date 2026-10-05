"""Split-brain (cross-view prediction) model for brain-to-text decoding.

The input channels are split into two disjoint views (e.g. the two Utah arrays).
Each view has its own encoder; the encoders never see the other view.
Their bottleneck embeddings are concatenated and decoded with CTC, and during
training each view's embedding is asked to predict the other view:

    T: the other view's (smoothed, quantized) activity   -> cross-entropy
    L: the other view's latent from an EMA target encoder -> cosine (BYOL style)
    M: both

forward(x, lengths) -> (logits, out_lengths), same signature as utils.model.Encoder_Decoder.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.augmentation import GaussianSmoothing


def make_split(n_channels, split, seed=0):
    """Return (idx_A, idx_B) channel index tensors, or None for split='none'."""
    if split == "none":
        return None
    half = n_channels // 2
    if split == "array":
        idx_a, idx_b = torch.arange(0, half), torch.arange(half, n_channels)
    elif split == "interleave":
        idx_a, idx_b = torch.arange(0, n_channels, 2), torch.arange(1, n_channels, 2)
    elif split == "random":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_channels, generator=g)
        idx_a, idx_b = perm[:half].sort().values, perm[half:].sort().values
    else:
        raise ValueError(f"unknown split '{split}'")
    return idx_a, idx_b


def output_lengths(lengths, kernel, stride):
    # identical to utils.augmentation.Unfolder
    return ((lengths - kernel) / stride).to(torch.int32)


def valid_mask(out_len, n_steps):
    """(B, T') bool, True on valid steps. Every sequence keeps at least one step."""
    t = torch.arange(n_steps, device=out_len.device)
    return t[None, :] < out_len.clamp(min=1)[:, None]


# =====================================================================
# trunks: (B, T', d) + pad_mask (B, T', True = pad) -> (B, T', out_dim)
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, d, kernel, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = self.norm(x)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + self.drop(y)


class ConvTrunk(nn.Module):
    """Local temporal conv net (ConvNeXt-1D style blocks)."""

    def __init__(self, d, n_layers, dropout, conv_kernel=5):
        super().__init__()
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.blocks = nn.ModuleList([ConvBlock(d, conv_kernel, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x, pad_mask):
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.norm(x)


class GRUTrunk(nn.Module):
    def __init__(self, d, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(d, d, n_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.out_dim = 2 * d

    def forward(self, x, pad_mask):
        return self.rnn(x)[0]


def sinusoidal_pe(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


class TransformerTrunk(nn.Module):
    def __init__(self, d, n_layers, dropout, nhead=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        try:
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        except TypeError:  # older torch
            self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x, pad_mask):
        x = x + sinusoidal_pe(x.shape[1], x.shape[2], x.device).to(x.dtype)
        return self.norm(self.enc(x, src_key_padding_mask=pad_mask))


class FeedForward(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, d, nhead, dropout, conv_kernel):
        super().__init__()
        self.ff1 = FeedForward(d, dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, conv_kernel, padding=conv_kernel // 2, groups=d)
        self.dw_norm = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.conv_drop = nn.Dropout(dropout)
        self.ff2 = FeedForward(d, dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        y = self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)[0]
        x = x + self.attn_drop(y)
        y = F.glu(self.pw1(self.conv_norm(x)), dim=-1)
        y = y.masked_fill(pad_mask[..., None], 0.0)
        y = self.dw(y.transpose(1, 2)).transpose(1, 2)
        y = self.pw2(F.silu(self.dw_norm(y)))
        x = x + self.conv_drop(y)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerTrunk(nn.Module):
    def __init__(self, d, n_layers, dropout, nhead=4, conv_kernel=15):
        super().__init__()
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.blocks = nn.ModuleList([ConformerBlock(d, nhead, dropout, conv_kernel) for _ in range(n_layers)])
        self.out_dim = d

    def forward(self, x, pad_mask):
        x = x + sinusoidal_pe(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for b in self.blocks:
            x = b(x, pad_mask)
        return x


def build_trunk(arch, d, n_layers, dropout, nhead, conv_kernel):
    if arch == "conv":
        return ConvTrunk(d, n_layers, dropout, conv_kernel)
    if arch == "gru":
        return GRUTrunk(d, n_layers, dropout)
    if arch == "transformer":
        return TransformerTrunk(d, n_layers, dropout, nhead)
    if arch == "conformer":
        return ConformerTrunk(d, n_layers, dropout, nhead, conv_kernel)
    raise ValueError(f"unknown arch '{arch}'")


class ViewEncoder(nn.Module):
    """(B, T, C_view) smoothed input -> (B, T', emb_dim) normalized embedding.

    stem = Conv1d(kernel, stride), identical to Unfolder + Linear of the original pipeline.
    The embedding goes through a LayerNorm without affine, so its scale is fixed
    (needed for embedding noise to be meaningful).
    """

    def __init__(self, in_ch, arch, d_model, emb_dim, n_layers, kernel, stride, dropout, nhead, conv_kernel):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, d_model, kernel, stride)
        self.stem_norm = nn.LayerNorm(d_model)
        self.stem_drop = nn.Dropout(dropout)
        self.trunk = build_trunk(arch, d_model, n_layers, dropout, nhead, conv_kernel)
        self.proj = nn.Linear(self.trunk.out_dim, emb_dim)
        self.out_norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(self, x, out_len):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        h = self.stem_drop(self.stem_norm(h))
        pad_mask = ~valid_mask(out_len, h.shape[1])
        h = self.trunk(h, pad_mask)
        return self.out_norm(self.proj(h))


def mlp(d_in, d_hidden, d_out):
    if d_hidden <= 0:
        return nn.Linear(d_in, d_out)
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


# =====================================================================
# full model
# =====================================================================
class SplitBrainNet(nn.Module):
    def __init__(self, n_channels=192, n_classes=32, split="array", split_seed=0,
                 arch="conv", d_model=256, emb_dim=32, enc_layers=4, enc_dropout=0.1, nhead=4, conv_kernel=5,
                 kernel=32, stride=4, smooth_sigma=2.0,
                 decoder="gru", hidden=1024, layers=5, dropout=0.4, bidir=True,
                 aux="none", n_bins=16, target_window="center", head_hidden=256, use_ema=True):
        super().__init__()
        if aux != "none" and split == "none":
            raise ValueError("split-brain aux loss needs a split (split != 'none')")
        self.kernel, self.stride = kernel, stride
        self.aux = aux
        self.use_T = aux in ("T", "M")
        self.use_L = aux in ("L", "M")
        self.n_bins = n_bins
        self.target_window = target_window
        self.use_ema = use_ema
        self.emb_noise_sd = 0.0  # set by the trainer, used only in train mode

        self.smoother = GaussianSmoothing(n_channels, 20, smooth_sigma, dim=1) if smooth_sigma > 0 else nn.Identity()

        def enc(in_ch, out_dim):
            return ViewEncoder(in_ch, arch, d_model, out_dim, enc_layers, kernel, stride, enc_dropout, nhead, conv_kernel)

        sp = make_split(n_channels, split, split_seed)
        self.split = split
        if sp is None:
            self.register_buffer("idx_all", torch.arange(n_channels))
            self.encoders = nn.ModuleList([enc(n_channels, 2 * emb_dim)])
        else:
            self.register_buffer("idx_A", sp[0])
            self.register_buffer("idx_B", sp[1])
            self.encoders = nn.ModuleList([enc(len(sp[0]), emb_dim), enc(len(sp[1]), emb_dim)])
        z_dim = 2 * emb_dim

        if decoder == "gru":
            self.rnn = nn.GRU(z_dim, hidden, layers, batch_first=True, bidirectional=bidir,
                              dropout=dropout if layers > 1 else 0.0)
            self.dec_out = nn.Linear(hidden * (2 if bidir else 1), n_classes)
        elif decoder == "linear":
            self.rnn = None
            self.dec_out = nn.Linear(z_dim, n_classes)
        else:
            raise ValueError(f"unknown decoder '{decoder}'")

        if self.use_T:
            if n_bins < 2:
                raise ValueError("n_bins must be >= 2")
            n_a, n_b = len(self.idx_A), len(self.idx_B)
            self.head_AB = mlp(emb_dim, head_hidden, n_b * n_bins)  # A predicts B
            self.head_BA = mlp(emb_dim, head_hidden, n_a * n_bins)  # B predicts A
            self.register_buffer("bin_edges", torch.zeros(n_channels, n_bins - 1))
            self.register_buffer("bin_entropy", torch.zeros(n_channels))  # marginal entropy of each channel's bins
            self.register_buffer("bin_weight", torch.ones(n_channels))  # 0 for constant channels
            self.register_buffer("bins_fitted", torch.zeros((), dtype=torch.bool))
        if self.use_L:
            self.pred_AB = mlp(emb_dim, head_hidden, emb_dim)
            self.pred_BA = mlp(emb_dim, head_hidden, emb_dim)
            if use_ema:
                self.ema_encoders = copy.deepcopy(self.encoders)
                for p in self.ema_encoders.parameters():
                    p.requires_grad_(False)
        self._cache = {}

    # ---------------- parameter groups ----------------
    def encoder_side_parameters(self):
        """Encoders + split-brain heads (everything trained in the pretraining phase)."""
        mods = [self.encoders]
        for name in ("head_AB", "head_BA", "pred_AB", "pred_BA"):
            if hasattr(self, name):
                mods.append(getattr(self, name))
        return [p for m in mods for p in m.parameters()]

    def decoder_parameters(self):
        mods = [self.dec_out] + ([self.rnn] if self.rnn is not None else [])
        return [p for m in mods for p in m.parameters()]

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "ema_encoders"):
            self.ema_encoders.eval()  # targets are deterministic
        return self

    # ---------------- forward ----------------
    def _views(self, xs):
        if self.split == "none":
            return [xs]
        return [xs.index_select(-1, self.idx_A), xs.index_select(-1, self.idx_B)]

    def encode(self, x, lengths):
        xs = self.smoother(x)
        out_len = output_lengths(lengths, self.kernel, self.stride)
        hs = [e(v, out_len) for e, v in zip(self.encoders, self._views(xs))]
        self._cache = {"x": x, "xs": xs, "hs": hs, "out_len": out_len}
        return torch.cat(hs, dim=-1), out_len

    def decode(self, z):
        if self.training and self.emb_noise_sd > 0:
            z = z + torch.randn_like(z) * self.emb_noise_sd
        if self.rnn is not None:
            z = self.rnn(z)[0]
        return self.dec_out(z)

    def forward(self, x, lengths):
        z, out_len = self.encode(x, lengths)
        return self.decode(z), out_len

    def get_embeddings(self):
        return torch.cat(self._cache["hs"], dim=-1), self._cache["out_len"]

    # ---------------- split-brain targets ----------------
    def window_means(self, xs, n_steps):
        """(B, T, C) -> (B, T', C): mean of xs over the part of each stem window used as target."""
        k, s = self.kernel, self.stride
        if self.target_window == "center":
            off, win = (k - s) // 2, s
        elif self.target_window == "full":
            off, win = 0, k
        else:
            raise ValueError(f"unknown target_window '{self.target_window}'")
        m = F.avg_pool1d(xs.float().transpose(1, 2)[..., off:], kernel_size=win, stride=s)
        return m[..., :n_steps].transpose(1, 2)

    def quantize(self, m, idx):
        edges = self.bin_edges.index_select(0, idx)  # (C_v, K-1)
        return (m[..., idx][..., None] > edges).sum(-1)  # (B, T', C_v) in [0, K-1]

    @torch.no_grad()
    def fit_bins(self, trials, max_samples=200000, seed=0):
        """Per-channel quantile bin edges + marginal bin entropy from a list of (T, C) training trials.

        Returns (number of windows used, number of constant channels excluded from the T loss).
        """
        device = self.bin_edges.device
        k, s = self.kernel, self.stride
        trials = [x for x in trials if x.shape[0] >= k]
        n_valid = [max(int((x.shape[0] - k) / s), 1) for x in trials]
        keep = min(1.0, max_samples / max(sum(n_valid), 1))
        g = torch.Generator().manual_seed(seed)
        frames = []
        for x, nv in zip(trials, n_valid):  # same windows / valid steps as in training
            m = self.window_means(self.smoother(x.to(device)[None].float()), (x.shape[0] - k) // s + 1)
            m = m[0, :nv].cpu()
            if keep < 1.0:
                m = m[torch.rand(m.shape[0], generator=g) < keep]
            frames.append(m)
        frames = torch.cat(frames, 0)
        q = torch.linspace(0, 1, self.n_bins + 1)[1:-1]
        edges, ent = [], []
        for c in range(frames.shape[1]):
            e = torch.quantile(frames[:, c], q)
            cls = (frames[:, c, None] > e).sum(-1)
            p = torch.bincount(cls, minlength=self.n_bins).float()
            p = p / p.sum()
            edges.append(e)
            ent.append(-(p * p.clamp_min(1e-12).log()).sum())
        ent = torch.stack(ent)
        self.bin_edges.copy_(torch.stack(edges))
        self.bin_entropy.copy_(ent)
        self.bin_weight.copy_((ent > 1e-3).float())
        self.bins_fitted.fill_(True)
        return frames.shape[0], int((ent <= 1e-3).sum())

    def _masked_mean(self, v, mask):
        mask = mask.float()
        return (v * mask).sum() / mask.sum().clamp(min=1.0)

    def _ce(self, logits, target, weight, mask):
        """Cross-entropy averaged over the (non-constant) channels of a view and the valid steps."""
        B, T, _ = logits.shape
        logits = logits.float().view(B, T, -1, self.n_bins)
        ce = F.cross_entropy(logits.reshape(-1, self.n_bins), target.reshape(-1), reduction="none").view(B, T, -1)
        return self._masked_mean((ce * weight).sum(-1) / weight.sum().clamp(min=1.0), mask)

    def _marginal_entropy(self, idx):
        w = self.bin_weight[idx]
        return (self.bin_entropy[idx] * w).sum() / w.sum().clamp(min=1.0)

    def _cos_loss(self, pred, target, mask):
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        return self._masked_mean(2.0 - 2.0 * (pred * target).sum(-1), mask)

    def aux_losses(self, x_clean=None):
        """Split-brain losses for the last encode() call.

        x_clean: un-noised input, used for the targets when input noise is on.
        Returns a dict with 'T' and/or 'L' losses plus a few diagnostics.
        """
        c = self._cache
        h_a, h_b = c["hs"]
        out_len = c["out_len"]
        n_steps = h_a.shape[1]
        mask = valid_mask(out_len, n_steps)
        x_tgt = c["x"] if x_clean is None else x_clean
        out = {}

        if self.use_T:
            assert bool(self.bins_fitted), "call fit_bins() before training with aux T/M"
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                # float32, exactly as in fit_bins
                m = self.window_means(self.smoother(x_tgt.float()), n_steps)
                t_a, t_b = self.quantize(m, self.idx_A), self.quantize(m, self.idx_B)
            ce_ab = self._ce(self.head_AB(h_a), t_b, self.bin_weight[self.idx_B], mask)
            ce_ba = self._ce(self.head_BA(h_b), t_a, self.bin_weight[self.idx_A], mask)
            out["T"] = 0.5 * (ce_ab + ce_ba)
            # nats per channel gained over the marginal (a lower bound on the cross-view information);
            # <= 0 means the other view is not predicted better than its own histogram
            out["T_gain"] = 0.5 * ((self._marginal_entropy(self.idx_B) - ce_ab)
                                   + (self._marginal_entropy(self.idx_A) - ce_ba)).detach()

        if self.use_L:
            with torch.no_grad():
                tgt_encoders = self.ema_encoders if self.use_ema else self.encoders
                xs = c["xs"] if x_clean is None else self.smoother(x_clean)
                v_a, v_b = self._views(xs)
                g_a, g_b = tgt_encoders[0](v_a, out_len), tgt_encoders[1](v_b, out_len)
            out["L"] = 0.5 * (self._cos_loss(self.pred_AB(h_a), g_b, mask) + self._cos_loss(self.pred_BA(h_b), g_a, mask))

        with torch.no_grad():  # collapse monitor (both views): ~1 healthy, -> 0 collapsed
            stds = [F.normalize(h.float()[mask], dim=-1).std(0).mean() * math.sqrt(h.shape[-1]) for h in (h_a, h_b)]
            out["emb_std"] = 0.5 * (stds[0] + stds[1])
        return out

    @torch.no_grad()
    def update_ema(self, decay):
        if not hasattr(self, "ema_encoders"):
            return
        for p_t, p_o in zip(self.ema_encoders.parameters(), self.encoders.parameters()):
            p_t.mul_(decay).add_(p_o.detach(), alpha=1.0 - decay)


def build_model(args):
    return SplitBrainNet(
        n_channels=192, n_classes=32,
        split=args["split"], split_seed=args["split_seed"],
        arch=args["arch"], d_model=args["d_model"], emb_dim=args["emb_dim"],
        enc_layers=args["enc_layers"], enc_dropout=args["enc_dropout"], nhead=args["nhead"],
        conv_kernel=args["conv_kernel"],
        kernel=args["kernel"], stride=args["stride"], smooth_sigma=args["smooth_sigma"],
        decoder=args["decoder"], hidden=args["hidden"], layers=args["layers"],
        dropout=args["dropout"], bidir=not args["no_bidir"],
        aux=args["aux"], n_bins=args["n_bins"], target_window=args["target_window"],
        head_hidden=args["head_hidden"], use_ema=not args["no_ema"],
    )
