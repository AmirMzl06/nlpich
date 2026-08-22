from utils.criterion import InfoNCE
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from utils.constants import CEBRA_DIR, NUM_TRAIN_DAYS
import sys
import numpy as np
from utils.macorn import Offset36_multi
from utils.sample_positive_negative import get_batch, get_batch_all, get_batch_all_flat
from tacorn_utils.fnet import FNetEncoderLayer
sys.path.append(str(CEBRA_DIR))
import cebra
import cebra.models.layers as cebra_layers
from edit_distance import SequenceMatcher
from utils.mlp import TwoLayerMLP
import math
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict
import random
import time
from tacorn_utils.perturb_utils import UnitDropOutPerturbation
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as dist_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch_brain.optim import SparseLamb
from utils.classifier import ClassifierHead

from utils.nlp_sampler import SameSessionBatchSamplerDDP

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def is_main_process():
    return get_rank() == 0


def setup_ddp():
    """
    Initialize torch.distributed from torchrun env variables.
    """
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    # LOCAL_RANK is set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # If numpy is used anywhere in the stack, seed it too
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def gather_with_grad(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    parts = dist_all_gather(x)  # returns list, keeps grad for local part
    return torch.cat(parts, dim=0)

class SameSessionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        session_to_indices = defaultdict(list)
        for idx, (_, _, d) in enumerate(dataset.items):
            session_to_indices[d].append(idx)

        self.batches = []
        for d, indices in session_to_indices.items():
            if shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) < batch_size and drop_last:
                    continue
                self.batches.append(batch)

        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


CHARS = [
    '>', ',', '?', '~', "'",
    'a', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
]
BLANK_TOKEN = "<BLANK>"


class Charset:
    def __init__(self, symbols: List[str]):
        # index 0 reserved for CTC blank
        self.idx2sym = [BLANK_TOKEN] + symbols
        self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
        self.sym2idx[BLANK_TOKEN] = 0

    @property
    def num_classes(self) -> int:
        return len(self.idx2sym)

    def text_to_int(self, text: str) -> List[int]:
        return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

    def int_to_text(self, ids: List[int]) -> str:
        return "".join(self.idx2sym[i] for i in ids if i != 0)


charset = Charset(CHARS)


# class TrialsDataset(Dataset):
#     """
#     Wraps a list of (features, transcript) pairs.
#     - features: FloatTensor of shape (T, F)
#     - transcript: str
#     """
#     def __init__(self, items: List[Tuple[torch.FloatTensor, str]]):
#         super().__init__()
#         self.items = items
#
#     def __len__(self):
#         return len(self.items)
#
#     def __getitem__(self, idx):
#         x, y = self.items[idx]
#         assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.dim() == 2, \
#             "Each trial must be FloatTensor of shape (T, F)."
#         assert isinstance(y, str), "Transcript must be a string."
#         return x, y

class TrialsDataset(Dataset):
    """
    items: List of (features, transcript, session_id)
    """

    def __init__(self, items: List[Tuple[torch.FloatTensor, str, int]]):
        super().__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, d = self.items[idx]

        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.dim() == 2
        assert isinstance(y, str)
        assert isinstance(d, int)
        y = torch.tensor(charset.text_to_int(y), dtype=torch.long)

        return x, y, d


class SpeechDataset(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, idx):
       x, y, z = self.items[idx]
       return x, y, z
    
    def __len__(self):
        return len(self.items)
    
def ctc_collate(batch: List[Tuple[torch.Tensor, str, int]]):
    """
    Returns:
      x_pad: (B, T_max, F)
      input_lengths: (B,)
      targets: (sum_L,)
      target_lengths: (B,)
      transcripts: list[str]
      sessions: (B,)
    """
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
        # if T < T_max:
        #     pad_len = T_max - T
        #     last_frame = x[-1].unsqueeze(0)
        #     x_pad[i, T:] = last_frame.expand(pad_len, -1)

    target_seqs = list(ys)
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    targets = torch.cat(target_seqs) if len(target_seqs) else torch.tensor([], dtype=torch.long)

    sessions = torch.tensor(ds, dtype=torch.long)

    return x_pad, input_lengths, targets, target_lengths, list(ys), sessions


# def ctc_collate(batch: List[Tuple[torch.Tensor, str]]):
#     """
#     Returns:
#       x_pad: (B, T_max, F)
#       input_lengths: (B,)
#       targets: (sum_L,)
#       target_lengths: (B,)
#       transcripts: list[str]
#     """
#     xs, ys = zip(*batch)
#     B = len(xs)
#     feat_dim = xs[0].shape[-1]
#     input_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
#     T_max = int(input_lengths.max().item())
#
#     x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
#     for i, x in enumerate(xs):
#         T = x.shape[0]
#         x_pad[i, :T] = x
#
#     target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
#     target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    # targets = torch.cat(target_seqs) if len(target_seqs) else torch.tensor([], dtype=torch.long)
#
#     return x_pad, input_lengths, targets, target_lengths, list(ys)


# -----------------------------
# Model
# # -----------------------------

# def fnet_forward(fnet, x, input_lengths):
#         out = torch.zeros((x.shape[0], x.shape[1], fnet.hidden), device=x.device)
#         for i in range(x.shape[0]):
#             length = input_lengths[i]
#             out[i, :length] = fnet(x[i, :length].unsqueeze(0)).squeeze(0)
        
#         return out
import torch
import math
import torch.nn.functional as F


def fnet_forward(
    fnet,
    x,
    input_lengths,
    window_size=200,
    stride=100,
    prediction_start=50,
):
    device = x.device
    B, T, N = x.shape

    prediction_end = prediction_start + stride
    hidden = fnet.hidden

    out = torch.zeros((B, T, hidden), device=device)

    for b in range(B):

        length = input_lengths[b]
        seq = x[b, :length]

        # ---------- padding ----------
        first_end = prediction_end - prediction_start

        n_strides = math.ceil((length - first_end) / stride)

        pad_end = window_size + n_strides * stride - prediction_start - length
        pad_front = prediction_start

        seq = F.pad(seq, (0, 0, pad_front, pad_end))

        # ---------- sliding windows ----------
        windows = []
        starts = []

        max_start = seq.shape[0] - window_size

        s = 0
        while s <= max_start:
            windows.append(seq[s : s + window_size])
            starts.append(s)
            s += stride

        windows = torch.stack(windows)  # (W, 200, N)

        # ---------- model forward ----------
        preds = fnet(windows)  # (W, 200, hidden)

        preds = preds[:, prediction_start:prediction_end]  # (W, 100, hidden)

        preds = preds.reshape(-1, hidden)  # (W*100, hidden)

        preds = preds[:length]

        out[b, :length] = preds

    return out, input_lengths


class CTCEncoder(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, feat_dim: int, hidden: int, nLayers: int, num_classes: int, bidir=True, linear=True, dr=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=nLayers,
            bidirectional=bidir,
            batch_first=True,
            dropout=dr
        )
        self.act = nn.Softsign()
        # self.dec = FNetEncoderLayer(feat_dim , hidden)
        # self.bn = nn.BatchNorm1d(feat_dim)
        # self.out = TwoLayerMLP(feat_dim, 64, num_classes)
        # self.out = nn.Linear(feat_dim, num_classes)
        
        self.out = TwoLayerMLP((2 if bidir else 1) * hidden, 64, num_classes) 
        # self.out = nn.Linear(hidden * (2 if bidir else 1), num_classes)
        # self.out = ClassifierHead(hidden, num_classes, d_hidden=64)
    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        # print(x.shape, self.bn)
        # x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.act(x)
        x, _ = self.lstm(x)  # (B, T, 2H)
        # print(self.lstm)
        # x, input_lengths = fnet_forward(self.dec, x, input_lengths)
        logits = self.out(x)  # (B, T, C)
        # print(logits.shape, input_lengths.max())
        return logits.transpose(0, 1), input_lengths  # (T, B, C), (B,)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class CTCEncoderTransformer(nn.Module):
    def __init__(self, feat_dim, hidden,num_classes , nhead=4, nLayers=3):
        super().__init__()

        self.proj = nn.Linear(feat_dim, hidden)
        self.pos = PositionalEncoding(hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=4*hidden,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nLayers)

        self.classifier = TwoLayerMLP(hidden, 64, num_classes)

    def forward(self, x, input_lengths):
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        self.embs = x
        logits = self.classifier(x)    # B T V
        # print(input_lengths.shape, x.shape)
        return logits.transpose(0, 1), input_lengths




def ctc_greedy_decode(log_probs: torch.Tensor,input_lengths, blank_id: int = 0, ) -> List[List[int]]:
    """
    log_probs: (T, B, C) after log_softmax
    Returns list of sequences (per batch) with repeats and blanks collapsed.
    """
    T, B, C = log_probs.shape
    argmax = log_probs.argmax(dim=-1)  # (T, B)
    results: List[List[int]] = []
    for b in range(B):
        prev = blank_id
        out = []
        for t in range(input_lengths[b]):
            k = argmax[t, b].item()
            if k != blank_id and k != prev:
                out.append(k)
            prev = k
        results.append(out)
    return results


def _levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance (costs: ins=1, del=1, sub=1)."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost  # substitution
            )
            prev = cur
    return dp[m]


def cer_score(refs: List[str], hyps: List[str]) -> float:
    """
    Corpus-level CER = sum(edit_distance) / sum(#ref_chars). If all refs empty -> 0.0
    """
    assert len(refs) == len(hyps)
    total_edits, total_chars = 0, 0
    for r, h in zip(refs, hyps):
        total_edits += _levenshtein(r, h)
        total_chars += len(r)
    return (total_edits / total_chars) if total_chars > 0 else 0.0


class CTCModel:
    def __init__(self,
                 feat_dim: int,
                 hidden: int = 256,
                 num_layers: int = 2,
                 lr: float = 2e-3,
                 weight_decay: float = 1e-2,
                 blank_id: int = 0,
                 device: Optional[str] = None,
                 model=None,
                 tacorn=False,
                 multi=False,
                 linear=True,
                 cebra_model=None,
                 kernel_len=1,
                 stride_len=1,
                 two_ops=False,
                 multi_decs=False,
                 ddp=False,
                 macorn=False,
                 lamb=False, speech=False,
                 use_checkpoint=False
                 ):
        self.use_checkpoint = use_checkpoint
        self.local_rank = setup_ddp() if ddp else 0
        self.device = torch.device(device or (f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"))
        self.blank_id = blank_id
        self.lamb = lamb
        # self.net = CTCEncoderTransformer(
        #     feat_dim=feat_dim,
        #     hidden=hidden,
        #     nLayers=num_layers,
        #     num_classes=charset.num_classes,
        # )
        self.net = CTCEncoder(
            feat_dim=feat_dim,
            hidden=hidden,
            nLayers=num_layers,
            num_classes=charset.num_classes if not speech else 41,
        )
        self.tacorn = tacorn and two_ops
        self.multi = False
        self.cebra = False
        if macorn:
            self.multi=True
            del self.net
            self.net = MACORN_DECODER(model)
            self.cebra=True
        
        elif model is not None and not tacorn:
            self.cebra = True
            self.net = CTCEncoderCEBRA(model, self.net)
        
        self.net = self.net.to(self.device)
        self.lr = lr
        # local_rank = setup_ddp()
        self.ddp = ddp
        if ddp:
            
            self.net = DDP(self.net, device_ids=[self.local_rank], output_device=self.local_rank)
        if not tacorn or not two_ops:
            print('1op')
            self.optimizers = [torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)]
            if tacorn and lamb:
                self.optimizers = []
                special_emb_params = list(self.net.model.unit_emb.parameters()) + list(self.net.model.session_emb.parameters())
                remaining_params = [
                    p
                    for n, p in self.net.named_parameters()
                    if "unit_emb" not in n and "session_emb" not in n
                ]
                optimizer = SparseLamb(
                    [
                        {"params": special_emb_params, "sparse": True, "weight_decay": 0.0},
                        {"params": remaining_params},
                    ],
                    lr=lr,
                    weight_decay=weight_decay,
                )
                self.optimizers.append(optimizer)
                print('hi')
        else:
            
                
            optimizer1 = torch.optim.AdamW((self.net.module.ctc.parameters()) if ddp else (list(self.net.ctc.parameters())), lr=lr, weight_decay=weight_decay)
            optimizer2 = torch.optim.AdamW((list(self.net.module.model.parameters())) if ddp else (list(self.net.model.parameters()) ), lr=lr, weight_decay=weight_decay)
            self.optimizers = [
                optimizer1, optimizer2
            ]
        self.criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True, reduction='mean')

    def _forward_logits(self, x_pad: torch.Tensor, input_lengths: torch.Tensor, days=None):
        if self.use_checkpoint:
            vals = (x_pad, input_lengths, )
            def forward_blocks():
                logits, out_lengths = self.net(*vals)
                return torch.log_softmax(logits, dim=-1), out_lengths

            
            log_probs, out_lengths = checkpoint(forward_blocks, use_reentrant=False)

        else:
            logits, out_lengths = self.net(x_pad, input_lengths) if not self.multi else self.net(x_pad, input_lengths, days)
            log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, out_lengths
    

    @staticmethod
    def next_batch(dataloader, iterator):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        return batch, iterator

        

    def predict_text(self, trials: List[torch.FloatTensor], batch_size: int = 16) -> List[str]:
        """
        trials: list of FloatTensor (T, F)
        returns list of decoded strings (greedy)
        """
        ds = TrialsDataset([(t, "", 0) for t in trials])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate)
        hyps: List[str] = []
        self.net.eval()
        with torch.no_grad():
            for x_pad, input_lengths, _, _, _ in dl:
                x_pad = x_pad.to(self.device)
                input_lengths = input_lengths.to(self.device)
                log_probs, _ = self._forward_logits(x_pad, input_lengths)
                ids_batch = ctc_greedy_decode(log_probs.cpu(), blank_id=self.blank_id)
                for ids in ids_batch:
                    hyps.append(charset.int_to_text(ids))
        return hyps

    def cal_blank_loss(self, out, out_lengths):
        out = out[:, :, self.blank_id]
        B, T = out.shape
        mask = torch.arange(T, device=out.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        return out[mask].sum()

    def fit(self,
            train_loaders: List[Tuple[torch.FloatTensor, str, int]],
            val_loaders: Optional[List[Tuple[torch.FloatTensor, str, int]]] = None,
            name="",
            iters: int = 10,
            batch_size: int = 16,
            num_workers: int = 0,
            grad_clip: float = 1.0,
            adv=False,
            eps=0.01,
            cont_batch=2048,
            cont_offset=1,
            cont_temp: float = 0.4,
            ctc_alpha=1.0,
            x_all=None,
            x_all_lengths=None,
            cont_only=False,
            steps=10, norm='linf', steps_between = 10,
            speech_dl=None,
            ) -> Dict[str, List[float]]:
        alpha = eps * 2 / steps
        """
        Trains the model on (trial, transcript) pairs.
        Returns a history dict with train_loss, val_loss, val_cer (if val provided).
        """
        warmup_steps = iters // 40
        plateau_steps = 3 * warmup_steps
        total_steps = iters
        iterators = [iter(dl) for dl in train_loaders]
        n_loaders = len(train_loaders)
        

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            elif step < plateau_steps:
                return 1.0

            else:
                decay_len = max(1, total_steps - plateau_steps)
                progress = float(step - plateau_steps) / decay_len
                return 0.1 + 0.5 * (1.0 - 0.1) * (1.0 + math.cos(math.pi * progress))

        self.net = self.net.to(self.device)
        contrastive_criterion = InfoNCE(cont_temp)
        
        
        
        schedulers = []
        
        
        
        if not self.lamb:
            scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizers[0], step_size=1 * (15000) , gamma=0.3)
        else:
            
            scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizers[0],
                max_lr=self.lr,
                total_steps= iters,
                pct_start=0.5,
                anneal_strategy="cos",
                div_factor=1,
            )
        
        schedulers.append(scheduler1)
       # scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        world_size = dist.get_world_size() if self.ddp else 1

        
        
        
        

        

        hist = {"train_loss": [], "val_loss": [], "val_cer": [], "train_inf_loss":[]}
        saved_after = False
        self.net.train()
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        best_val_cer = 5.
        inf_losses = 0
        break_whole=False
        prog_bar = trange(1, iters + 1)

        for ep in prog_bar:
            if break_whole: break
            
            opt_index = 0
            if self.tacorn:
                opt_index = 0 if (ep % steps_between != 0) else 1
            self.optimizer = self.optimizers[opt_index]
            scheduler = schedulers[opt_index]
            # --- train ---
            self.net.train()
            total_loss, n_items = 0.0, 0
            total_backward_loss = 0.0
            c_loss = 0
            if self.tacorn:
                m = self.net.module if self.ddp else self.net
                if opt_index == 0:
                    for p in m.model.parameters():
                        p.requires_grad_(False)
                    for p in m.ctc.parameters():
                        p.requires_grad_(True)
                else:
                    for p in m.ctc.parameters():
                        p.requires_grad_(False)
                    for p in m.model.parameters():
                        p.requires_grad_(True)
            
            for dl_idx in range(n_loaders):
                loader, curr_iter = train_loaders[dl_idx], iterators[dl_idx]
                batch, next_iter = self.next_batch(loader, curr_iter)
                iterators[dl_idx] = next_iter
            
                x_pad, input_lengths, targets, target_lengths, _, days = batch
                torch.cuda.empty_cache()
                    
                    


                    
                x_pad = x_pad.to(self.device)
                input_lengths = input_lengths.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                # div_by =  1.
                # x_pad += (torch.randn(x_pad.shape, device=self.device) * (0.3 / div_by))

                # x_pad += (
                #             torch.randn([x_pad.shape[0], 1, x_pad.shape[2]], device=self.device)
                #             * 0.15 / div_by
                #     )

                bsz = x_pad.size(0)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        if not(x_all is not None and cont_only):
                            log_probs, out_lengths = self._forward_logits(x_pad, input_lengths, days=days)
                        loss = 0.0 if cont_only else (self.criterion(log_probs, targets, out_lengths, target_lengths) * ctc_alpha)
                        if self.cebra:
                            if x_all is None:
                                embeddings = self.net.get_embeddings() if not self.ddp else self.net.module.get_embeddings()
                                reference, positive, negative, ref_batch_idx, ref_time_idx, pos_time_idx, neg_batch_idx, neg_time_idx = get_batch(
                                    embeddings, out_lengths, cont_batch, cont_offset)
                            else:
                                reference, positive, negative = get_batch_all_flat(
                                    x_pad, input_lengths, cont_batch, cont_offset, x_all, x_all_lengths, self.net.offset.left, self.net.offset.right, two_negs=False)
                                
                                negative = self.net.only_cebra(negative )
                                positive = self.net.only_cebra(positive)
                                reference = self.net.only_cebra(reference)
                                
                            reference = gather_with_grad(reference)
                            positive = gather_with_grad(positive)
                            negative = gather_with_grad(negative)
                            loss_contrastive = contrastive_criterion.forward(reference, positive, negative)[0]
                            # if loss_contrastive.item() < 3.2: self.cebra = False
                            c_loss += loss_contrastive.item() * bsz
                            loss = loss + loss_contrastive
                            # total_backward_loss = total_backward_loss + loss / n_loaders
                            # self.optimizer.zero_grad(set_to_none=True)
                            total_loss += loss.item() * bsz
                            n_items += bsz
                self.optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                        print("Non-finite loss encountered. Skipping step.")
                        
                        inf_losses += 1
                        if inf_losses > 10:
                            break_whole = True
                        break
                        continue
                inf_losses = 0
                scaler.scale(loss).backward()
                # scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                scaler.step(self.optimizer)
                scaler.update()

            # self.optimizer.zero_grad(set_to_none=True)
            # if not torch.isfinite(total_backward_loss):
            #         print("Non-finite loss encountered. Skipping step.")
                    
            #         inf_losses += 1
            #         if inf_losses > 10:
            #             break_whole = True
            #         break
            #         continue
            # inf_losses = 0
               
                                    
                
            # scaler.scale(total_backward_loss).backward()
            # scaler.unscale_(self.optimizer)
            # nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            # scaler.step(self.optimizer)
            # scaler.update()
                        
            
            
            scheduler.step()


            # self.optimizer.zero_grad(set_to_none=True)
            # if not torch.isfinite(total_backward_loss):
            #             print("Non-finite loss encountered. Skipping step.")
            #             self.optimizer.zero_grad(set_to_none=True)
            #             inf_losses += 1
            #             if inf_losses > 10:
            #                 break_whole = True
            #                 break
            #             continue
            # inf_losses = 0
            # scaler.scale(total_backward_loss).backward()
            #         # n = next(self.net.model.session_emb.parameters())

            # if grad_clip is not None:
            #             torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            #         # print(n.grad)
                    
            # scaler.step(self.optimizer)
            # scaler.update()

                
            c_loss /= max(1, n_items)
            tr_loss = total_loss / max(1, n_items)
            hist["train_loss"].append(tr_loss)
            hist["train_inf_loss"].append(c_loss)
            
            
            ctc_loss = (tr_loss - c_loss) / ctc_alpha
            

            
            
                

            if val_loaders is not None and ((ep+1) % 100 == 0) and self.local_rank == 0:
                torch.cuda.empty_cache()
                days_scores = {}
                self.net.eval()
                va_cers = []
                va_losses = []
                
                
                with torch.no_grad():
                    
                    for val_dl in val_loaders:
                        edit_distance = 0
                        length = 0
                        tot_loss, n = 0.0, 0
                        all_refs, all_hyps = [], []
                        all_days = []
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                            for x_pad, input_lengths, targets, target_lengths, refs, days in val_dl:
                                x_pad = x_pad.to(self.device)
                                input_lengths = input_lengths.to(self.device)
                                targets = targets.to(self.device)
                                target_lengths = target_lengths.to(self.device)

                                log_probs, out_lengths = self._forward_logits(x_pad, input_lengths, days=days)
                                loss = self.criterion(log_probs, targets, out_lengths, target_lengths)

                                bsz = x_pad.size(0)
                                tot_loss += loss.item() * bsz
                                n += bsz
                                all_days += days.detach().cpu().numpy().tolist()
                                ids_batch = ctc_greedy_decode(log_probs.cpu(),out_lengths.cpu(), blank_id=self.blank_id)

                                

                                all_refs.extend(refs)
                                all_hyps.extend(list(ids_batch))

                    

                        for r, h, day in zip(all_refs, all_hyps, all_days):
                            if day not in days_scores:
                                days_scores[day] = []
                            s = SequenceMatcher(a=r, b=h)
                            edit_distance += s.distance()
                            length += len(r)
                            days_scores[day].append(s.distance() / len(r))


                        va_loss = tot_loss / max(1, n)
                        va_cer = edit_distance / max(length, 1)
                        va_losses.append(va_loss)
                        va_cers.append(va_cer)
                
                hist["val_loss"].append(va_losses)
                hist["val_cer"].append(va_cers)
                va_cer = np.mean(va_cers)
                if va_cer < best_val_cer:
                    best_val_cer = va_cer
                    
                    self.save_state(f'{name}.pth')
                with open(f'{name}.txt', 'w') as f:
                    f.write(str(hist))
                output = f"[Iter {ep}] train_loss={tr_loss:.3f} | val_losses={[f'{v:.2f}' for v in va_losses]} | val_CERs={[f'{v:.2f}' for v in va_cers]}"
                val_iter = True
            else:
                val_iter = False
                output = f"[Iter {ep}] train_loss={tr_loss:.3f} c_loss {c_loss:.3f} ctc_loss {ctc_loss:.3f} {name} "
            # print(output)
            prog_bar.set_description_str(output)
            if ep % 10 == 0 or val_iter:
                with open(f'{name}.log', 'a') as f:
                    f.write(output)
                    f.write("\n")
            
            if ep % 500 == 0:
                time.sleep(30)
            # if opt_index==1:
                # time.sleep(30)
            # time.sleep(30)
            # if ep % 5 == 0: time.sleep(80)
       
        return hist

    def save_state(self, path):
        checkpoint = {
            'net':self.net.state_dict(),
            'opt1':self.optimizers[0].state_dict()
        }
        if len(self.optimizers) > 1:
            checkpoint['opt2'] = self.optimizers[1].state_dict()
        torch.save(checkpoint, path)
    
    def load_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        opts = len(self.optimizers)
        for i in range(opts):
            self.optimizers[i].load_state_dict(checkpoint[f'opt{i+1}'])


    def cer(self,
            items: List[Tuple[torch.FloatTensor, str]],
            batch_size: int = 16,
            num_workers: int = 0,
            all_days: bool = False, speech=False):
        ds = SpeechDataset(items) if speech else TrialsDataset(items)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=ctc_collate, num_workers=num_workers, pin_memory=True
        )
        self.net.eval()
        cer_list = []

        with torch.no_grad():
            distance = 0
            length = 0
            for x_pad, input_lengths, _, _, refs, days in tqdm(dl):
                x_pad = x_pad.to(self.device)
                input_lengths = input_lengths.to(self.device)
                log_probs, out_lengths = self._forward_logits(x_pad, input_lengths, days=days)
                ids_batch = ctc_greedy_decode(log_probs.cpu(),out_lengths.cpu(), blank_id=self.blank_id)

                            


                for ids, ref in zip(ids_batch, refs):
                    

                    s = SequenceMatcher(a=ref, b=ids)
                    cer = s.distance() / len(ref)
                    distance += cer * len(ref)
                    length += len(ref)
                    cer_list.append(cer)

        cer = distance / length

        if all_days:
            return cer_list, cer
        else:
            return cer


from tacorn_utils.tokenizers import  nlp_tok, tokenize
# from tacorn_utils.SPINT import tokenize
import torch.nn.functional as F
import math

kernel_len = 1
stride_len = 10
feat_dim = 128
MAX_LENGTH = 2940 // 2
T = 0.01
from torch.utils.checkpoint import checkpoint


def expand_index(index: torch.Tensor, offset, length: int) -> torch.Tensor:
    """

    Args:
        index: A one-dimensional tensor of type long containing indices
            to select from the dataset.

    Returns:
        An expanded index of shape ``(len(index), len(self.offset))`` where
        the elements will be
        ``expanded_index[i,j] = index[i] + j - self.offset.left`` for all ``j``
        in ``range(0, len(self.offset))``.

    Note:
        Requires the :py:attr:`offset` to be set.
    """
    off = torch.arange(-offset.left,
                       offset.right,
                       device=index.device)

    index = torch.clamp(index, offset.left,
                        length - offset.right)

    return index[:, None] + off[None, :]





class CTCEncoderCEBRA(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, model, ctc):
        super().__init__()
        self.model = model
        self.offset = model.get_offset()
        self.unfolder = torch.nn.Unfold(
            (1, 1), dilation=1, padding=0, stride=2
        )
        self.kernelLen = 1
        self.strideLen = 2
        self.ctc = ctc
        self.norm = cebra_layers._Norm()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        x = F.pad(x, (0, 0, self.offset.left, self.offset.right - 1), mode="replicate")
        x = x.permute(0, 2, 1)

        self.embeds = self.model(x).permute(0, 2, 1)  # (B, T, H)
        x = self.embeds
        self.embeds = self.norm(self.embeds.permute(0, 2, 1)).permute(0, 2, 1)
        return self.ctc(x, input_lengths)

    def get_embeddings(self):
        return self.embeds


class MACORN_DECODER(nn.Module):
    def __init__(self, model: Offset36_multi):
        super().__init__()
        self.model = model
        self.unfolder = torch.nn.Unfold(
            (1, 1), dilation=1, padding=0, stride=2
        )
        self.ctc = nn.ModuleList()
        for _ in range(2):
            self.ctc.append(CTCEncoder(self.model.out_dim, 128, 2, charset.num_classes))
            
        self.ctc.append(CTCEncoder(self.model.out_dim, 256, 3, 41))
        self.norm = cebra_layers._Norm()
        self.offset = model.get_offset()
        
    
        
    
    def forward(self, x, input_lengths, days):
        session = str(days[0].item())
        x = F.pad(x, (0, 0, self.offset.left, self.offset.right - 1), mode="replicate")
        x = x.permute(0, 2, 1)
        
        
        x = self.embeds = self.model(x, session).permute(0, 2, 1) 
        self.embeds = self.norm(self.embeds.permute(0, 2, 1)).permute(0, 2, 1)
        return self.ctc[days[0]](x, input_lengths)
    
    def get_embeddings(self):
        return self.embeds