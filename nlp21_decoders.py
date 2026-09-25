"""Frozen-feature probes: linear/GRU CTC and autoregressive attention CE.

Attention CE uses teacher forcing for training ONLY. CER always uses free
generation with BOS/EOS, never target text or target lengths. CTC is never
applied to the Jigsaw encoder. No phoneme metric is claimed for character data.
"""
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from nlp21_data import (PAD, BOS, EOS, N_CLASSES, AR_CLASSES, CHARS,
                       decode_ids, edit_distance, ctc_collapse, corpus_metrics, dump_json)

DECODERS = ("linear_ctc", "gru_ctc", "attention_ce")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_trials(trials):
    lengths = torch.tensor([len(t.x) for t in trials], dtype=torch.long)
    y_lengths = torch.tensor([len(t.targets) for t in trials], dtype=torch.long)
    x = torch.empty(len(trials), int(lengths.max()), trials[0].x.shape[1])
    y = torch.zeros(len(trials), int(y_lengths.max()), dtype=torch.long)
    for i, trial in enumerate(trials):
        n, m = len(trial.x), len(trial.targets)
        x[i, :n] = torch.from_numpy(trial.x)
        x[i, n:] = x[i, n - 1:n]
        y[i, :m] = torch.tensor(trial.targets, dtype=torch.long)
    return x, lengths, y, y_lengths, trials


def make_loader(trials, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(trials, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_trials, generator=generator,
                      pin_memory=torch.cuda.is_available())


def validate_ctc_lengths(trials):
    for t in trials:
        repeated = sum(a == b for a, b in zip(t.targets, t.targets[1:]))
        required = len(t.targets) + repeated
        if len(t.x) < required:
            raise ValueError(
                f"CTC infeasible for {t.uid}: {len(t.x)} output steps but needs "
                f"{required} (=characters + adjacent repeats). Reduce --decoder-stride "
                "(try 1). No trial is silently skipped and infinite losses are not zeroed.")


class LinearCTC(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.readout = nn.Linear(input_dim, N_CLASSES)

    def forward(self, x, lengths):
        return self.readout(x)


class TemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden),
                                              nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        self.output_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.input_projection(x), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed, hidden = self.gru(packed)
        memory, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        final = torch.cat((hidden[-2], hidden[-1]), -1) if self.bidirectional else hidden[-1]
        return memory, final


class GRUCTC(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.temporal = TemporalGRU(input_dim, hidden, layers, dropout, bidirectional)
        self.readout = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.temporal.output_dim, N_CLASSES))

    def forward(self, x, lengths):
        memory, _ = self.temporal(x, lengths)
        return self.readout(memory)


class AttentionCE(nn.Module):
    """Additive attention + GRUCell; train next characters, no alignment labels."""
    def __init__(self, input_dim, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.temporal = TemporalGRU(input_dim, hidden, layers, dropout, bidirectional)
        memory_dim = self.temporal.output_dim
        self.embedding = nn.Embedding(AR_CLASSES, hidden, padding_idx=PAD)
        self.key = nn.Linear(memory_dim, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)
        self.initialize = nn.Linear(memory_dim, hidden)
        self.cell = nn.GRUCell(hidden + memory_dim, hidden)
        self.output = nn.Linear(hidden + memory_dim, AR_CLASSES)
        self.dropout = nn.Dropout(dropout)

    def prepare(self, x, lengths):
        memory, final = self.temporal(x, lengths)
        mask = torch.arange(memory.shape[1], device=x.device)[None] >= lengths.to(x.device)[:, None]
        return memory, self.key(memory), mask, torch.tanh(self.initialize(final))

    def step(self, token, hidden, memory, keys, mask):
        energy = self.energy(torch.tanh(keys + self.query(hidden)[:, None])).squeeze(-1)
        weights = energy.masked_fill(mask, float("-inf")).softmax(-1)
        context = torch.bmm(weights[:, None], memory).squeeze(1)
        hidden = self.cell(torch.cat((self.dropout(self.embedding(token)), context), -1), hidden)
        logits = self.output(self.dropout(torch.cat((hidden, context), -1)))
        # PAD and BOS are input-only symbols; EOS is a genuine output class.
        invalid = torch.zeros(AR_CLASSES, dtype=torch.bool, device=logits.device)
        invalid[PAD] = invalid[BOS] = True
        return logits.masked_fill(invalid, -1e4), hidden

    def loss(self, x, lengths, y, y_lengths):
        memory, keys, mask, hidden = self.prepare(x, lengths)
        target = torch.zeros(len(y), y.shape[1] + 1, dtype=torch.long, device=y.device)
        target[:, :y.shape[1]] = y
        target.scatter_(1, y_lengths.to(y.device)[:, None], EOS)
        token = torch.full((len(y),), BOS, dtype=torch.long, device=y.device)
        total = x.new_zeros(())
        for i in range(target.shape[1]):
            logits, hidden = self.step(token, hidden, memory, keys, mask)
            total = total + F.cross_entropy(logits, target[:, i], ignore_index=PAD, reduction="sum")
            token = target[:, i]  # teacher forcing, training loss only
        return total / (y_lengths.sum().to(x.device) + len(y))

    @torch.no_grad()
    def generate(self, x, lengths, max_chars):
        memory, keys, mask, hidden = self.prepare(x, lengths)
        token = torch.full((len(x),), BOS, dtype=torch.long, device=x.device)
        finished = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        output = []
        for _ in range(max_chars):
            logits, hidden = self.step(token, hidden, memory, keys, mask)
            token = logits.argmax(-1)
            token = torch.where(finished, torch.full_like(token, EOS), token)
            output.append(token)
            finished |= token == EOS
            if bool(finished.all()):
                break
        paths = torch.stack(output, 1).cpu().tolist()
        sequences = [path[:path.index(EOS)] if EOS in path else path for path in paths]
        return sequences, (~finished).cpu().tolist()


def build_decoder(kind, input_dim, cfg):
    factory = {"linear_ctc": LinearCTC, "gru_ctc": GRUCTC, "attention_ce": AttentionCE}[kind]
    return factory(input_dim, hidden=cfg["hidden"], layers=cfg["layers"],
                   dropout=cfg["dropout"], bidirectional=cfg["bidirectional"])


def training_loss(model, kind, x, lengths, y, y_lengths):
    if kind == "attention_ce":
        return model.loss(x, lengths, y, y_lengths)
    logits = model(x, lengths)
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), y,
                      lengths.cpu(), y_lengths.cpu(), blank=0,
                      reduction="mean", zero_infinity=False)


@torch.inference_mode()
def evaluate_decoder(model, kind, trials, cfg, device):
    model.eval()
    rows = []
    for x, lengths, _, _, batch_trials in make_loader(trials, cfg["batch_size"], False, 0):
        x = x.to(device)
        if kind == "attention_ce":
            # No targets are passed to generate; repeat characters are kept.
            predictions, hits = model.generate(x, lengths, cfg["max_decode_chars"])
        else:
            paths = model(x, lengths).argmax(-1).cpu()
            predictions = [ctc_collapse(path[:int(n)].tolist()) for path, n in zip(paths, lengths)]
            hits = [False] * len(predictions)
        for trial, prediction, hit in zip(batch_trials, predictions, hits):
            rows.append(dict(uid=trial.uid, group=trial.group,
                             reference=decode_ids(trial.targets), prediction=decode_ids(prediction),
                             edits=edit_distance(trial.targets, prediction),
                             reference_length=len(trial.targets),
                             prediction_length=len(prediction), generation_limit_hit=hit))
    return corpus_metrics(rows), rows


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_decoder(kind, train, dev, cfg, out_dir, device, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    if kind.endswith("ctc"):
        validate_ctc_lengths(train)
        validate_ctc_lengths(dev)
    seed_all(seed)
    model = build_decoder(kind, train[0].x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = make_loader(train, cfg["batch_size"], True, seed + 1)
    iterator = iter(loader)
    best_cer, best_step, history = float("inf"), 0, []
    started, running, seen = time.monotonic(), 0.0, 0
    for step in range(1, cfg["steps"] + 1):
        try:
            x, lengths, y, y_lengths, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, lengths, y, y_lengths, _ = next(iterator)
        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, kind, x, lengths, y, y_lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{kind}: nonfinite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        running += float(loss.detach())
        seen += 1
        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            print(f"  {kind} step {step}/{cfg['steps']} train loss={running / seen:.5f}", flush=True)
            running, seen = 0.0, 0
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            metrics, _ = evaluate_decoder(model, kind, dev, cfg, device)
            history.append(dict(step=step, dev=metrics))
            if metrics["cer"] < best_cer:
                best_cer, best_step = metrics["cer"], step
                torch.save(dict(kind=kind, input_dim=train[0].x.shape[1], config=cfg,
                                seed=seed, step=step, dev_cer=best_cer,
                                alphabet=CHARS, state_dict=model.state_dict()),
                           out_dir / "decoder_best.pt")
            print(f"  {kind} DEV CER={metrics['cer_percent']:.2f}% | "
                  f"best={100 * best_cer:.2f}% at step {best_step}", flush=True)
            dump_json(out_dir / "history.json", history)
    model.load_state_dict(load_torch(out_dir / "decoder_best.pt", device)["state_dict"])
    model.eval()
    return model, dict(best_step=best_step, best_dev_cer=best_cer,
                       decoder_parameters=sum(p.numel() for p in model.parameters()),
                       training_seconds=time.monotonic() - started)
