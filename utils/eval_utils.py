import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from utils.dataset import SpeechDataset, charset, HandwritingDataset, BrainToTextDataset
from utils.model import Encoder_Decoder
import os
import numpy as np
from utils.criterion import InfoNCE
from tqdm import trange, tqdm
from utils.sample_positive_negative import get_batch
import time
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List


def ctc_collate_nejm(
    batch: List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ]
):

    xs, ys, input_lengths, target_lengths, sessions = zip(*batch)

    B = len(xs)
    feat_dim = xs[0].shape[-1]

    input_lengths = torch.stack(input_lengths)
    target_lengths = torch.stack(target_lengths)

    T_max = int(input_lengths.max())

    x_pad = torch.zeros(
        B,
        T_max,
        feat_dim,
        dtype=torch.float32,
    )

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x

        if T < T_max:
            x_pad[i, T:] = x[-1:]

    max_target_len = int(target_lengths.max())

    targets_padded = torch.zeros(
        B,
        max_target_len,
        dtype=torch.long,
    )

    for i, y in enumerate(ys):
        L = y.shape[0]
        targets_padded[i, :L] = y

    sessions = torch.tensor(
        sessions,
        dtype=torch.int32,
    )

    return (
        x_pad,            # (B,T,F)
        targets_padded,   # (B,L)
        input_lengths,    # (B,)
        target_lengths,   # (B,)
        sessions,         # (B,)
    )

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

    target_seqs = [torch.tensor(charset.text_to_int(y), dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([t.numel() for t in target_seqs], dtype=torch.long)
    targets = torch.cat(target_seqs) if len(target_seqs) else torch.tensor([], dtype=torch.long)
    max_target_len = max(target_lengths) if len(target_lengths) > 0 else 0
    targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)

    offset = 0
    for i, length in enumerate(target_lengths):
        targets_padded[i, :length] = targets[offset:offset + length]
        offset += length


    sessions = torch.tensor(ds, dtype=torch.long)

    return x_pad, targets_padded, input_lengths, target_lengths, sessions




def _padding(batch):
    X, y, X_lens, y_lens, days = zip(*batch)

    B = len(X)
    feat_dim = X[0].shape[-1]
    max_T = max(x.shape[0] for x in X)

    X_padded = torch.zeros(B, max_T, feat_dim, dtype=X[0].dtype)

    for i, x in enumerate(X):
        T = x.shape[0]
        X_padded[i, :T] = x
        if T < max_T:
            X_padded[i, T:] = x[-1:]  

    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def get_dataset_loader_speech(
        datasetName,
        batchSize,
        gauss_in=False
    ):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    


    test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in)


    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return test_loader



def get_dataset_loader_nlp_10(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    final_day = 5
    test_input_1 = get_10_input(dataset_name, norm=True, train=False, valid=True, days=range(final_day), gauss=not gauss_in, gauss_sigma=2.0)
    test_input_2 = get_10_input(dataset_name, norm=True ,train=False, valid=False, days=range(final_day, 10), gauss=not gauss_in, gauss_sigma=2.0)
    test_input = test_input_1 + test_input_2
    valid_set = HandwritingDataset(test_input)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return test_loader


def get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in=False):
    with open(dataset_name, 'rb') as f:
        dataset_pkl = pickle.load(f)

    train_file_set = dataset_pkl['train'][:23]
    val_file_paths = dataset_pkl['test']
    train_ds = BrainToTextDataset(train_file_set, )
    valid_ds = BrainToTextDataset(val_file_paths)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate_nejm,
                              persistent_workers=True)

    test_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate_nejm,
    )
    return test_loader

def merge_by_borders(data1, borders1, data2, borders2):
    ends1 = borders1[1:] + [len(data1)]
    ends2 = borders2[1:] + [len(data2)]

    merged = []

    for start1, end1, start2, end2 in zip(
        borders1, ends1, borders2, ends2
    ):
        merged.extend(data1[start1:end1])
        merged.extend(data2[start2:end2])

    return merged

def get_dataset_loader_nlp_21(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    valid_input_1, borders_1 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/no_recalibration/mat/"),
        norm=True,
        valid=False,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0,
        return_borders=True
    )
    valid_input_2, borders_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        valid=False,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0,
        return_borders=True
    )
    valid_input = merge_by_borders(valid_input_1, borders_1, valid_input_2, borders_2)
    assert len(valid_input) == len(valid_input_1) + len(valid_input_2)

    valid_input_0 = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        valid=True, 
        gauss_sigma=2.0
    )
    print(len(valid_input), len(valid_input_0))
    valid_input = valid_input_0 + valid_input
    valid_set = HandwritingDataset(valid_input)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return test_loader

def get_dataset_loader(
        dataset_name,
        batch_size, 
        gauss_in=True, 
        speech=True,
        nlp_10=False,
        is_nejm=False
    ):
    if speech:
        if is_nejm: return get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in)
        return get_dataset_loader_speech(dataset_name, batch_size, gauss_in)
    if not nlp_10:
        return get_dataset_loader_nlp_21(dataset_name, batch_size, gauss_in)
    return get_dataset_loader_nlp_10(dataset_name, batch_size, gauss_in)


def eval_model(model, test_loader, device='cuda'):
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    error_and_lengths = []
    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        for X, y, X_len, y_len, testDayIdx in tqdm(test_loader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                X, y, X_len, y_len, testDayIdx = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                    testDayIdx.to(device),
                )
                pred, lengths = model(X, X_len)
                loss = ctc_criterion(
                    torch.permute(pred.log_softmax(2), [1, 0, 2]),
                    y,
                    lengths,
                    y_len,
                )
                loss = torch.sum(loss)
                allLoss.append(loss.cpu().detach().numpy())  
                for iterIdx in range(pred.shape[0]):
                    decodedSeq = torch.argmax(
                        torch.tensor(pred[iterIdx, 0: lengths[iterIdx], :]),
                        dim=-1,
                    )  # [num_seq,]
                    decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                    decodedSeq = decodedSeq.cpu().detach().numpy()
                    decodedSeq = np.array([i for i in decodedSeq if i != 0])

                    trueSeq = np.array(
                        y[iterIdx][0: y_len[iterIdx]].cpu().detach()
                    )
                    matcher = SequenceMatcher(
                        a=trueSeq.tolist(), b=decodedSeq.tolist()
                    )
                    distance = matcher.distance()
                    total_edit_distance += distance
                    total_seq_length += len(trueSeq)
                    error_and_lengths.append((distance, len(trueSeq)))

        avgDayLoss = np.sum(allLoss) / len(test_loader)
        cer = total_edit_distance / total_seq_length
        return cer, avgDayLoss, error_and_lengths

def fix_logits(logits):
    logits = torch.roll(torch.Tensor(logits), shifts=-6, dims=-1)
    logits[:, :, [26, 31]] = logits[:, :, [31, 26]]
    logits[:, :, [26, 27, 28, 29, 30]] = logits[:, :, [27, 28, 26, 30, 29]]
    return logits

def model_logits(model, test_loader, device='cuda', nlp=False):
    rnn_outputs = {"logits":[], "logitLengths":[], "trueSeqs":[]}
    with torch.no_grad():
        model.eval()

        for X, y, X_len, y_len, testDayIdx in tqdm(test_loader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                X, y, X_len, y_len, testDayIdx = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                    testDayIdx.to(device),
                )
                pred, lengths = model(X, X_len)
            pred = pred.float()
            if nlp:
                pred = fix_logits(pred)
            for iterIdx in range(pred.shape[0]):
                    trueSeq = np.array(y[iterIdx][0: y_len[iterIdx]].cpu().detach())

                    rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy().tolist())
                    rnn_outputs["logitLengths"].append(
                        lengths[iterIdx].cpu().detach().item()
                    )
                    rnn_outputs["trueSeqs"].append(trueSeq.tolist())


    return rnn_outputs