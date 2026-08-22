import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from utils.dataset import SpeechDataset, charset, HandwritingDataset
from utils.model import Encoder_Decoder
import os
import numpy as np
from utils.criterion import InfoNCE
from tqdm import trange
from utils.sample_positive_negative import get_batch
import time
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List
from utils.eval_utils import get_dataset_loaders_speech_nejm

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
    test_input = get_10_input(dataset_name, norm=True ,train=False, valid=True, days=range(10), gauss=not gauss_in, gauss_sigma=2.0)
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


def get_dataset_loader_nlp_21(
        dataset_name, 
        batch_size,
        gauss_in=True
    ):
    
    valid_input_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        valid=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input_3 = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        valid=True, 
        gauss_sigma=2.0
    )
    valid_input = valid_input_3 + valid_input_2
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

def get_dataset_loaders(
        speech_dataset,
        nlp10_dataset,
        nlp21_dataset,
        nejm_dataset,
        batch_size, 
        gauss_in=True, 
    ):
    speech_loader = get_dataset_loader_speech(speech_dataset, batch_size, gauss_in)
    nlp21_loader = get_dataset_loader_nlp_21(nlp21_dataset, batch_size, gauss_in)
    nlp10_loader = get_dataset_loader_nlp_10(nlp10_dataset, batch_size, gauss_in)
    nejm_dataset = get_dataset_loaders_speech_nejm(nejm_dataset, batch_size, gauss_in)
    all_loaders = [
        ('speech',  speech_loader),
        ('nlp10', nlp10_loader),
        ('nlp21', nlp21_loader),
        ('nejm', nejm_dataset),
    ]
    return all_loaders

def eval_model(model, session_name,test_loader, device='cuda'):
    ctc_criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    model = model.to(device)
    error_and_lengths = []
    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        for X, y, X_len, y_len, testDayIdx in test_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                X, y, X_len, y_len, testDayIdx = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                    testDayIdx.to(device),
                )
                pred, lengths = model(X, X_len, session_name)
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
        