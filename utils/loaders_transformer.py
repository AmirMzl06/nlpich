import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.dataset import SpeechDataset, charset, HandwritingDataset
import os
import numpy as np
from tqdm import trange
from edit_distance import SequenceMatcher
from utils.data_10_loader import get_input as get_10_input
from utils.data_loader import get_input
from typing import Tuple, List
from utils.trainer import BrainToTextDataset, ctc_collate_nejm

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

def get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in=False,         encoder=None, session_name=None):
    with open(dataset_name, 'rb') as f:
        dataset_pkl = pickle.load(f)

    train_file_set = dataset_pkl['train'][:23]
    val_file_paths = dataset_pkl['test']
    train_ds = BrainToTextDataset(train_file_set, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    valid_ds = BrainToTextDataset(val_file_paths, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
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
    return train_loader, test_loader, None



def get_dataset_loaders_speech(
        datasetName,
        batchSize,
        gauss_in=False,
        encoder=None, session_name=None

):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    train_ds = SpeechDataset(loadedData["train"], transform=None, gauss=not gauss_in, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    test_ds = SpeechDataset(loadedData["test"], gauss=not gauss_in, encoder=encoder, session_name=session_name, device=torch.device('cuda'))

    train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=_padding,
                              persistent_workers=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData


def get_dataset_loaders_nlp_10(
        dataset_name,
        batch_size,
        gauss_in=True,
        encoder=None, session_name=None

):
    final_day = 5
    train_input = get_10_input(dataset_name, norm=True, train=True, days=range(final_day), gauss=not gauss_in,
                               gauss_sigma=2.0)
    test_input = get_10_input(dataset_name, norm=True, train=False, days=range(10), valid=True, gauss=not gauss_in,
                              gauss_sigma=2.0)

    valid_set = HandwritingDataset(test_input, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    train_set = HandwritingDataset(train_input, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None


def get_dataset_loaders_nlp_21(
        dataset_name,
        batch_size,
        gauss_in=True,
        encoder=None, session_name=None
):
    train_input = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=True,
        gauss_sigma=2.0
    )
    valid_input_1 = get_input(
        os.path.join(dataset_name, "seed_model_training_data/mat/"),
        norm=True,
        gauss=not gauss_in,
        train=False,
        valid=True,
        gauss_sigma=2.0
    )
    valid_input_2 = get_input(
        os.path.join(dataset_name, "online_evaluation_data/recalibration/mat/"),
        norm=True,
        valid=True,
        gauss=not gauss_in,
        train=False,
        gauss_sigma=2.0
    )
    valid_input = valid_input_1 + valid_input_2
    valid_set = HandwritingDataset(valid_input, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    train_set = HandwritingDataset(train_input, encoder=encoder, session_name=session_name, device=torch.device('cuda'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=ctc_collate,
                              persistent_workers=True)
    test_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=ctc_collate,
    )
    return train_loader, test_loader, None


def get_dataset_loaders(
        speech_dataset,
        nlp10_dataset,
        nlp21_dataset,
        nejm_dataset,
        batch_size,
        gauss_in=True,
        encoder=None
):
    speech_data = get_dataset_loaders_speech(speech_dataset, batch_size, gauss_in, encoder=encoder, session_name='speech')
    nlp21_data = get_dataset_loaders_nlp_21(nlp21_dataset, batch_size, gauss_in, encoder=encoder, session_name='nlp21')
    nlp10_data = get_dataset_loaders_nlp_10(nlp10_dataset, batch_size, gauss_in, encoder=encoder, session_name='nlp10')
    nejm_data = get_dataset_loaders_speech_nejm(nejm_dataset, batch_size, gauss_in, encoder=encoder, session_name='nejm')
    all_loaders = [
        ('speech', 'speech', 256, 41, speech_data),
        ('nlp10', 'nlp', 192, 32, nlp10_data),
        ('nlp21', 'nlp', 192, 32, nlp21_data),
        ('nejm', 'speech', 512, 41, nejm_data)
    ]
    return all_loaders

def get_dataset_loaders_single(
        dataset_name,
        batch_size,
        gauss_in=True,
        speech=True,
        nlp_10=False,
        is_nejm=False, encoder=False
    ):
    if speech:
        if is_nejm: return get_dataset_loaders_speech_nejm(dataset_name, batch_size, gauss_in, encoder=encoder, session_name='default')
        return get_dataset_loaders_speech(dataset_name, batch_size, gauss_in, encoder=encoder, session_name='default')
    if not nlp_10:
        return get_dataset_loaders_nlp_21(dataset_name, batch_size, gauss_in, encoder=encoder, session_name='default')
    return get_dataset_loaders_nlp_10(dataset_name, batch_size, gauss_in, encoder=encoder, session_name='default')
