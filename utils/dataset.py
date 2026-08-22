import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple
from utils.normalization import gaussSmooth
from utils.augmentation import GaussianSmoothing
import h5py
from tqdm import tqdm, trange


class BrainToTextDataset(Dataset):

    def __init__(self, dataset, feature_subset=None, encoder=None, session_name=None, device=None):
        self.feature_subset = feature_subset
        self.device = device
        self.samples = []


        for session_idx, trials in tqdm(enumerate(dataset)):
            for trial in trials:
                if session_name is not None and encoder is not None:
                    x = torch.tensor(trial["x"]).to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        trial["x"], _ = encoder.forward_encoder(x, torch.tensor([x.shape[1]]), session_name)
                        trial["x"] = trial["x"].squeeze(0).cpu().numpy()
                self.samples.append((session_idx, trial))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        session, sample = self.samples[idx]
        x = torch.tensor(sample["x"])
        y = torch.tensor(sample["y"])
        x_len = torch.tensor(x.shape[0], dtype=torch.int64)
        y_len = torch.tensor(y.shape[0], dtype=torch.int64)


        return x, y, x_len, y_len, session

class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, gauss=False, encoder=None, session_name=None, device=None):
        self.device = device
        if gauss:
            smoother = GaussianSmoothing(256, 20, 2.0, dim=1)
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum(len(d["sentenceDat"]) for d in data)

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []

        for day in trange(self.n_days):
            day_data = data[day]
            n_trials = len(day_data["sentenceDat"])
            for trial in range(n_trials):
                neural = day_data["sentenceDat"][trial]
                neural = torch.as_tensor(neural, dtype=torch.float32)
                phones = day_data["phonemes"][trial]
                if gauss:
                    with torch.no_grad():
                        neural = smoother(neural.unsqueeze(0)).squeeze(0)
                if session_name is not None and encoder is not None:
                    x = neural.to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        neural, _ = encoder.forward_encoder(x, torch.tensor([x.shape[1]]), session_name)
                        neural = neural.squeeze(0).cpu()
                self.neural_feats.append(neural)
                self.phone_seqs.append(torch.as_tensor(phones, dtype=torch.int32))
                self.neural_time_bins.append(neural.shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = self.neural_feats[idx]
        phone_seq = self.phone_seqs[idx]
        neural_time_bin = self.neural_time_bins[idx]
        phone_seq_len = self.phone_seq_lens[idx]
        day = self.days[idx]

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,                               # already float32
            phone_seq,                                  # already int32
            torch.tensor(neural_time_bin, dtype=torch.int32),
            torch.tensor(phone_seq_len, dtype=torch.int32),
            torch.tensor(day, dtype=torch.int64),
        )


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


class HandwritingDataset(Dataset):
    """
    items: List of (features, transcript, session_id)
    """
    def __init__(self, items: List[Tuple[torch.FloatTensor, str, int]], encoder=None, session_name=None, device=None):
        super().__init__()
        self.items = items
        self.device = device
        self.charset = charset
        if encoder is not None:
            n = len(items)
            for i in trange(n):
                x, y, z = items[i]
                x = x.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    x, _ = encoder.forward_encoder(x, torch.tensor([x.shape[1]]), session_name)
                    x = x.squeeze(0).cpu()
                self.items[i] = (x, y, z)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, d = self.items[idx]

        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.dim() == 2
        assert isinstance(y, str)
        assert isinstance(d, int)

        return x, y, d

