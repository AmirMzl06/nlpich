import torch.nn as nn
import torch
from utils.constants import CEBRA_DIR
import sys
sys.path.append(str(CEBRA_DIR))
from cebra.models import Offset36Dropoutv2, Offset10Model
import cebra.models.layers as cebra_layers
from utils.augmentation import GaussianSmoothing, Unfolder
from utils.mlp import TwoLayerMLP
import torch.nn.functional as F
from utils.macorn_options import Offset36_multi

class RNN_Decoder(nn.Module):
    def __init__(self, dim, gru, rnn_hidden, rnn_layers, rnn_bidir, rnn_dr, num_classes, use_rnn=True, ):
        super().__init__()
        self.use_rnn = use_rnn
        if self.use_rnn:
            if gru:
                self.rnn = nn.GRU(
                    dim, 
                    rnn_hidden,
                    rnn_layers,
                    batch_first=True, 
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                    )
            else:
                self.rnn = nn.LSTM(
                    dim,
                    rnn_hidden, 
                    rnn_layers,
                    batch_first=True,
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                )
        
        self.final_decoder = nn.Linear((2 if rnn_bidir else 1) * rnn_hidden, num_classes)
        
    
    def forward(self, x):
        if self.use_rnn:
            x, _ = self.rnn(x)
        x = self.final_decoder(x)
        return x


class Encoder_Decoder_Multi(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, cebra_out_dim, kernel, stride, rnn_hidden, rnn_layers, rnn_dr = 0.4, rnn_bidir=True, cebra_unfolder=False, gru = False, smooth_width=2.0, gauss_in=True, shared_rnn=False,
                  pooling=False, n_layers=16, batch_norm=False, normalize=True):
        super().__init__()
        print(n_layers, pooling, batch_norm, normalize)
        def init_cebra():
            
            ceb_model = Offset36_multi(256, cebra_out_dim, normalize, pooling, n_layers, batch_norm)
            self.offset = ceb_model.get_offset().left
            return ceb_model

        self.norm_func = cebra_layers._Norm()
        self.shared_rnn = shared_rnn

        self.cebra_unfolder = cebra_unfolder
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn_dr = rnn_dr
        self.rnn_bidir = rnn_bidir
        self.cebra = None
        self.gru = gru
        self.smooth_width = smooth_width
        self.gauss_in = gauss_in
        self.smoothers = nn.ModuleDict()
        self.cebra_dim_mul = 1 if cebra_unfolder else kernel
        current_dim = 1
        
            

        if cebra_unfolder:
            self.cebra = init_cebra()
            current_dim = cebra_out_dim
            

        self.unfolder = Unfolder(kernel, stride)
        current_dim *= kernel

        if not cebra_unfolder:
            self.cebra = init_cebra()
            current_dim = cebra_out_dim
        
        self.rnn_dim = current_dim
        if self.shared_rnn:
            if gru:
                self.rnn = nn.GRU(
                    self.rnn_dim, 
                    rnn_hidden,
                    rnn_layers,
                    batch_first=True, 
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                    )
            else:
                self.rnn = nn.LSTM(
                    self.rnn_dim,
                    rnn_hidden, 
                    rnn_layers,
                    batch_first=True,
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                )
        
        self.decoders = nn.ModuleDict()

    def add_session(self, session_name, num_neurons, num_outputs):
        if self.cebra is not None:
            self.cebra.add_session(session_name, num_neurons * self.cebra_dim_mul)
        new_dec = RNN_Decoder(
            self.rnn_dim,
            self.gru, 
            self.rnn_hidden,
            self.rnn_layers,
            self.rnn_bidir,
            self.rnn_dr, 
            num_outputs,
            not self.shared_rnn
        )
        self.decoders[session_name] = new_dec
        smoother = (GaussianSmoothing(num_neurons, 20, self.smooth_width, dim=1)) if self.gauss_in else (nn.Identity())
        self.smoothers[session_name] = smoother
    
    def _apply_cebra(self, x, lengths, session_name):
        """Helper to permute, pad, forward CEBRA, and permute back."""
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = F.pad(x, (self.offset, self.offset - 1), mode='replicate')
        x = self.cebra(x, session_name).permute(0, 2, 1)  # (B, T, C)
        self.embeddings = self.norm_func(x.permute(0, 2, 1)).permute(0, 2, 1)
        self.emb_lengths = lengths
        return x
    
    def get_cebra_embs(self):
        return self.embeddings, self.emb_lengths
    
    def forward(self, x, lengths, session_name):
        x = self.smoothers[session_name](x)
        if self.cebra_unfolder:
            x = self._apply_cebra(x, lengths, session_name)
        x, lengths = self.unfolder(x, lengths)
        if not self.cebra_unfolder:
            x = self._apply_cebra(x, lengths, session_name)
        if self.shared_rnn:
            x, _ = self.rnn(x)
        x = self.decoders[session_name](x)
        return x, lengths

