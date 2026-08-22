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

class GRU_Decoder(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, neural_dim, kernel, stride, num_classes, rnn_hidden, rnn_layers, rnn_dr = 0.4, rnn_bidir=True,  gru = False, smooth_width=2.0, gauss_in=True):
        super().__init__()
        
        
        current_dim = neural_dim
        self.unfolder = Unfolder(1, 2)
        self.smoother = (GaussianSmoothing(neural_dim, 20, smooth_width, dim=1)) if gauss_in else (nn.Identity())



       

        if gru:
                self.rnn = nn.GRU(
                    current_dim, 
                    rnn_hidden,
                    rnn_layers,
                    batch_first=True, 
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                    )
        else:
                self.rnn = nn.LSTM(
                    current_dim,
                    rnn_hidden, 
                    rnn_layers,
                    batch_first=True,
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                )
        current_dim = rnn_hidden * (2 if rnn_bidir else 1)
        
        
        self.final_decoder = nn.Linear(current_dim, num_classes)
    
    
    
    
    def forward(self, x, lengths):
        x = self.smoother(x)
        x, lengths = self.unfolder(x, lengths)
        x, _ = self.rnn(x)
        x = self.final_decoder(x)
        return x, lengths

