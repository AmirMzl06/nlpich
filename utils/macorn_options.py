from utils.constants import CEBRA_DIR
import sys
import torch
import torch.nn as nn
sys.path.append(str(CEBRA_DIR))
import math
import cebra
import cebra.models.layers as cebra_layers
import torch.nn.functional as F


class Z_score_pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(3, 1)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - mean) / (std)
        x = self.pool(x)
        return x


class Session_layer(nn.Module):
    def __init__(self, num_units, kernel=2) -> None:
        super().__init__()
        self.num_units = num_units
        self.kernel = kernel
        self.session_dict = nn.ModuleDict()
    
    def add_session(self, session_name, num_neurons):
        self.session_dict.add_module(session_name, nn.Conv1d(num_neurons, self.num_units, self.kernel))
        
    
    def forward(self, x, session_name):
        return self.session_dict[session_name](x)
    
    
class Offset36_multi(nn.Module):
    def _make_layers(self, num_units, p, n):
        blocks = [
            torch.nn.Dropout1d(p=p),
            nn.Conv1d(num_units, num_units, 3, padding=1 if self.pooling else 0),
            nn.BatchNorm1d(num_units) if self.batch_norm else nn.Identity(),
            nn.GELU()
        ]
        if self.pooling:
            # blocks += [nn.AvgPool1d(3, 1)]
            blocks += [
                Z_score_pooling(),
            ]

        return [
            cebra_layers._Skip(*blocks)
            for _ in range(n)
        ]
    
    
    def __init__(self, num_units, num_outputs, normalize=True, pooling=False, n_layers=16, batch_norm=False) -> None:
        super().__init__()
        self.num_units = num_units
        self.pooling = pooling
        self.batch_norm = batch_norm
        self.n_layers = n_layers
        self.window = 2 * n_layers + 4
        self.session_layer = Session_layer(num_units, 2)
        net = [ torch.nn.Dropout1d(p=0.1),
                nn.GELU(),
                *self._make_layers(num_units, 0.1, n_layers),
                # nn.Conv1d(num_units, num_outputs, 3),
                ]
        self.first_net = nn.Sequential(*net)
        self.last_layer_multi = Session_layer(num_outputs, 3)
        net = []
        if normalize:
            net += (cebra_layers._Norm(),)
        net += (cebra_layers.Squeeze(),)
        self.last_net = nn.Sequential(*net)
    
    def forward(self, x, session):
        x = self.session_layer(x, session)
        x = self.first_net(x)
        x = self.last_layer_multi(x, session)
        x = self.last_net(x)
        return x

    def add_session(self, session_name, num_neurons):
        self.session_layer.add_session(session_name, num_neurons)
        self.last_layer_multi.add_session(session_name, self.num_units)
        
        
    def get_offset(self):
        return cebra.data.Offset(math.floor(self.window / 2), math.ceil(self.window / 2))    





