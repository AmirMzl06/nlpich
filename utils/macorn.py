from utils.constants import CEBRA_DIR
import sys
import torch
import torch.nn as nn
sys.path.append(str(CEBRA_DIR))
import cebra
import cebra.models.layers as cebra_layers
import torch.nn.functional as F

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
        return [
            cebra_layers._Skip(torch.nn.Dropout1d(p=p),
                               nn.Conv1d(num_units, num_units, 3),
                                nn.GELU())
            for _ in range(n)
        ]
    
    
    def __init__(self, num_units, num_outputs, normalize=True) -> None:
        super().__init__()
        print("205")
        self.num_units = num_units
        self.session_layer = Session_layer(num_units, 2)
        net = [ torch.nn.Dropout1d(p=0.1),
                nn.GELU(),
                *self._make_layers(num_units, 0.1, 16),
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
        return cebra.data.Offset(18, 18)    





