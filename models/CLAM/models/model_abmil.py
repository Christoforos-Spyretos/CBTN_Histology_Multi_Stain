import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout:bool=False, dropout_rate:float=0.25, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout_rate))
            self.attention_b.append(nn.Dropout(dropout_rate))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
Two layer gated attention following the implementation details of UNI (https://doi.org/10.1038/s41591-024-02857-3 , Weakly supervised slide classification)
"""

class ABMIL(nn.Module):
    def __init__(self, feature_encoding_size:int=1024, dropout:bool=True, features_dropout_rate:float=0.1, attention_layer_dropout_rate:float=0.25, n_classes=2, **args):
        super(ABMIL, self).__init__()
        layers_sizes = [feature_encoding_size, 512, 384]

        # map the features to the same intermediate dimension
        fc = [nn.Linear(layers_sizes[0], layers_sizes[1]), nn.ReLU()]

        # add feature dropout
        if dropout:
            fc.append(nn.Dropout(features_dropout_rate))
        
        # build gated attention layers
        attention_layer = Attn_Net_Gated(L = layers_sizes[1], D = layers_sizes[2], dropout = dropout, dropout_rate=attention_layer_dropout_rate, n_classes = 1)
        fc.append(attention_layer)
        self.attention_net = nn.Sequential(*fc)

        # make output classification layer
        self.classifier = nn.Linear(layers_sizes[1], n_classes)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)
    
    def forward(self, h, label=None, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        # attention weighting 
        M = torch.mm(A, h) 

        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits, Y_prob, Y_hat, A_raw, results_dict