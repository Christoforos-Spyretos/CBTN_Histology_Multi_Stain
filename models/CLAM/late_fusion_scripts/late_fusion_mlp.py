# %% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import nympy as np

# %% SIMPLE MODEL
class Simple_MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Simple_MLP,self).__init__()        
        self.fc = nn.Linear(input_dim, n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        output = self.relu(x)
        return output
    
# %% ONE HIDDEN LAYER MODEL  
class One_Hidden_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(One_Hidden_MLP,self).__init__()        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output
    
# %% ATTENTION BASED MODELS
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim

        # linear transformations for Q, K, V
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # apply linear transformations to compute keys, queries, and values
        keys = self.key(x)    
        queries = self.query(x)  
        values = self.value(x)  

        # scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))  
        scores = scores / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))

        # attention weights
        attention_weights = F.softmax(scores, dim=-1)  

        # multiply weights with values
        output = torch.matmul(attention_weights, values)  

        return output
       

       
