# %% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% SIMPLE MODEL
class Simple_MLP(nn.Module):
    def __init__(self, input_dim, n_classes): # vector_dim, n_classes
        super(Simple_MLP,self).__init__()        
        self.fc = nn.Linear(input_dim, n_classes) # single layer 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        output = self.relu(x)
        return output
    
# %% ONE HIDDEN LAYER MODEL  
class One_Hidden_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes): # vector_dim, hidden_dim, n_classes
        super(One_Hidden_MLP,self).__init__()        
        self.fc1 = nn.Linear(input_dim, hidden_dim) # hidden layer
        self.fc2 = nn.Linear(hidden_dim, n_classes) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output
    
# %% ATTENTION BASED MODEL
class Attention(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        
        if output.dim() == 3:
            output = output.mean(dim=1)
        logits = self.classifier(output)
        logits = self.relu(logits)
        return logits
       

       
