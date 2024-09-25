# torch imports
import torch
import torch.nn as nn

# simple MLP model to perfrom late fusion
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
