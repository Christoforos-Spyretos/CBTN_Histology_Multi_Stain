# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# linear layer model
class Linear_Layer(nn.Module):
    def __init__(self, input_dim, n_classes): # vector_dim, n_classes
        super(Linear_Layer,self).__init__()        
        self.fc = nn.Linear(input_dim, n_classes) # single layer 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        output = self.relu(x)
        return output
    
# one hidden layer model 
class One_Hidden_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes): # vector_dim, hidden_dim, n_classes
        super(One_Hidden_Layer,self).__init__()        
        self.fc1 = nn.Linear(input_dim, hidden_dim) # hidden layer
        self.fc2 = nn.Linear(hidden_dim, n_classes) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output
    
# two hidden layer model
class Two_Hidden_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, n_classes): # vector_dim, hidden_dim1, hidden_dim2, n_classes
        super(Two_Hidden_Layer,self).__init__()        
        self.fc1 = nn.Linear(input_dim, hidden_dim1) # first hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) # second hidden layer
        self.fc3 = nn.Linear(hidden_dim2, n_classes) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc3(x)
        return output
    
# attention layer model
class Attention_Layer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Attention_Layer, self).__init__()
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
       

       
