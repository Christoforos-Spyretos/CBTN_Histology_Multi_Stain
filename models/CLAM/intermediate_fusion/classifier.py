# imports
import os
import torch
import torch.nn.functional as F

# classifier
class Classifier(torch.nn.Module):

    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Linear(in_features, num_classes) for _ in range(8)
        ])

    def forward(self, M):
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return Y_hat, Y_prob
