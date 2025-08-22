# imports
import os
import torch
import torch.nn.functional as F

# classifier
class Classifier(torch.nn.Module):

    def __init__(self, embed_dim, n_classes):
        super(Classifier, self).__init__()
        self.classifiers = torch.nn.Linear(embed_dim, n_classes)

    def forward(self, M):
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat
