# %%%
import torch
import torch.nn as nn
from typing import Union

def age_encoder():
    """
    Encodes age using one dense layer into a three-valued vector.
    Returns:
        torch.nn.Linear: Linear layer mapping age (1D) to 3D representation
    """
    return torch.nn.Linear(in_features=1, out_features=3)

def fuse_image_age(image_features, age_features):
    """
    Fuses image features (512D) with age features (3D) by concatenation.
    
    Args:
        image_features: Image feature vector of shape (batch_size, 512)
        age_features: Age feature vector of shape (batch_size, 3)
        
    Returns:
        torch.Tensor: Concatenated features of shape (batch_size, 515)
    """
    return torch.cat([image_features, age_features], dim=1)

# classifier with age fusion
class Classifier(torch.nn.Module):

    def __init__(self, embed_dim, n_classes, age_dim=3, drop_out=0.0, use_age=True):
        super(Classifier, self).__init__()
        self.use_age = use_age
        
        if self.use_age:
            # Encode age to 3D representation
            self.age_encoder = torch.nn.Linear(1, age_dim)
            # Classifier on concatenated features: 512 (image) + 3 (age) = 515
            classifier_input_dim = embed_dim + age_dim
        else:
            # No age fusion - just image features
            self.age_encoder = None
            classifier_input_dim = embed_dim
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(drop_out) if drop_out > 0 else None
        # Classifier
        self.classifier = torch.nn.Linear(classifier_input_dim, n_classes)

    def forward(self, M, age=None):
        if self.use_age and age is not None:
            # Encode age to 3D representation
            age_features = self.age_encoder(age)
            # Concatenate image and age features
            fused_features = torch.cat([M, age_features], dim=1)
        else:
            # Use only image features
            fused_features = M
        
        # Apply dropout if enabled
        if self.dropout is not None:
            fused_features = self.dropout(fused_features)

        # Final classification
        logits = self.classifier(fused_features)
        
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = torch.nn.functional.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat
    
# %%


