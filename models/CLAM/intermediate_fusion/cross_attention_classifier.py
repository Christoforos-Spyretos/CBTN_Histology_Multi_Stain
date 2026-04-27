# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionClassifier(nn.Module):
    """
    cross-attention fusion + linear classifier.

    modality_1 patient vector is used as query; modality_2 patient vector is
    used as key and value, so modality_1 is informed by modality_2.

    Given M1, M2 in R^{1 x d}:
        Q = (W_Q * M1)^T                  in R^{d x 1}
        K = (W_K * M2)^T                  in R^{d x 1}
        V = (W_V * M2)^T                  in R^{d x 1}
        Att = softmax(Q * K^T / sqrt(d))  in R^{d x d}
        M_cross = (Att * V)^T             in R^{1 x d}
        logits = W_cls * M_cross          in R^{n_classes}

    W_Q, W_K, W_V, and W_cls are all trained end-to-end with the
    classification loss, so the projections are actually informative.
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, m1: torch.Tensor, m2: torch.Tensor):
        # m1: (1, d) modality 1 patient-level vector (query)
        # m2: (1, d) modality 2 patient-level vector (key & value)
        q_t = self.W_Q(m1).t()  # (d, 1)
        k_t = self.W_K(m2).t()  # (d, 1)
        v_t = self.W_V(m2).t()  # (d, 1)

        attn_score = torch.matmul(q_t, k_t.t()) / self.scale    # (d, d)
        attn_weight = F.softmax(attn_score, dim=-1)             # (d, d)
        fused = torch.matmul(attn_weight, v_t).t()              # (1, d)

        logits = self.classifier(fused)                         # (1, n_classes)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat
