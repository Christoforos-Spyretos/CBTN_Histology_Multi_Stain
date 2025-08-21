# imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf


# seeding
def seed_torch(seed=7, device=None):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Cross-Attention Mechanism
class CrossAttention(nn.Module):
    """
    Cross-attention mechanism between two 1 x dim vectors.
    Given query (q) and key (k), both shape (1, dim), computes attention-weighted value.
    """
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, q, k):
        q_proj = self.query_proj(q)
        k_proj = self.key_proj(k)
        v_proj = self.value_proj(k)
        attn_score = torch.matmul(q_proj, k_proj.t()) / self.scale
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_output = torch.matmul(attn_weight, v_proj)
        return attn_output

def build_experiment_name(cfg):
    return '_'.join([

        str(cfg.seed),

        str(cfg.modality_1_dir),
        str(cfg.modality_2_dir),

        str(cfg.save_dir),

        str(cfg.n_folds),
        str(cfg.splits_dir),
        str(cfg.split),

        str(cfg.feature_type),
        str(cfg.embed_dim)

    ])

@hydra.main(version_base="1.3.2", 
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion',
            config_name='cross_attention')

def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device)

    experiment_name = build_experiment_name(cfg)

    save_dir = str(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    splits_dir = cfg.splits_dir
    assert os.path.isdir(splits_dir)

    n_folds = cfg.n_folds

    modality_1_dir = cfg.modality_1_dir
    modality_2_dir = cfg.modality_2_dir

    embed_dim = cfg.embed_dim

    for fold in range(n_folds):
        fold_modality_1 = os.path.join(modality_1_dir, f'fold_{fold}')
        fold_modality_2 = os.path.join(modality_2_dir, f'fold_{fold}')

        fold_save = os.path.join(save_dir, f'fold_{fold}')
        os.makedirs(fold_save, exist_ok=True)

        if not os.path.exists(fold_modality_1) or not os.path.exists(fold_modality_2):
            print(f"Missing fold: {fold}")
            continue

        subjects = [f for f in os.listdir(fold_modality_1) if f.endswith('.pt')]
        ca = CrossAttention(embed_dim)

        ca.eval()

        for subj_file in subjects:
            file_1 = os.path.join(fold_modality_1, subj_file)
            file_2 = os.path.join(fold_modality_2, subj_file)

            if not os.path.exists(file_2):
                print(f"Missing modality 2 for {subj_file} in fold {fold}")
                continue

            data_1 = torch.load(file_1)
            data_2 = torch.load(file_2)

            attn_1 = data_1['subject_attention'] if isinstance(data_1, dict) else data_1
            attn_2 = data_2['subject_attention'] if isinstance(data_2, dict) else data_2

            attn_1 = torch.tensor(attn_1, dtype=torch.float32).view(1, -1)
            attn_2 = torch.tensor(attn_2, dtype=torch.float32).view(1, -1)

            with torch.no_grad():
                cross_attn = ca(attn_1, attn_2)

            save_path = os.path.join(fold_save, subj_file)
            torch.save({'cross_attended': cross_attn.cpu().numpy()}, save_path)
            print(f"Saved cross-attended for {subj_file} in fold {fold}")

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")