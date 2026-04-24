# imports
import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import csv

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
            config_name='concatenation')

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

        fold_save = os.path.join(save_dir, f'fold_{fold}', 'pt_files')
        os.makedirs(fold_save, exist_ok=True)

        if not os.path.exists(fold_modality_1) or not os.path.exists(fold_modality_2):
            print(f"Missing fold: {fold}")
            continue

        # load split file for this fold (CSV)
        split_file = os.path.join(splits_dir, f'splits_{fold}.csv')
        if not os.path.exists(split_file):
            print(f"Missing split file for fold {fold}: {split_file}")
            continue
        split_key = str(cfg.split)
        split_subjects = set()
        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj_id = row.get(split_key)
                if subj_id is not None and subj_id.strip() != '':
                    split_subjects.add(str(subj_id).strip())

        if not split_subjects:
            print(f"[DEBUG] No subjects found for split '{split_key}' in fold {fold}.")

        subjects = [f for f in os.listdir(fold_modality_1) if f.endswith('.pt')]

        for subj_file in subjects:
            subj_id = os.path.splitext(subj_file)[0].strip()
            if subj_id not in split_subjects:
                continue  # only process subjects in the current split for this fold

            file_1 = os.path.join(fold_modality_1, subj_file)
            file_2 = os.path.join(fold_modality_2, subj_file)

            if not os.path.exists(file_2):
                print(f"Missing modality 2 for {subj_file} in fold {fold}")
                continue

            data_1 = torch.load(file_1, weights_only=False)
            data_2 = torch.load(file_2, weights_only=False)

            attn_1 = data_1['subject_attention'] if isinstance(data_1, dict) else data_1
            attn_2 = data_2['subject_attention'] if isinstance(data_2, dict) else data_2

            attn_1 = torch.tensor(attn_1, dtype=torch.float32).view(1, -1)  # (1, dim)
            attn_2 = torch.tensor(attn_2, dtype=torch.float32).view(1, -1)  # (1, dim)

            # concatenate along feature dimension: (1, dim) + (1, dim) -> (1, 2*dim)
            concatenated = torch.cat([attn_1, attn_2], dim=-1)

            save_path = os.path.join(fold_save, subj_file)
            torch.save({'concatenated': concatenated.cpu().numpy()}, save_path)
            print(f"Saved concatenated for {subj_file} in fold {fold} ({split_key})")

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")
