# imports
import os
import torch
import pandas as pd
import sys
import numpy as np 
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

# internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_clam import CLAM_MB, CLAM_SB
from utils.eval_utils import initiate_model

# seeding
def seed_torch(seed=7, device=None):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# function to get subject level attention (got it from create_heatmaps.py)
def get_subject_level_attention(model, features, device):
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            model_results_dict = model(features)
            _, _, _, _, _, M = model(features)
            M = M.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError
    return M

def build_experiment_name(cfg):
    return '_'.join([

        str(cfg.stain_modality.description),

        str(cfg.seed),

        str(cfg.n_classes),

        str(cfg.stain_modality.description),

        str(cfg.model_type),
        str(cfg.model_size),
        str(cfg.drop_out),
        str(cfg.n_classes),
        str(cfg.embed_dim),
        str(cfg.n_folds),
        str(cfg.feature_type),
        str(cfg.split),
        str(cfg.splits_dir),
        str(cfg.features_dir)
    ])

@hydra.main(version_base="1.3.2", 
			config_path= '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion', 
			config_name='extract_subject_level_attentions')

def main(cfg:DictConfig):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device)

    experiment_name = build_experiment_name(cfg)

    features = cfg.stain_modality.features_dir
    subjects = os.listdir(features)

    save_dir = str(cfg.stain_modality.save_dir)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    splits_dir = cfg.splits_dir

    models_dir = os.path.join(cfg.results_dir, str(cfg.stain_modality.models_exp_code))
    print(models_dir)

    if cfg.splits_dir is None:
        cfg.splits_dir = models_dir

    assert os.path.isdir(models_dir)
    assert os.path.isdir(cfg.splits_dir)

    n_folds = cfg.n_folds
    for fold in range(n_folds):
        split_csv = os.path.join(splits_dir, f'splits_{fold}.csv')
        if not os.path.exists(split_csv):
            print(f"Split file not found: {split_csv}")
            continue
        df = pd.read_csv(split_csv)
        train_subjects = df[cfg.split].dropna().tolist()

        model_path = os.path.join(models_dir, f's_{fold}_checkpoint.pt')
        if not os.path.exists(model_path):
            print(f"Model checkpoint not found: {model_path}")
            continue

        # Prepare args for initiate_model
        class Args:
            pass
        args = Args()

        args.model_type = cfg.model_type
        args.model_size = cfg.model_size
        args.drop_out = cfg.drop_out
        args.n_classes = cfg.n_classes
        args.embed_dim = cfg.embed_dim

        # Load model and set to eval mode for deterministic feature extraction (as in eval_utils.py)
        model = initiate_model(args, model_path, device=device)
        model.eval()

        fold_save_path = os.path.join(save_dir, f'fold_{fold}')
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)

        for subject in train_subjects:
            feature_file = f'{subject}.pt' if not subject.endswith('.pt') else subject
            feature_path = os.path.join(features, feature_file)
            if not os.path.exists(feature_path):
                print(f"Feature file not found: {feature_path}")
                continue
            features_tensor = torch.load(feature_path, map_location=device)
            if isinstance(features_tensor, dict) and 'features' in features_tensor:
                features_tensor = features_tensor['features']
            M = get_subject_level_attention(model, features_tensor, device)
            save_file = os.path.join(fold_save_path, feature_file)
            torch.save({'subject_attention': M}, save_file)
            print(f"[Fold {fold}] Saved subject-level attention for {subject} to {save_file}")

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")