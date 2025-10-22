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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'CLAM'))
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
        
        str(cfg.seed),
        
        str(cfg.stain_modality.description),
        str(cfg.n_classes),

        str(cfg.features_dir),
        str(cfg.feature_type),
        str(cfg.embed_dim),

        str(cfg.save_dir),

        str(cfg.results_dir),
        str(cfg.models_exp_code),
        str(cfg.model_type),
        str(cfg.model_size),
        str(cfg.drop_out),        
    ])

@hydra.main(version_base="1.3.2", 
			config_path= '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/data_utilities', 
			config_name='extract_learned_subject_level_features')

def main(cfg:DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device)

    experiment_name = build_experiment_name(cfg)

    features_dir = cfg.features_dir

    save_dir = str(cfg.save_dir)
    print(f"Save directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Use the model path directly from models_exp_code (fold 10 checkpoint)
    model_path = os.path.join(cfg.results_dir, str(cfg.models_exp_code))
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Prepare args for initiate_model
    class Args:
        pass
    args = Args()

    args.model_type = cfg.model_type
    args.model_size = cfg.model_size
    args.drop_out = cfg.drop_out
    args.n_classes = cfg.n_classes
    args.embed_dim = cfg.embed_dim

    # Load model and set to eval mode for deterministic feature extraction
    print("Loading model...")
    model = initiate_model(args, model_path, device=device)
    model.eval()
    print("Model loaded successfully!")

    # Get all .pt files from features_dir
    print(f"\nSearching for feature files in: {features_dir}")
    if not os.path.isdir(features_dir):
        raise NotADirectoryError(f"Features directory not found: {features_dir}")
    
    all_feature_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]
    print(f"Found {len(all_feature_files)} feature files to process.\n")

    # Process each feature file
    for idx, feature_file in enumerate(all_feature_files, 1):
        feature_path = os.path.join(features_dir, feature_file)
        
        try:
            features_tensor = torch.load(feature_path, map_location=device)
            if isinstance(features_tensor, dict) and 'features' in features_tensor:
                features_tensor = features_tensor['features']
            
            # Extract subject-level attention
            M = get_subject_level_attention(model, features_tensor, device)
            
            # Save the attention
            save_file = os.path.join(save_dir, feature_file)
            torch.save({'subject_attention': M}, save_file)
            
            print(f"[{idx}/{len(all_feature_files)}] Saved subject-level attention for {feature_file}")
            
        except Exception as e:
            print(f"[{idx}/{len(all_feature_files)}] Error processing {feature_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")