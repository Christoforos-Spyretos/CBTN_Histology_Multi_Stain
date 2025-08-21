# %% IMPORTS 
import os
import torch
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_clam import CLAM_MB, CLAM_SB

# %% GET SUBJECT LEVEL ATTENTION
def get_subject_level_attention(model, features):
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            model_results_dict = model(features)
            _, _, _, _, _, M = model(features)
           
            M = M.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

    return M

# %% LOAD PATHS
HE_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1/pt_files'
save_path_HE = '/local/data3/chrsp39/CBTN_v2/Merged_HE_SUBJECT_LEVEL_ATTENTION/features/conch_v1/pt_files'

subjects = os.listdir(HE_features)

path_to_splits = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_LGG_vs_HGG_0.7_0.1_0.2_100'

path_to_trained_model = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/LGG_vs_HGG/LGG_vs_HGG_HE_small_clam_sb_conch_v1_s1'

if os.path.exists(save_path_HE) is False:
    os.makedirs(save_path_HE)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% MAIN EXECUTION
if __name__ == "__main__":
    n_folds = 50
    for fold in range(n_folds):
        split_csv = os.path.join(path_to_splits, f'splits_{fold}.csv')
        if not os.path.exists(split_csv):
            print(f"Split file not found: {split_csv}")
            continue
        df = pd.read_csv(split_csv)
        train_subjects = df['train'].dropna().tolist()

        model_path = os.path.join(path_to_trained_model, f's_{fold}_checkpoint.pt')
        if not os.path.exists(model_path):
            print(f"Model checkpoint not found: {model_path}")
            continue
        model = CLAM_SB(gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2, embed_dim=512)
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict_result = model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            state_dict_result = model.load_state_dict(checkpoint, strict=False)
        # Print missing and unexpected keys for debugging
        if hasattr(state_dict_result, 'missing_keys') and hasattr(state_dict_result, 'unexpected_keys'):
            print("Missing keys:", state_dict_result.missing_keys)
            print("Unexpected keys:", state_dict_result.unexpected_keys)
        model = model.to(device)
        model.eval()

        fold_save_path = os.path.join(save_path_HE, f'fold_{fold}')
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)

        for subject in train_subjects:
            feature_file = f'{subject}.pt' if not subject.endswith('.pt') else subject
            feature_path = os.path.join(HE_features, feature_file)
            if not os.path.exists(feature_path):
                print(f"Feature file not found: {feature_path}")
                continue
            features = torch.load(feature_path, map_location=device)
            if isinstance(features, dict) and 'features' in features:
                features = features['features']
            M = get_subject_level_attention(model, features)
            save_file = os.path.join(fold_save_path, feature_file)
            torch.save({'subject_attention': M}, save_file)
            print(f"[Fold {fold}] Saved subject-level attention for {subject} to {save_file}")

# %%
