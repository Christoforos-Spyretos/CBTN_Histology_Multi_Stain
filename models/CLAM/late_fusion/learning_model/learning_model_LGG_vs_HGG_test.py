'''
TO DO LIST:
Function for testing the models with arguments:
- loading the model
- loading the test data
- saving the results
'''

# %% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from typing import Dict, Tuple, Optional

# local imports
from late_fusion_models import Single_Layer, One_Hidden_Layer, Two_Hidden_Layer, Attention_Layer

# %% UTILITY FUNCTIONS
# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can choose any seed value


# %% TESTING FUNCTION
def test_model(
    model_type: str,
    X_test_folds: Dict[str, np.ndarray],
    y_test_folds: Dict[str, np.ndarray],
    slide_ids_folds: Dict[str, np.ndarray],
    folds: list,
    model_dir: str,
    results_dir: str,
    hidden_dim: Optional[int] = None,
    hidden_dim1: Optional[int] = None,
    hidden_dim2: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Test a model across multiple folds and save results.
    
    Args:
        model_type: Type of model ('single_layer', 'one_hidden', 'two_hidden', 'attention')
        X_test_folds: Dictionary mapping fold names to test data
        y_test_folds: Dictionary mapping fold names to test labels
        slide_ids_folds: Dictionary mapping fold names to slide IDs
        folds: List of fold names
        model_dir: Directory containing trained model files
        results_dir: Directory to save test results
        hidden_dim: Hidden dimension for one_hidden model
        hidden_dim1: First hidden dimension for two_hidden model
        hidden_dim2: Second hidden dimension for two_hidden model
        verbose: Whether to print progress and results
        
    Returns:
        Dictionary containing test metrics for each fold
    """
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    test_metrics = {
        'losses': {},
        'accuracies': {},
        'balanced_accuracies': {},
        'mcc': {}
    }
    
    for fold in folds:
        if fold in X_test_folds:
            X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
            y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
            slide_ids = slide_ids_folds[fold]
            
            input_dim = X_test.shape[1]
            n_classes = len(np.unique(y_test))
            
            # Initialize model based on type
            if model_type == 'single_layer':
                model = Single_Layer(input_dim, n_classes)
            elif model_type == 'one_hidden':
                if hidden_dim is None:
                    raise ValueError("hidden_dim must be provided for one_hidden model")
                model = One_Hidden_Layer(input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
            elif model_type == 'two_hidden':
                if hidden_dim1 is None or hidden_dim2 is None:
                    raise ValueError("hidden_dim1 and hidden_dim2 must be provided for two_hidden model")
                model = Two_Hidden_Layer(input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, n_classes=n_classes)
            elif model_type == 'attention':
                model = Attention_Layer(input_dim, n_classes)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load model weights
            model_load_path = os.path.join(model_dir, f'{fold}.pth')
            model.load_state_dict(torch.load(model_load_path, weights_only=True))
            model.eval()
            
            # Test the model
            with torch.no_grad():
                outputs = model(X_test)
                probabilities = F.softmax(outputs, dim=1).numpy()
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.numpy()
                
                # Calculate metrics
                loss = F.cross_entropy(outputs, y_test).item()
                accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)
                balanced_acc = balanced_accuracy_score(y_test.numpy(), predicted)
                mcc = matthews_corrcoef(y_test.numpy(), predicted)
                
                test_metrics['losses'][fold] = loss
                test_metrics['accuracies'][fold] = accuracy
                test_metrics['balanced_accuracies'][fold] = balanced_acc
                test_metrics['mcc'][fold] = mcc
            
            # Prepare results DataFrame
            results = pd.DataFrame({
                'slide_id': slide_ids,
                'Y': y_test.numpy(),
                'Y_hat': predicted,
                'p_0': probabilities[:, 0],
                'p_1': probabilities[:, 1],
                'logits_0': outputs[:, 0].numpy(),
                'logits_1': outputs[:, 1].numpy()
            })
            
            # Save results
            results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)
    
    # Calculate and print mean metrics
    if verbose and test_metrics['losses']:
        mean_loss = np.mean(list(test_metrics['losses'].values()))
        mean_accuracy = np.mean(list(test_metrics['accuracies'].values()))
        mean_balanced_acc = np.mean(list(test_metrics['balanced_accuracies'].values()))
        mean_mcc = np.mean(list(test_metrics['mcc'].values()))
        
        print(f'\nMean Test Metrics:')
        print(f'  Loss: {mean_loss:.4f}')
        print(f'  Accuracy: {mean_accuracy:.4f}')
        print(f'  Balanced Accuracy: {mean_balanced_acc:.4f}')
        print(f'  MCC: {mean_mcc:.4f}')
    
    return test_metrics

    
# %% LOAD DATA & CONTENTS
# HE DATA & CONTENTS
HE_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1_5'

HE_test_contents = os.listdir(HE_test)
HE_test_folds_dict = {} 

for content in HE_test_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_test + '/' + content)
        HE_test_folds_dict[name] = df

# KI67 DATA & CONTENTS
KI67_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5'

KI67_test_contents = os.listdir(KI67_test)
KI67_test_folds_dict = {}

for content in KI67_test_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(KI67_test + '/' + content)
        KI67_test_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED HE & KI67
# Prepare the data
X_test_folds = {}
y_test_folds = {}
slide_ids_folds = {}

for fold in folds:
    if fold in HE_test_folds_dict and fold in KI67_test_folds_dict:
        HE_logits = HE_test_folds_dict[fold][['logits_0', 'logits_1']].values
        KI67_logits = KI67_test_folds_dict[fold][['logits_0', 'logits_1']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        labels = HE_test_folds_dict[fold]['Y'].values
        slide_ids = HE_test_folds_dict[fold]['slide_id'].values
        X_test_folds[fold] = merged_logits
        y_test_folds[fold] = labels
        slide_ids_folds[fold] = slide_ids

# %% TEST SINGLE LAYER MODEL
print("=" * 80)
print("Testing Single Layer Model")
print("=" * 80)

test_metrics_single = test_model(
    model_type='single_layer',
    X_test_folds=X_test_folds,
    y_test_folds=y_test_folds,
    slide_ids_folds=slide_ids_folds,
    folds=folds,
    model_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/50%_split/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5',
    results_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5',
    verbose=True
)

# %% TEST ONE HIDDEN LAYER MODEL
print("=" * 80)
print("Testing One Hidden Layer Model")
print("=" * 80)

test_metrics_one_hidden = test_model(
    model_type='one_hidden',
    X_test_folds=X_test_folds,
    y_test_folds=y_test_folds,
    slide_ids_folds=slide_ids_folds,
    folds=folds,
    model_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/50%_split/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5',
    results_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5',
    hidden_dim=4,
    verbose=True
)

# %% TEST TWO HIDDEN LAYER MODEL
print("=" * 80)
print("Testing Two Hidden Layer Model")
print("=" * 80)

test_metrics_two_hidden = test_model(
    model_type='two_hidden',
    X_test_folds=X_test_folds,
    y_test_folds=y_test_folds,
    slide_ids_folds=slide_ids_folds,
    folds=folds,
    model_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/50%_split/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5',
    results_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5',
    hidden_dim1=6,
    hidden_dim2=2,
    verbose=True
)

# %% TEST ATTENTION LAYER MODEL
print("=" * 80)
print("Testing Attention Layer Model")
print("=" * 80)

test_metrics_attention = test_model(
    model_type='attention',
    X_test_folds=X_test_folds,
    y_test_folds=y_test_folds,
    slide_ids_folds=slide_ids_folds,
    folds=folds,
    model_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/50%_split/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5',
    results_dir='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5',
    verbose=True
)

# %%