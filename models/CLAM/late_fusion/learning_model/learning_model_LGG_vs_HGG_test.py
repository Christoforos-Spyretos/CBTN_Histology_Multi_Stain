# %% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random 

# local imports
from late_fusion_models import Single_Layer, One_Hidden_Layer, Attention_Layer

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
    
# %% LOAD DATA & CONTENTS
# HE DATA & CONTENTS
HE_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1'

HE_test_contents = os.listdir(HE_test)
HE_test_folds_dict = {} 

for content in HE_test_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_test + '/' + content)
        HE_test_folds_dict[name] = df

# KI67 DATA & CONTENTS
KI67_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1'

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
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = Single_Layer(input_dim, n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1]
        })

        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean accuracy and loss per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))

print(f'Mean Test Loss: {mean_test_loss}')
print(f'Mean Test Accuracy: {mean_test_accuracy}')

# %% TEST ONE HIDDEN LAYER MODEL
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}

hidden_dim = 4

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = One_Hidden_Layer(input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1]
        })

        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean accuracy and loss per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))

print(f'Mean Test Loss: {mean_test_loss}')
print(f'Mean Test Accuracy: {mean_test_accuracy}')

# %% TEST ATTENTION LAYER MODEL
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = Attention_Layer(input_dim, n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/LGG_vs_HGG/LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1]
        })

        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean accuracy and loss per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))

print(f'Mean Test Loss: {mean_test_loss}')
print(f'Mean Test Accuracy: {mean_test_accuracy}')

# %%