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
HE_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1'

HE_test_contents = os.listdir(HE_test)
HE_test_folds_dict = {} 

for content in HE_test_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_test + '/' + content)
        HE_test_folds_dict[name] = df

# KI67 DATA & CONTENTS
KI67_test = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1'

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
        HE_logits = HE_test_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_test_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        labels = HE_test_folds_dict[fold]['Y'].values
        slide_ids = HE_test_folds_dict[fold]['slide_id'].values
        X_test_folds[fold] = merged_logits
        y_test_folds[fold] = labels
        slide_ids_folds[fold] = slide_ids

# %% TEST SIMPLE MODEL
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}
test_balanced_accuracies = {}
test_mcc = {}
test_balanced_accuracies = {}
test_mcc = {}

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = Single_Layer(input_dim, n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)
            balanced_acc = balanced_accuracy_score(y_test.numpy(), predicted)
            mcc = matthews_corrcoef(y_test.numpy(), predicted)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy
            test_balanced_accuracies[fold] = balanced_acc
            test_mcc[fold] = mcc

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1],
            'p_2': probabilities[:, 2],
            'p_3': probabilities[:, 3],
            'p_4': probabilities[:, 4]
        })

        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean metrics per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))
mean_test_balanced_accuracy = np.mean(list(test_balanced_accuracies.values()))
mean_test_mcc = np.mean(list(test_mcc.values()))

print(f'Single Layer Model - Mean Test Loss: {mean_test_loss:.4f}')
print(f'Single Layer Model - Mean Test Accuracy: {mean_test_accuracy:.4f}')
print(f'Single Layer Model - Mean Test Balanced Accuracy: {mean_test_balanced_accuracy:.4f}')
print(f'Single Layer Model - Mean Test MCC: {mean_test_mcc:.4f}')

# %% TEST ONE HIDDEN LAYER MODEL
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}
test_balanced_accuracies = {}
test_mcc = {}

hidden_dim = 10

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = One_Hidden_Layer(input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)
            balanced_acc = balanced_accuracy_score(y_test.numpy(), predicted)
            mcc = matthews_corrcoef(y_test.numpy(), predicted)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy
            test_balanced_accuracies[fold] = balanced_acc
            test_mcc[fold] = mcc

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1],
            'p_2': probabilities[:, 2],
            'p_3': probabilities[:, 3],
            'p_4': probabilities[:, 4]
        })
    
        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean metrics per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))
mean_test_balanced_accuracy = np.mean(list(test_balanced_accuracies.values()))
mean_test_mcc = np.mean(list(test_mcc.values()))

print(f'One Hidden Layer Model - Mean Test Loss: {mean_test_loss:.4f}')
print(f'One Hidden Layer Model - Mean Test Accuracy: {mean_test_accuracy:.4f}')
print(f'One Hidden Layer Model - Mean Test Balanced Accuracy: {mean_test_balanced_accuracy:.4f}')
print(f'One Hidden Layer Model - Mean Test MCC: {mean_test_mcc:.4f}')

# %% TEST ATTENTION BASED MODEL
results_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

test_losses = {}
test_accuracies = {}
test_balanced_accuracies = {}
test_mcc = {}

for fold in folds:
    if fold in X_test_folds:
        X_test = torch.tensor(X_test_folds[fold], dtype=torch.float32)
        y_test = torch.tensor(y_test_folds[fold], dtype=torch.long)
        slide_ids = slide_ids_folds[fold]

        input_dim = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        model = Attention_Layer(input_dim, n_classes)
        model_load_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/fold_{fold}.pth'
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        with torch.no_grad():
            outputs = model(X_test)
            probabilities = F.softmax(outputs, dim=1).numpy()
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.numpy()

            loss = F.cross_entropy(outputs, y_test).item()
            accuracy = (predicted == y_test.numpy()).sum().item() / y_test.size(0)
            balanced_acc = balanced_accuracy_score(y_test.numpy(), predicted)
            mcc = matthews_corrcoef(y_test.numpy(), predicted)

            test_losses[fold] = loss
            test_accuracies[fold] = accuracy
            test_balanced_accuracies[fold] = balanced_acc
            test_mcc[fold] = mcc

        results = pd.DataFrame({
            'slide_id': slide_ids,
            'Y': y_test.numpy(),
            'Y_hat': predicted,
            'p_0': probabilities[:, 0],
            'p_1': probabilities[:, 1],
            'p_2': probabilities[:, 2],
            'p_3': probabilities[:, 3],
            'p_4': probabilities[:, 4]
        })

        results.to_csv(os.path.join(results_dir, f'{fold}.csv'), index=False)

# Calculate and print mean metrics per fold
mean_test_loss = np.mean(list(test_losses.values()))
mean_test_accuracy = np.mean(list(test_accuracies.values()))
mean_test_balanced_accuracy = np.mean(list(test_balanced_accuracies.values()))
mean_test_mcc = np.mean(list(test_mcc.values()))

print(f'Attention Layer Model - Mean Test Loss: {mean_test_loss:.4f}')
print(f'Attention Layer Model - Mean Test Accuracy: {mean_test_accuracy:.4f}')
print(f'Attention Layer Model - Mean Test Balanced Accuracy: {mean_test_balanced_accuracy:.4f}')
print(f'Attention Layer Model - Mean Test MCC: {mean_test_mcc:.4f}')

# %%