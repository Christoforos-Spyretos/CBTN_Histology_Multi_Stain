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
from sklearn.utils.class_weight import compute_class_weight 

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

set_seed(42)  # choose any seed value

# %% LOAD DATA & CONTENTS
# HE DATA & CONTENTS
HE_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_HE_train_logits_small_clam_sb_conch_v1'
HE_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_HE_val_logits_small_clam_sb_conch_v1'

HE_train_contents = os.listdir(HE_train)
HE_train_folds_dict = {}

for content in HE_train_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_train + '/' + content)
        HE_train_folds_dict[name] = df

HE_val_contents = os.listdir(HE_val)
HE_val_folds_dict = {}

for content in HE_val_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_val + '/' + content)
        HE_val_folds_dict[name] = df

# KI67 DATA & CONTENTS
KI67_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_KI67_train_logits_small_clam_sb_conch_v1'
KI67_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_KI67_val_logits_small_clam_sb_conch_v1'

KI67_train_contents = os.listdir(KI67_train)
KI67_train_folds_dict = {}

for content in KI67_train_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(KI67_train + '/' + content)
        KI67_train_folds_dict[name] = df

KI67_val_contents = os.listdir(KI67_val)
KI67_val_folds_dict = {}

for content in KI67_val_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(KI67_val + '/' + content)
        KI67_val_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED HE & KI67
# Prepare the data
X_train_folds = {}
y_train_folds = {}
X_val_folds = {}
y_val_folds = {}

for fold in folds:
    if fold in HE_train_folds_dict and fold in KI67_train_folds_dict:
        HE_logits = HE_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        labels = HE_train_folds_dict[fold]['Y'].values
        X_train_folds[fold] = merged_logits
        y_train_folds[fold] = labels
    
    if fold in HE_val_folds_dict and fold in KI67_val_folds_dict:
        HE_logits = HE_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        labels = HE_val_folds_dict[fold]['Y'].values
        X_val_folds[fold] = merged_logits
        y_val_folds[fold] = labels

# %% TRAIN SIMPLE MODEL
num_epochs = 1200
all_train_losses = {}
all_train_accuracies = {}
all_train_balanced_accuracies = {}
all_train_mcc = {}
all_val_losses = {}
all_val_accuracies = {}
all_val_balanced_accuracies = {}
all_val_mcc = {}

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Single_Layer(input_dim, n_classes)
        
        # Calculate class weights for handling class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses = []
        train_accuracies = []
        train_balanced_accuracies = []
        train_mcc = []
        val_losses = []
        val_accuracies = []
        val_balanced_accuracies = []
        val_mcc = []
        
        # L1 regularization parameter
        l1_lambda = 0.001
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            
            # Calculate L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, y_train) + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            # Calculate balanced accuracy and MCC for training
            train_balanced_acc = balanced_accuracy_score(y_train.numpy(), predicted.numpy())
            train_balanced_accuracies.append(train_balanced_acc)
            train_mcc_score = matthews_corrcoef(y_train.numpy(), predicted.numpy())
            train_mcc.append(train_mcc_score)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
                
                # Calculate balanced accuracy and MCC for validation
                val_balanced_acc = balanced_accuracy_score(y_val.numpy(), val_predicted.numpy())
                val_balanced_accuracies.append(val_balanced_acc)
                val_mcc_score_val = matthews_corrcoef(y_val.numpy(), val_predicted.numpy())
                val_mcc.append(val_mcc_score_val)
            
            print(f'{fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}, '
                  f'Train MCC: {train_mcc_score:.4f}, Val MCC: {val_mcc_score_val:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_train_balanced_accuracies[fold] = train_balanced_accuracies
        all_train_mcc[fold] = train_mcc
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies
        all_val_balanced_accuracies[fold] = val_balanced_accuracies
        all_val_mcc[fold] = val_mcc

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# plot loss, balanced accuracy and MCC curves in one plot
plt.figure(figsize=(16, 10))

# Plot train loss curves
plt.subplot(3, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(3, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train balanced accuracy curves
plt.subplot(3, 2, 3)
for fold in folds:
    if fold in all_train_balanced_accuracies:
        plt.plot(all_train_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot validation balanced accuracy curves
plt.subplot(3, 2, 4)
for fold in folds:
    if fold in all_val_balanced_accuracies:
        plt.plot(all_val_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot train MCC curves
plt.subplot(3, 2, 5)
for fold in folds:
    if fold in all_train_mcc:
        plt.plot(all_train_mcc[fold], label=f'Fold {fold}')
plt.title('Train MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')

# Plot validation MCC curves
plt.subplot(3, 2, 6)
for fold in folds:
    if fold in all_val_mcc:
        plt.plot(all_val_mcc[fold], label=f'Fold {fold}')
plt.title('Validation MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')

plt.tight_layout()
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/plot.png')
plt.show()

# aggregate plots across folds (mean ± std)
def _stack_metrics(d):
    if not d:
        return None, 0
    min_len = min(len(v) for v in d.values())
    if min_len == 0:
        return None, 0
    arr = np.stack([np.array(v[:min_len]) for v in d.values()], axis=0)
    return arr, min_len

tr_loss_arr, T = _stack_metrics(all_train_losses)
val_loss_arr, _ = _stack_metrics(all_val_losses)
tr_bal_acc_arr, _ = _stack_metrics(all_train_balanced_accuracies)
val_bal_acc_arr, _ = _stack_metrics(all_val_balanced_accuracies)
tr_mcc_arr, _ = _stack_metrics(all_train_mcc)
val_mcc_arr, _ = _stack_metrics(all_val_mcc)

if T > 0:
    epochs = np.arange(T)
    plt.figure(figsize=(16, 10))

    # train loss
    plt.subplot(3, 2, 1)
    if tr_loss_arr is not None:
        m = tr_loss_arr.mean(axis=0)
        s = tr_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # val loss
    plt.subplot(3, 2, 2)
    if val_loss_arr is not None:
        m = val_loss_arr.mean(axis=0)
        s = val_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # train balanced accuracy
    plt.subplot(3, 2, 3)
    if tr_bal_acc_arr is not None:
        m = tr_bal_acc_arr.mean(axis=0)
        s = tr_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # val balanced accuracy
    plt.subplot(3, 2, 4)
    if val_bal_acc_arr is not None:
        m = val_bal_acc_arr.mean(axis=0)
        s = val_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # train MCC
    plt.subplot(3, 2, 5)
    if tr_mcc_arr is not None:
        m = tr_mcc_arr.mean(axis=0)
        s = tr_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    # val MCC
    plt.subplot(3, 2, 6)
    if val_mcc_arr is not None:
        m = val_mcc_arr.mean(axis=0)
        s = val_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/plot_aggregate.png')
    plt.show()

# %% TRAIN ONE HIDDEN LAYER MODEL
num_epochs = 1200
all_train_losses = {}
all_train_accuracies = {}
all_train_balanced_accuracies = {}
all_train_mcc = {}
all_val_losses = {}
all_val_accuracies = {}
all_val_balanced_accuracies = {}
all_val_mcc = {}

hidden_dim = 15

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = One_Hidden_Layer(input_dim, hidden_dim=hidden_dim, n_classes=n_classes)
        
        # Calculate class weights for handling class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses = []
        train_accuracies = []
        train_balanced_accuracies = []
        train_mcc = []
        val_losses = []
        val_accuracies = []
        val_balanced_accuracies = []
        val_mcc = []
        
        # L1 regularization parameter
        l1_lambda = 0.001
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            
            # Calculate L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, y_train) + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            # Calculate balanced accuracy and MCC for training
            train_balanced_acc = balanced_accuracy_score(y_train.numpy(), predicted.numpy())
            train_balanced_accuracies.append(train_balanced_acc)
            train_mcc_score = matthews_corrcoef(y_train.numpy(), predicted.numpy())
            train_mcc.append(train_mcc_score)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
                
                # Calculate balanced accuracy and MCC for validation
                val_balanced_acc = balanced_accuracy_score(y_val.numpy(), val_predicted.numpy())
                val_balanced_accuracies.append(val_balanced_acc)
                val_mcc_score_val = matthews_corrcoef(y_val.numpy(), val_predicted.numpy())
                val_mcc.append(val_mcc_score_val)
            
            print(f'{fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}, '
                  f'Train MCC: {train_mcc_score:.4f}, Val MCC: {val_mcc_score_val:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_train_balanced_accuracies[fold] = train_balanced_accuracies
        all_train_mcc[fold] = train_mcc
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies
        all_val_balanced_accuracies[fold] = val_balanced_accuracies
        all_val_mcc[fold] = val_mcc

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# plot loss, balanced accuracy and MCC curves in one plot
plt.figure(figsize=(16, 10))

# Plot train loss curves
plt.subplot(3, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(3, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train balanced accuracy curves
plt.subplot(3, 2, 3)
for fold in folds:
    if fold in all_train_balanced_accuracies:
        plt.plot(all_train_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot validation balanced accuracy curves
plt.subplot(3, 2, 4)
for fold in folds:
    if fold in all_val_balanced_accuracies:
        plt.plot(all_val_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot train MCC curves
plt.subplot(3, 2, 5)
for fold in folds:
    if fold in all_train_mcc:
        plt.plot(all_train_mcc[fold], label=f'Fold {fold}')
plt.title('Train MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')

# Plot validation MCC curves
plt.subplot(3, 2, 6)
for fold in folds:
    if fold in all_val_mcc:
        plt.plot(all_val_mcc[fold], label=f'Fold {fold}')
plt.title('Validation MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')
plt.tight_layout()

# save plot
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/plot.png')
plt.show()

# Aggregate plots across folds (mean ± std)
def _stack_metrics(d):
    if not d:
        return None, 0
    min_len = min(len(v) for v in d.values())
    if min_len == 0:
        return None, 0
    arr = np.stack([np.array(v[:min_len]) for v in d.values()], axis=0)
    return arr, min_len

tr_loss_arr, T = _stack_metrics(all_train_losses)
val_loss_arr, _ = _stack_metrics(all_val_losses)
tr_bal_acc_arr, _ = _stack_metrics(all_train_balanced_accuracies)
val_bal_acc_arr, _ = _stack_metrics(all_val_balanced_accuracies)
tr_mcc_arr, _ = _stack_metrics(all_train_mcc)
val_mcc_arr, _ = _stack_metrics(all_val_mcc)

if T > 0:
    epochs = np.arange(T)
    plt.figure(figsize=(16, 10))

    # Train loss
    plt.subplot(3, 2, 1)
    if tr_loss_arr is not None:
        m = tr_loss_arr.mean(axis=0)
        s = tr_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Val loss
    plt.subplot(3, 2, 2)
    if val_loss_arr is not None:
        m = val_loss_arr.mean(axis=0)
        s = val_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Train balanced accuracy
    plt.subplot(3, 2, 3)
    if tr_bal_acc_arr is not None:
        m = tr_bal_acc_arr.mean(axis=0)
        s = tr_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Val balanced accuracy
    plt.subplot(3, 2, 4)
    if val_bal_acc_arr is not None:
        m = val_bal_acc_arr.mean(axis=0)
        s = val_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Train MCC
    plt.subplot(3, 2, 5)
    if tr_mcc_arr is not None:
        m = tr_mcc_arr.mean(axis=0)
        s = tr_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    # Val MCC
    plt.subplot(3, 2, 6)
    if val_mcc_arr is not None:
        m = val_mcc_arr.mean(axis=0)
        s = val_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/plot_aggregate.png')
    plt.show()

# %% TRAIN TWO HIDDEN LAYER MODEL
num_epochs = 1200
all_train_losses = {}
all_train_accuracies = {}
all_train_balanced_accuracies = {}
all_train_mcc = {}
all_val_losses = {}
all_val_accuracies = {}
all_val_balanced_accuracies = {}
all_val_mcc = {}

hidden_dim1 = 15
hidden_dim2 = 10

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Two_Hidden_Layer(input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, n_classes=n_classes)
        
        # Calculate class weights for handling class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses = []
        train_accuracies = []
        train_balanced_accuracies = []
        train_mcc = []
        val_losses = []
        val_accuracies = []
        val_balanced_accuracies = []
        val_mcc = []
        
        # L1 regularization parameter
        l1_lambda = 0.001
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            
            # Calculate L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, y_train) + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            # Calculate balanced accuracy and MCC for training
            train_balanced_acc = balanced_accuracy_score(y_train.numpy(), predicted.numpy())
            train_balanced_accuracies.append(train_balanced_acc)
            train_mcc_score = matthews_corrcoef(y_train.numpy(), predicted.numpy())
            train_mcc.append(train_mcc_score)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
                
                # Calculate balanced accuracy and MCC for validation
                val_balanced_acc = balanced_accuracy_score(y_val.numpy(), val_predicted.numpy())
                val_balanced_accuracies.append(val_balanced_acc)
                val_mcc_score_val = matthews_corrcoef(y_val.numpy(), val_predicted.numpy())
                val_mcc.append(val_mcc_score_val)
            
            print(f'{fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}, '
                  f'Train MCC: {train_mcc_score:.4f}, Val MCC: {val_mcc_score_val:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_train_balanced_accuracies[fold] = train_balanced_accuracies
        all_train_mcc[fold] = train_mcc
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies
        all_val_balanced_accuracies[fold] = val_balanced_accuracies
        all_val_mcc[fold] = val_mcc

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1/{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# plot loss, balanced accuracy and MCC curves in one plot
plt.figure(figsize=(16, 10))

# Plot train loss curves
plt.subplot(3, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(3, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train balanced accuracy curves
plt.subplot(3, 2, 3)
for fold in folds:
    if fold in all_train_balanced_accuracies:
        plt.plot(all_train_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot validation balanced accuracy curves
plt.subplot(3, 2, 4)
for fold in folds:
    if fold in all_val_balanced_accuracies:
        plt.plot(all_val_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot train MCC curves
plt.subplot(3, 2, 5)
for fold in folds:
    if fold in all_train_mcc:
        plt.plot(all_train_mcc[fold], label=f'Fold {fold}')
plt.title('Train MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')

# Plot validation MCC curves
plt.subplot(3, 2, 6)
for fold in folds:
    if fold in all_val_mcc:
        plt.plot(all_val_mcc[fold], label=f'Fold {fold}')
plt.title('Validation MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')
plt.tight_layout()

# save plot
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1/plot.png')
plt.show()

# Aggregate plots across folds (mean ± std)
def _stack_metrics(d):
    if not d:
        return None, 0
    min_len = min(len(v) for v in d.values())
    if min_len == 0:
        return None, 0
    arr = np.stack([np.array(v[:min_len]) for v in d.values()], axis=0)
    return arr, min_len

tr_loss_arr, T = _stack_metrics(all_train_losses)
val_loss_arr, _ = _stack_metrics(all_val_losses)
tr_bal_acc_arr, _ = _stack_metrics(all_train_balanced_accuracies)
val_bal_acc_arr, _ = _stack_metrics(all_val_balanced_accuracies)
tr_mcc_arr, _ = _stack_metrics(all_train_mcc)
val_mcc_arr, _ = _stack_metrics(all_val_mcc)

if T > 0:
    epochs = np.arange(T)
    plt.figure(figsize=(16, 10))

    # Train loss
    plt.subplot(3, 2, 1)
    if tr_loss_arr is not None:
        m = tr_loss_arr.mean(axis=0)
        s = tr_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Val loss
    plt.subplot(3, 2, 2)
    if val_loss_arr is not None:
        m = val_loss_arr.mean(axis=0)
        s = val_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Train balanced accuracy
    plt.subplot(3, 2, 3)
    if tr_bal_acc_arr is not None:
        m = tr_bal_acc_arr.mean(axis=0)
        s = tr_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Val balanced accuracy
    plt.subplot(3, 2, 4)
    if val_bal_acc_arr is not None:
        m = val_bal_acc_arr.mean(axis=0)
        s = val_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Train MCC
    plt.subplot(3, 2, 5)
    if tr_mcc_arr is not None:
        m = tr_mcc_arr.mean(axis=0)
        s = tr_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    # Val MCC
    plt.subplot(3, 2, 6)
    if val_mcc_arr is not None:
        m = val_mcc_arr.mean(axis=0)
        s = val_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1/plot_aggregate.png')
    plt.show()

# %% TRAIN ATTENTION BASED MODEL
num_epochs = 1200
all_train_losses = {}
all_train_accuracies = {}
all_train_balanced_accuracies = {}
all_train_mcc = {}
all_val_losses = {}
all_val_accuracies = {}
all_val_balanced_accuracies = {}
all_val_mcc = {}

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Attention_Layer(input_dim, n_classes)
        
        # Calculate class weights for handling class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses = []
        train_accuracies = []
        train_balanced_accuracies = []
        train_mcc = []
        val_losses = []
        val_accuracies = []
        val_balanced_accuracies = []
        val_mcc = []
        
        # L1 regularization parameter
        l1_lambda = 0.001
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            
            # Calculate L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, y_train) + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            # Calculate balanced accuracy and MCC for training
            train_balanced_acc = balanced_accuracy_score(y_train.numpy(), predicted.numpy())
            train_balanced_accuracies.append(train_balanced_acc)
            train_mcc_score = matthews_corrcoef(y_train.numpy(), predicted.numpy())
            train_mcc.append(train_mcc_score)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
                
                # Calculate balanced accuracy and MCC for validation
                val_balanced_acc = balanced_accuracy_score(y_val.numpy(), val_predicted.numpy())
                val_balanced_accuracies.append(val_balanced_acc)
                val_mcc_score_val = matthews_corrcoef(y_val.numpy(), val_predicted.numpy())
                val_mcc.append(val_mcc_score_val)
            
            print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, 'f'Val Loss: {val_loss.item():.4f}, '
                  f'Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}, '
                  f'Train MCC: {train_mcc_score:.4f}, Val MCC: {val_mcc_score_val:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_train_balanced_accuracies[fold] = train_balanced_accuracies
        all_train_mcc[fold] = train_mcc
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies
        all_val_balanced_accuracies[fold] = val_balanced_accuracies
        all_val_mcc[fold] = val_mcc

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# plot loss, balanced accuracy and MCC curves in one plot
plt.figure(figsize=(16, 10))

# Plot train loss curves
plt.subplot(3, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(3, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train balanced accuracy curves
plt.subplot(3, 2, 3)
for fold in folds:
    if fold in all_train_balanced_accuracies:
        plt.plot(all_train_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot validation balanced accuracy curves
plt.subplot(3, 2, 4)
for fold in folds:
    if fold in all_val_balanced_accuracies:
        plt.plot(all_val_balanced_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Balanced Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')

# Plot train MCC curves
plt.subplot(3, 2, 5)
for fold in folds:
    if fold in all_train_mcc:
        plt.plot(all_train_mcc[fold], label=f'Fold {fold}')
plt.title('Train MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')

# Plot validation MCC curves
plt.subplot(3, 2, 6)
for fold in folds:
    if fold in all_val_mcc:
        plt.plot(all_val_mcc[fold], label=f'Fold {fold}')
plt.title('Validation MCC Curves')
plt.xlabel('Epoch')
plt.ylabel('MCC')
plt.tight_layout()

# save plot
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/plot.png')
plt.show()

# Aggregate plots across folds (mean ± std)
def _stack_metrics(d):
    if not d:
        return None, 0
    min_len = min(len(v) for v in d.values())
    if min_len == 0:
        return None, 0
    arr = np.stack([np.array(v[:min_len]) for v in d.values()], axis=0)
    return arr, min_len

tr_loss_arr, T = _stack_metrics(all_train_losses)
val_loss_arr, _ = _stack_metrics(all_val_losses)
tr_bal_acc_arr, _ = _stack_metrics(all_train_balanced_accuracies)
val_bal_acc_arr, _ = _stack_metrics(all_val_balanced_accuracies)
tr_mcc_arr, _ = _stack_metrics(all_train_mcc)
val_mcc_arr, _ = _stack_metrics(all_val_mcc)

if T > 0:
    epochs = np.arange(T)
    plt.figure(figsize=(16, 10))

    # Train loss
    plt.subplot(3, 2, 1)
    if tr_loss_arr is not None:
        m = tr_loss_arr.mean(axis=0)
        s = tr_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Val loss
    plt.subplot(3, 2, 2)
    if val_loss_arr is not None:
        m = val_loss_arr.mean(axis=0)
        s = val_loss_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Loss (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Train balanced accuracy
    plt.subplot(3, 2, 3)
    if tr_bal_acc_arr is not None:
        m = tr_bal_acc_arr.mean(axis=0)
        s = tr_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Val balanced accuracy
    plt.subplot(3, 2, 4)
    if val_bal_acc_arr is not None:
        m = val_bal_acc_arr.mean(axis=0)
        s = val_bal_acc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation Balanced Accuracy (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    # Train MCC
    plt.subplot(3, 2, 5)
    if tr_mcc_arr is not None:
        m = tr_mcc_arr.mean(axis=0)
        s = tr_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C0', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='±1 std')
    plt.title('Train MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    # Val MCC
    plt.subplot(3, 2, 6)
    if val_mcc_arr is not None:
        m = val_mcc_arr.mean(axis=0)
        s = val_mcc_arr.std(axis=0)
        plt.plot(epochs, m, color='C1', label='Mean')
        plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='±1 std')
    plt.title('Validation MCC (mean ± std)')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/plot_aggregate.png')
    plt.show()

# %%
