# %% IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random 

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

# %% SIMPLE MODEL
class Simple_MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Simple_MLP,self).__init__()        
        self.fc = nn.Linear(input_dim, n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        output = self.relu(x)
        return output

# %% LOAD DATA & CONTENTS
# HE DATA & CONTENTS
Merged_HE_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Train_Logits/small_clam_sb/conch/EVAL_5_class_Merged_HE_small_clam_sb_conch'
Merged_HE_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Validation_Logits/small_clam_sb/conch/EVAL_5_class_Merged_HE_small_clam_sb_conch'

HE_train_contents = os.listdir(Merged_HE_train)
HE_train_folds_dict = {} 

for content in HE_train_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_HE_train + '/' + content)
        HE_train_folds_dict[name] = df

HE_val_contents = os.listdir(Merged_HE_val)
HE_val_folds_dict = {}

for content in HE_val_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_HE_val + '/' + content)
        HE_val_folds_dict[name] = df

# KI67 DATA & CONTENTS
Merged_KI67_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Train_Logits/small_clam_sb/conch/EVAL_5_class_Merged_KI67_small_clam_sb_conch'
Merged_KI67_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Validation_Logits/small_clam_sb/conch/EVAL_5_class_Merged_KI67_small_clam_sb_conch'

KI67_train_contents = os.listdir(Merged_KI67_train)
KI67_train_folds_dict = {}

for content in KI67_train_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_KI67_train + '/' + content)
        KI67_train_folds_dict[name] = df  

KI67_val_contents = os.listdir(Merged_KI67_val)
KI67_val_folds_dict = {}

for content in KI67_val_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_KI67_val + '/' + content)
        KI67_val_folds_dict[name] = df

# GFAP DATA & CONTENTS
Merged_GFAP_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Train_Logits/small_clam_sb/conch/EVAL_5_class_Merged_GFAP_small_clam_sb_conch'
Merged_GFAP_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Single_Stain_Validation_Logits/small_clam_sb/conch/EVAL_5_class_Merged_GFAP_small_clam_sb_conch'

GFAP_train_contents = os.listdir(Merged_GFAP_train)
GFAP_train_folds_dict = {}

for content in GFAP_train_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_GFAP_train + '/' + content)
        GFAP_train_folds_dict[name] = df

GFAP_val_contents = os.listdir(Merged_GFAP_val)
GFAP_val_folds_dict = {}

for content in GFAP_val_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_GFAP_val + '/' + content)
        GFAP_val_folds_dict[name] = df

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
        KI67_logits = KI67_train_folds_dict[fold][['logits_0', 'logits_1',  'logits_2', 'logits_3', 'logits_4']].values
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

# %% TRAIN MODEL PER FOLD
num_epochs = 500
all_train_losses = {}
all_train_accuracies = {}
all_val_losses = {}
all_val_accuracies = {}

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Simple_MLP(input_dim, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
            
            print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Late_Fusion/mlp/simple_model/small_clam_sb/conch/EVAL_5_class_Merged_HE_KI67_small_clam_sb_conch/fold_{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# %% PLOT LOSS AND ACCURACY CURVES
plt.figure(figsize=(12, 8))

# Plot train loss curves
plt.subplot(2, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(2, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train accuracy curves
plt.subplot(2, 2, 3)
for fold in folds:
    if fold in all_train_accuracies:
        plt.plot(all_train_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot validation accuracy curves
plt.subplot(2, 2, 4)
for fold in folds:
    if fold in all_val_accuracies:
        plt.plot(all_val_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# %% MERGED HE & GFAP
# Prepare the data
X_train_folds = {}
y_train_folds = {}
X_val_folds = {}
y_val_folds = {}

for fold in folds:
    if fold in HE_train_folds_dict and fold in GFAP_train_folds_dict:
        HE_logits = HE_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        GFAP_logits = GFAP_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, GFAP_logits), axis=1)
        labels = HE_train_folds_dict[fold]['Y'].values
        X_train_folds[fold] = merged_logits
        y_train_folds[fold] = labels
    
    if fold in HE_val_folds_dict and fold in GFAP_val_folds_dict:
        HE_logits = HE_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        GFAP_logits = GFAP_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, GFAP_logits), axis=1)
        labels = HE_val_folds_dict[fold]['Y'].values
        X_val_folds[fold] = merged_logits
        y_val_folds[fold] = labels

# %% TRAIN MODEL PER FOLD
num_epochs = 500
all_train_losses = {}
all_train_accuracies = {}
all_val_losses = {}
all_val_accuracies = {}

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Simple_MLP(input_dim, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
            
            print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Late_Fusion/mlp/simple_model/small_clam_sb/conch/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_conch/fold_{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# %% PLOT LOSS AND ACCURACY CURVES
plt.figure(figsize=(12, 8))

# Plot train loss curves
plt.subplot(2, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(2, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train accuracy curves
plt.subplot(2, 2, 3)
for fold in folds:
    if fold in all_train_accuracies:
        plt.plot(all_train_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot validation accuracy curves
plt.subplot(2, 2, 4)
for fold in folds:
    if fold in all_val_accuracies:
        plt.plot(all_val_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()


# %% MERGED HE, KI67 & GFAP
# Prepare the data
X_train_folds = {}
y_train_folds = {}
X_val_folds = {}
y_val_folds = {}

for fold in folds:
    if fold in HE_train_folds_dict and fold in KI67_train_folds_dict:
        HE_logits = HE_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        GFAP_logits = GFAP_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits, GFAP_logits), axis=1)
        labels = HE_train_folds_dict[fold]['Y'].values
        X_train_folds[fold] = merged_logits
        y_train_folds[fold] = labels
    
    if fold in HE_val_folds_dict and fold in KI67_val_folds_dict:
        HE_logits = HE_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        GFAP_logits = GFAP_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        merged_logits = np.concatenate((HE_logits, KI67_logits, GFAP_logits), axis=1)
        labels = HE_val_folds_dict[fold]['Y'].values
        X_val_folds[fold] = merged_logits
        y_val_folds[fold] = labels

# %% TRAIN MODEL PER FOLD
num_epochs = 500
all_train_losses = {}
all_train_accuracies = {}
all_val_losses = {}
all_val_accuracies = {}

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)

        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = Simple_MLP(input_dim, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                val_accuracies.append(val_accuracy)
            
            print(f'Fold {fold}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        all_train_losses[fold] = train_losses
        all_train_accuracies[fold] = train_accuracies
        all_val_losses[fold] = val_losses
        all_val_accuracies[fold] = val_accuracies

        # Save the model for each fold
        model_save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments/5_class/Late_Fusion/mlp/simple_model/small_clam_sb/conch/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch/fold_{fold}.pth'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(model.state_dict(), model_save_path)

# %% PLOT LOSS AND ACCURACY CURVES
plt.figure(figsize=(12, 8))

# Plot train loss curves
plt.subplot(2, 2, 1)
for fold in folds:
    if fold in all_train_losses:
        plt.plot(all_train_losses[fold], label=f'Fold {fold}')
plt.title('Train Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss curves
plt.subplot(2, 2, 2)
for fold in folds:
    if fold in all_val_losses:
        plt.plot(all_val_losses[fold], label=f'Fold {fold}')
plt.title('Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot train accuracy curves
plt.subplot(2, 2, 3)
for fold in folds:
    if fold in all_train_accuracies:
        plt.plot(all_train_accuracies[fold], label=f'Fold {fold}')
plt.title('Train Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot validation accuracy curves
plt.subplot(2, 2, 4)
for fold in folds:
    if fold in all_val_accuracies:
        plt.plot(all_val_accuracies[fold], label=f'Fold {fold}')
plt.title('Validation Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# %%
