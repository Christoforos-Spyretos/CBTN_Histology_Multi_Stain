# %% IMPORTS
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight 
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler

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

# Early stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 50
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity
                       monitored has stopped decreasing; in 'max' mode it will stop when the quantity
                       monitored has stopped increasing.
                            Default: 'min'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_epoch = 0
        
    def __call__(self, val_metric, epoch):
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

def train_and_evaluate_model(
    model_type: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 1200,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    use_early_stopping: bool = False,
    patience: int = 50,
    use_l1_regularization: bool = False,
    l1_lambda: float = 0.001,
    use_class_weights: bool = True,
    save_path: Optional[str] = None,
    fold_name: str = 'fold',
    hidden_dim: Optional[int] = None,
    hidden_dim1: Optional[int] = None,
    hidden_dim2: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train and evaluate a model with specified configuration.
    
    Args:
        model_type: Type of model ('single_layer', 'one_hidden', 'two_hidden', 'attention')
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer (L2 regularization)
        use_early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        use_l1_regularization: Whether to use L1 regularization
        l1_lambda: Lambda parameter for L1 regularization
        use_class_weights: Whether to use class weights for imbalanced data
        save_path: Path to save the trained model
        fold_name: Name of the fold for logging
        hidden_dim: Hidden dimension for one_hidden model
        hidden_dim1: First hidden dimension for two_hidden model
        hidden_dim2: Second hidden dimension for two_hidden model
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training and validation metrics
    """
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
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
    
    # Setup loss function with optional class weights
    if use_class_weights:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
        class_weights_tensor = torch.FloatTensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize early stopping if needed
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=verbose, mode='min')
    
    # Tracking metrics
    metrics = {
        'train_losses': [],
        'train_accuracies': [],
        'train_balanced_accuracies': [],
        'train_mcc': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_balanced_accuracies': [],
        'val_mcc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        
        # Calculate loss with optional L1 regularization
        loss = criterion(outputs, y_train)
        if use_l1_regularization:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
        metrics['train_losses'].append(loss.item())
        
        # Calculate training metrics
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
        metrics['train_accuracies'].append(train_accuracy)
        
        train_balanced_acc = balanced_accuracy_score(y_train.numpy(), predicted.numpy())
        metrics['train_balanced_accuracies'].append(train_balanced_acc)
        
        train_mcc_score = matthews_corrcoef(y_train.numpy(), predicted.numpy())
        metrics['train_mcc'].append(train_mcc_score)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            metrics['val_losses'].append(val_loss.item())
            
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
            metrics['val_accuracies'].append(val_accuracy)
            
            val_balanced_acc = balanced_accuracy_score(y_val.numpy(), val_predicted.numpy())
            metrics['val_balanced_accuracies'].append(val_balanced_acc)
            
            val_mcc_score = matthews_corrcoef(y_val.numpy(), val_predicted.numpy())
            metrics['val_mcc'].append(val_mcc_score)
        
        # Print progress
        if verbose:
            print(f'{fold_name}, Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Train Bal Acc: {train_balanced_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}, '
                  f'Train MCC: {train_mcc_score:.4f}, Val MCC: {val_mcc_score:.4f}')
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss.item(), epoch)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1} for {fold_name}")
                    print(f"Best epoch was {early_stopping.best_epoch+1}")
                
                # Fill remaining epochs with NaN
                remaining_epochs = num_epochs - (epoch + 1)
                for _ in range(remaining_epochs):
                    metrics['train_losses'].append(np.nan)
                    metrics['train_accuracies'].append(np.nan)
                    metrics['train_balanced_accuracies'].append(np.nan)
                    metrics['train_mcc'].append(np.nan)
                    metrics['val_losses'].append(np.nan)
                    metrics['val_accuracies'].append(np.nan)
                    metrics['val_balanced_accuracies'].append(np.nan)
                    metrics['val_mcc'].append(np.nan)
                break
    
    # Save model if path is provided
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    
    return metrics


def plot_training_curves(
    all_metrics: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title_prefix: str = ''
):
    """
    Plot training curves for all folds.
    
    Args:
        all_metrics: Dictionary mapping fold names to their metrics
        save_path: Path to save the plot
        title_prefix: Optional prefix for plot titles
    """
    plt.figure(figsize=(16, 10))
    
    # Plot train loss curves
    plt.subplot(3, 2, 1)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['train_losses'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Train Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot validation loss curves
    plt.subplot(3, 2, 2)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['val_losses'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot train balanced accuracy curves
    plt.subplot(3, 2, 3)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['train_balanced_accuracies'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Train Balanced Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    
    # Plot validation balanced accuracy curves
    plt.subplot(3, 2, 4)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['val_balanced_accuracies'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Validation Balanced Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    
    # Plot train MCC curves
    plt.subplot(3, 2, 5)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['train_mcc'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Train MCC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    
    # Plot validation MCC curves
    plt.subplot(3, 2, 6)
    for fold_name, metrics in all_metrics.items():
        plt.plot(metrics['val_mcc'], label=f'{fold_name}')
    plt.title(f'{title_prefix}Validation MCC Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.show()


def plot_aggregate_curves(
    all_metrics: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title_prefix: str = ''
):
    """
    Plot aggregate (mean ± std) training curves across folds.
    
    Args:
        all_metrics: Dictionary mapping fold names to their metrics
        save_path: Path to save the plot
        title_prefix: Optional prefix for plot titles
    """
    def _stack_metrics(metric_name: str) -> Tuple[Optional[np.ndarray], int]:
        metric_lists = [metrics[metric_name] for metrics in all_metrics.values() if metric_name in metrics]
        if not metric_lists:
            return None, 0
        min_len = min(len(v) for v in metric_lists)
        if min_len == 0:
            return None, 0
        arr = np.stack([np.array(v[:min_len]) for v in metric_lists], axis=0)
        return arr, min_len
    
    tr_loss_arr, T = _stack_metrics('train_losses')
    val_loss_arr, _ = _stack_metrics('val_losses')
    tr_bal_acc_arr, _ = _stack_metrics('train_balanced_accuracies')
    val_bal_acc_arr, _ = _stack_metrics('val_balanced_accuracies')
    tr_mcc_arr, _ = _stack_metrics('train_mcc')
    val_mcc_arr, _ = _stack_metrics('val_mcc')
    
    if T > 0:
        epochs = np.arange(T)
        plt.figure(figsize=(16, 10))
        
        # Train loss
        plt.subplot(3, 2, 1)
        if tr_loss_arr is not None:
            m = np.nanmean(tr_loss_arr, axis=0)
            s = np.nanstd(tr_loss_arr, axis=0)
            plt.plot(epochs, m, color='C0', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Train Loss (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Val loss
        plt.subplot(3, 2, 2)
        if val_loss_arr is not None:
            m = np.nanmean(val_loss_arr, axis=0)
            s = np.nanstd(val_loss_arr, axis=0)
            plt.plot(epochs, m, color='C1', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Validation Loss (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Train balanced accuracy
        plt.subplot(3, 2, 3)
        if tr_bal_acc_arr is not None:
            m = np.nanmean(tr_bal_acc_arr, axis=0)
            s = np.nanstd(tr_bal_acc_arr, axis=0)
            plt.plot(epochs, m, color='C0', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Train Balanced Accuracy (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        
        # Val balanced accuracy
        plt.subplot(3, 2, 4)
        if val_bal_acc_arr is not None:
            m = np.nanmean(val_bal_acc_arr, axis=0)
            s = np.nanstd(val_bal_acc_arr, axis=0)
            plt.plot(epochs, m, color='C1', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Validation Balanced Accuracy (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        
        # Train MCC
        plt.subplot(3, 2, 5)
        if tr_mcc_arr is not None:
            m = np.nanmean(tr_mcc_arr, axis=0)
            s = np.nanstd(tr_mcc_arr, axis=0)
            plt.plot(epochs, m, color='C0', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C0', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Train MCC (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.legend()
        
        # Val MCC
        plt.subplot(3, 2, 6)
        if val_mcc_arr is not None:
            m = np.nanmean(val_mcc_arr, axis=0)
            s = np.nanstd(val_mcc_arr, axis=0)
            plt.plot(epochs, m, color='C1', label='Mean')
            plt.fill_between(epochs, m - s, m + s, color='C1', alpha=0.2, label='± std')
        plt.title(f'{title_prefix}Validation MCC (mean ± std)')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.legend()
        
        plt.tight_layout()
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.show()

# %% LOAD DATA & CONTENTS
# HE DATA & CONTENTS
HE_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_HE_train_logits_small_clam_sb_uni2-h'
HE_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_HE_val_logits_small_clam_sb_uni2-h'

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
KI67_train = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_KI67_train_logits_small_clam_sb_uni2-h'
KI67_val = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/EVAL_5_class_KI67_val_logits_small_clam_sb_uni2-h'

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

# %% PREPARE DATA
folds = [f'fold_{i}' for i in range(50)]

USE_NORMALIZATION = False  # Set to False to disable normalization

if USE_NORMALIZATION:
    print("Normalization with StandardScaler: mean=0, std=1 per modality")
else:
    print("Normalization not applied, raw logits used")

# Prepare the data
X_train_folds = {}
y_train_folds = {}
X_val_folds = {}
y_val_folds = {}
scalers_dict = {}

for fold in folds:
    if fold in HE_train_folds_dict and fold in KI67_train_folds_dict:
        # Extract raw logits
        HE_logits = HE_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_train_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        
        if USE_NORMALIZATION:
            scaler_HE = StandardScaler()
            scaler_KI67 = StandardScaler()
            
            HE_logits_norm = scaler_HE.fit_transform(HE_logits)
            KI67_logits_norm = scaler_KI67.fit_transform(KI67_logits)
            
            scalers_dict[fold] = {'HE': scaler_HE, 'KI67': scaler_KI67}
            
            merged_logits = np.concatenate((HE_logits_norm, KI67_logits_norm), axis=1)
        else:
            merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        
        labels = HE_train_folds_dict[fold]['Y'].values
        X_train_folds[fold] = merged_logits
        y_train_folds[fold] = labels
    
    if fold in HE_val_folds_dict and fold in KI67_val_folds_dict:
        HE_logits = HE_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        KI67_logits = KI67_val_folds_dict[fold][['logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']].values
        
        if USE_NORMALIZATION:
            HE_logits_norm = scalers_dict[fold]['HE'].transform(HE_logits)
            KI67_logits_norm = scalers_dict[fold]['KI67'].transform(KI67_logits)
            
            merged_logits = np.concatenate((HE_logits_norm, KI67_logits_norm), axis=1)
        else:
            merged_logits = np.concatenate((HE_logits, KI67_logits), axis=1)
        
        labels = HE_val_folds_dict[fold]['Y'].values
        X_val_folds[fold] = merged_logits
        y_val_folds[fold] = labels

# %% TRAIN SIMPLE MODEL
print("=" * 80)
print("Training Single Layer Model")
print("=" * 80)

all_metrics_single = {}
save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_uni2-h'

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)
        
        metrics = train_and_evaluate_model(
            model_type='single_layer',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_epochs=1500,
            learning_rate=0.001,
            weight_decay=0.0001,
            use_early_stopping=True,
            patience=200,
            use_l1_regularization=True,
            l1_lambda=0.001,
            use_class_weights=True,
            save_path=f'{save_dir}/{fold}.pth',
            fold_name=fold,
            verbose=True
        )
        
        all_metrics_single[fold] = metrics

# Plot individual fold curves
plot_training_curves(
    all_metrics_single,
    save_path=f'{save_dir}/plot.png',
    title_prefix=''
)

# Plot aggregate curves
plot_aggregate_curves(
    all_metrics_single,
    save_path=f'{save_dir}/plot_aggregate.png',
    title_prefix=''
)

# %% TRAIN ONE HIDDEN LAYER MODEL
print("=" * 80)
print("Training One Hidden Layer Model")
print("=" * 80)

all_metrics_one_hidden = {}
save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_uni2-h'
hidden_dim = 15

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)
        
        metrics = train_and_evaluate_model(
            model_type='one_hidden',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_epochs=1200,
            learning_rate=0.001,
            weight_decay=0.0001,
            use_early_stopping=True,
            patience=100,
            use_l1_regularization=True,
            l1_lambda=0.001,
            use_class_weights=True,
            save_path=f'{save_dir}/{fold}.pth',
            fold_name=fold,
            hidden_dim=hidden_dim,
            verbose=True
        )
        
        all_metrics_one_hidden[fold] = metrics

# Plot individual fold curves
plot_training_curves(
    all_metrics_one_hidden,
    save_path=f'{save_dir}/plot.png',
    title_prefix=''
)

# Plot aggregate curves
plot_aggregate_curves(
    all_metrics_one_hidden,
    save_path=f'{save_dir}/plot_aggregate.png',
    title_prefix=''
)

# %% TRAIN TWO HIDDEN LAYER MODEL
print("=" * 80)
print("Training Two Hidden Layer Model")
print("=" * 80)

all_metrics_two_hidden = {}
save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_uni2-h'
hidden_dim1 = 15
hidden_dim2 = 10

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)
        
        metrics = train_and_evaluate_model(
            model_type='two_hidden',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_epochs=1200,
            learning_rate=0.001,
            weight_decay=0.0001,
            use_early_stopping=True,
            patience=100,
            use_l1_regularization=True,
            l1_lambda=0.001,
            use_class_weights=True,
            save_path=f'{save_dir}/{fold}.pth',
            fold_name=fold,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            verbose=True
        )
        
        all_metrics_two_hidden[fold] = metrics

# Plot individual fold curves
plot_training_curves(
    all_metrics_two_hidden,
    save_path=f'{save_dir}/plot.png',
    title_prefix=''
)

# Plot aggregate curves
plot_aggregate_curves(
    all_metrics_two_hidden,
    save_path=f'{save_dir}/plot_aggregate.png',
    title_prefix=''
)

# %% TRAIN ATTENTION LAYER MODEL
print("=" * 80)
print("Training Attention Layer Model")
print("=" * 80)

all_metrics_attention = {}
save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/results/5_class/5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_uni2-h'

for fold in folds:
    if fold in X_train_folds and fold in X_val_folds:
        X_train = torch.tensor(X_train_folds[fold], dtype=torch.float32)
        y_train = torch.tensor(y_train_folds[fold], dtype=torch.long)
        X_val = torch.tensor(X_val_folds[fold], dtype=torch.float32)
        y_val = torch.tensor(y_val_folds[fold], dtype=torch.long)
        
        metrics = train_and_evaluate_model(
            model_type='attention',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_epochs=1500,
            learning_rate=0.001,
            weight_decay=0.0001,
            use_early_stopping=True,
            patience=30,
            use_l1_regularization=True,
            l1_lambda=0.001,
            use_class_weights=True,
            save_path=f'{save_dir}/{fold}.pth',
            fold_name=fold,
            verbose=True
        )
        
        all_metrics_attention[fold] = metrics

# Plot individual fold curves
plot_training_curves(
    all_metrics_attention,
    save_path=f'{save_dir}/plot.png',
    title_prefix=''
)

# Plot aggregate curves
plot_aggregate_curves(
    all_metrics_attention,
    save_path=f'{save_dir}/plot_aggregate.png',
    title_prefix=''
)

# %%
