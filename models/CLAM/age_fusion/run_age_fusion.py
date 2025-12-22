"""
Age Fusion Training Script
Loads pre-computed subject-level features (512D), fuses with age, and trains classifier
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, matthews_corrcoef, balanced_accuracy_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from age_fusion_utils import Classifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgeFusionDataset(Dataset):
    """Dataset that loads subject-level features and age"""
    
    def __init__(self, case_ids, labels, ages, feature_path):
        self.case_ids = case_ids
        self.labels = labels
        self.ages = ages
        self.feature_path = feature_path
        
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        label = self.labels[idx]
        age = self.ages[idx]
        
        # Load features
        feature_file = os.path.join(self.feature_path, f"{case_id}.pt")
        data = torch.load(feature_file, map_location='cpu', weights_only=False)
        
        # Extract feature vector
        if isinstance(data, dict):
            if 'subject_attention' in data:
                features = data['subject_attention']
            elif 'features' in data:
                features = data['features']
            else:
                features = list(data.values())[0]
        else:
            features = data
            
        # Ensure it's a tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()
            
        # Ensure shape is (512,)
        if features.dim() > 1:
            features = features.squeeze()
        
        # Age as tensor (1,)
        age_tensor = torch.tensor([age], dtype=torch.float32)
        
        # Label as tensor (scalar)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features, age_tensor, label_tensor


def load_data_with_age(csv_path, excel_path):
    """Load CSV and add age information (without normalization)"""
    df = pd.read_csv(csv_path)
    
    # Load age from Excel
    histological_df = pd.read_excel(excel_path, sheet_name='Histological Diagnoses', engine='openpyxl')
    
    # Map age to case_id
    df['age_at_diagnosis_days'] = df['case_id'].map(
        histological_df.set_index('External Id')['Age at Diagnosis (Days)'].to_dict()
    )
    
    # Convert to years (do NOT normalize yet - will do per-fold to avoid leakage)
    df['age_years'] = df['age_at_diagnosis_days'] / 365.25
    
    print(f"\nAge statistics (before normalization):")
    print(f"  Mean: {df['age_years'].mean():.2f} years, Std: {df['age_years'].std():.2f} years")
    print(f"  Range: [{df['age_years'].min():.2f}, {df['age_years'].max():.2f}]")
    
    # Remove rows with missing age
    df = df.dropna(subset=['age_years'])
    print(f"  Total samples with age: {len(df)}")
    
    return df


def load_split(split_file, df):
    """Load train/val/test split"""
    split_df = pd.read_csv(split_file)
    
    train_ids = split_df['train'].dropna().tolist()
    val_ids = split_df['val'].dropna().tolist()
    test_ids = split_df['test'].dropna().tolist()
    
    # Filter dataframe for each split
    train_df = df[df['case_id'].isin(train_ids)]
    val_df = df[df['case_id'].isin(val_ids)]
    test_df = df[df['case_id'].isin(test_ids)]
    
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, feature_path, batch_size, label_dict):
    """Create dataloaders for train/val/test with proper age normalization"""
    
    # Compute age normalization from TRAINING SET ONLY (avoid data leakage)
    age_mean = train_df['age_years'].mean()
    age_std = train_df['age_years'].std()
    
    # Normalize age for all splits using training statistics
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['age_normalized'] = (train_df['age_years'] - age_mean) / age_std
    val_df['age_normalized'] = (val_df['age_years'] - age_mean) / age_std
    test_df['age_normalized'] = (test_df['age_years'] - age_mean) / age_std
    
    print(f"  Age normalization (from training set): mean={age_mean:.2f}, std={age_std:.2f}")
    
    # Map string labels to integers using provided label_dict
    if train_df['label'].dtype == 'object':
        train_labels = train_df['label'].map(label_dict).tolist()
        val_labels = val_df['label'].map(label_dict).tolist()
        test_labels = test_df['label'].map(label_dict).tolist()
    else:
        train_labels = train_df['label'].tolist()
        val_labels = val_df['label'].tolist()
        test_labels = test_df['label'].tolist()
    
    train_dataset = AgeFusionDataset(
        train_df['case_id'].tolist(),
        train_labels,
        train_df['age_normalized'].tolist(),
        feature_path
    )
    
    val_dataset = AgeFusionDataset(
        val_df['case_id'].tolist(),
        val_labels,
        val_df['age_normalized'].tolist(),
        feature_path
    )
    
    test_dataset = AgeFusionDataset(
        test_df['case_id'].tolist(),
        test_labels,
        test_df['age_normalized'].tolist(),
        feature_path
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for features, age, labels in tqdm(loader, desc="Training", leave=False):
        features = features.to(device)
        age = age.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, probs, preds = model(features, age)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return avg_loss, accuracy, balanced_acc, mcc


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, age, labels in tqdm(loader, desc="Evaluating", leave=False):
            features = features.to(device)
            age = age.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, probs, preds = model(features, age)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Calculate AUC
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) == 2:  # Binary classification
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:  # Multi-class
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    return avg_loss, accuracy, balanced_acc, mcc, auc, all_preds, all_labels


def train_fold(fold_idx, train_df, val_df, test_df, cfg):
    """Train model for one fold"""
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, cfg.feature_path, cfg.batch_size, cfg.label_dict
    )
    
    # Initialize model
    use_age = cfg.get('use_age', True)  # Default to True for backwards compatibility
    model = Classifier(embed_dim=cfg.embed_dim, n_classes=cfg.n_classes, age_dim=cfg.age_dim, drop_out=cfg.drop_out, use_age=use_age).to(device)
    
    results = {
        'fold': fold_idx,
        'best_val_auc': None,
        'test_loss': None,
        'test_acc': None,
        'test_bal_acc': None,
        'test_mcc': None,
        'test_auc': None,
        'test_preds': None,
        'test_labels': None,
        'test_case_ids': None,
        'test_probs': None
    }
    
    # Create model subdirectory
    model_save_dir = os.path.join(cfg.results_dir, cfg.model_exp_code)
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f'fold_{fold_idx}_model.pt')
    
    # Training and Validation Phase
    if cfg.enable_training:
        print("\n=== TRAINING PHASE ===")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.reg)
        
        # Learning rate scheduler
        if cfg.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
        else:
            scheduler = None
        
        # Early stopping
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(cfg.max_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.max_epochs}")
            
            # Train
            train_loss, train_acc, train_bal_acc, train_mcc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc, val_bal_acc, val_mcc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Bal Acc: {train_bal_acc:.4f}, Train MCC: {train_mcc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}, Val MCC: {val_mcc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Learning rate scheduler step
            if scheduler is not None:
                scheduler.step()
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if cfg.early_stopping:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), model_path)
                    print(f"  *** New best Val AUC: {best_val_auc:.4f} - Model saved ***")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            else:
                best_val_auc = val_auc
        
        # Save final model if not using early stopping
        if not cfg.early_stopping:
            torch.save(model.state_dict(), model_path)
            print(f"\nTraining complete. Saved final model after {cfg.max_epochs} epochs (Final Val AUC: {best_val_auc:.4f})")
        else:
            print(f"\nTraining complete. Best Val AUC: {best_val_auc:.4f}")
        
        results['best_val_auc'] = best_val_auc
    else:
        print("\n=== TRAINING SKIPPED ===")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return results
    
    # Testing Phase
    if cfg.enable_testing:
        print("\n=== TESTING PHASE ===")
        
        # Load best model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from: {model_path}")
        else:
            print(f"WARNING: Model file not found: {model_path}")
            return results
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_bal_acc, test_mcc, test_auc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        
        # Get case IDs and probabilities
        test_case_ids = test_df['case_id'].tolist()
        
        # Get probabilities for all test samples
        model.eval()
        all_probs = []
        with torch.no_grad():
            for features, age, labels in test_loader:
                features = features.to(device)
                age = age.to(device)
                logits, probs, _ = model(features, age)
                all_probs.append(probs.cpu().numpy())
        
        test_probs = np.vstack(all_probs)
        
        # Create eval subdirectory
        eval_save_dir = os.path.join(cfg.eval_results_dir, cfg.eval_exp_code)
        os.makedirs(eval_save_dir, exist_ok=True)
        
        # Save detailed test results to eval_results_dir
        test_results_df = pd.DataFrame({
            'case_id': test_case_ids,
            'target': test_labels,
            'prediction': test_preds,
        })
        
        # Add probability columns for each class
        for class_idx in range(cfg.n_classes):
            test_results_df[f'prob_class_{class_idx}'] = test_probs[:, class_idx]
        
        test_results_path = os.path.join(eval_save_dir, f'fold_{fold_idx}_test_results.csv')
        test_results_df.to_csv(test_results_path, index=False)
        print(f"Saved test results to: {test_results_path}")
        
        results.update({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_bal_acc': test_bal_acc,
            'test_mcc': test_mcc,
            'test_auc': test_auc,
            'test_preds': test_preds,
            'test_labels': test_labels,
            'test_case_ids': test_case_ids,
            'test_probs': test_probs
        })
        
        print(f"\nTest Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Bal Acc: {test_bal_acc:.4f}, MCC: {test_mcc:.4f}, AUC: {test_auc:.4f}")
    else:
        print("\n=== TESTING SKIPPED ===")
    
    print(f"{'='*60}")
    
    return results


@hydra.main(version_base="1.3.2", 
            config_path='../../../configs/classification', 
            config_name='run_age_fusion')
def main(cfg: DictConfig):
    
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Print configuration
    print("\n" + "="*80)
    print("AGE FUSION TRAINING CONFIGURATION")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80 + "\n")
    
    # Create results directory
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.eval_results_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(cfg.results_dir, f'{cfg.model_exp_code}_config.yaml')
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Configuration saved to: {config_save_path}\n")
    
    # Load data with age
    print("Loading data with age information...")
    df = load_data_with_age(cfg.csv_path, cfg.excel_path)
    
    # Train all folds
    all_results = []
    
    for fold_idx in range(cfg.k_start, cfg.k_end):
        split_file = os.path.join(cfg.split_dir, f'splits_{fold_idx}.csv')
        
        if not os.path.exists(split_file):
            print(f"Split file not found: {split_file}")
            continue
        
        # Load split
        train_df, val_df, test_df = load_split(split_file, df)
        
        # Train fold
        results = train_fold(fold_idx, train_df, val_df, test_df, cfg)
        all_results.append(results)
    
    # Save summary - only include folds that completed testing
    valid_results = [r for r in all_results if r['test_auc'] is not None]
    
    if valid_results:
        results_df = pd.DataFrame([{
            'fold': r['fold'],
            'val_auc': r['best_val_auc'],
            'test_loss': r['test_loss'],
            'test_acc': r['test_acc'],
            'test_bal_acc': r['test_bal_acc'],
            'test_mcc': r['test_mcc'],
            'test_auc': r['test_auc']
        } for r in valid_results])
        
        # Create eval subdirectory for summary
        eval_save_dir = os.path.join(cfg.eval_results_dir, cfg.eval_exp_code)
        os.makedirs(eval_save_dir, exist_ok=True)
        
        summary_path = os.path.join(eval_save_dir, f'{cfg.eval_exp_code}.csv')
        results_df.to_csv(summary_path, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS - AGE FUSION MODEL")
        print(f"{'='*80}")
        print(f"Task: {cfg.task}")
        print(f"Feature Type: {cfg.feature_type}")
        print(f"Seed: {cfg.seed}")
        print(f"Folds: {cfg.k_start} to {cfg.k_end}")
        print(f"-"*80)
        print(f"Mean Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")
        print(f"Mean Test Acc: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")
        print(f"Mean Test Balanced Acc: {results_df['test_bal_acc'].mean():.4f} ± {results_df['test_bal_acc'].std():.4f}")
        print(f"Mean Test MCC: {results_df['test_mcc'].mean():.4f} ± {results_df['test_mcc'].std():.4f}")
        print(f"Mean Val AUC: {results_df['val_auc'].mean():.4f} ± {results_df['val_auc'].std():.4f}")
        print(f"-"*80)
        print(f"Models saved to: {cfg.results_dir}")
        print(f"Eval results saved to: {cfg.eval_results_dir}")
        print(f"Summary file: {summary_path}")
        print(f"{'='*80}")
    else:
        print("\n" + "="*80)
        print("No test results to summarize (testing was disabled)")
        print("="*80)


if __name__ == "__main__":
    main()
    print("\nFinished!")

