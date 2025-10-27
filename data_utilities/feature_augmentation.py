# %% IMPORTS
import os
import numpy as np
import random
import torch
import pandas as pd

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

def add_gaussian_noise(features, noise_level=0.01, augment_ratio=1.0):
    """
    Add Gaussian noise to feature vectors.
    
    Args:
        features: torch.Tensor of shape (n_patches, feature_dim)
        noise_level: Standard deviation of the Gaussian noise
        augment_ratio: Proportion of elements to augment (0.0 to 1.0)
                      1.0 = all elements, 0.5 = 50% of elements, etc.
    
    Returns:
        Augmented features with added noise
    """
    if augment_ratio >= 1.0:
        # Augment all elements
        noise = torch.randn_like(features) * noise_level
        augmented_features = features + noise
    else:
        # Augment only a random subset of elements
        augmented_features = features.clone()
        
        # Create a random mask indicating which elements to augment
        mask = torch.rand_like(features) < augment_ratio
        
        # Generate noise only for selected elements
        noise = torch.randn_like(features) * noise_level
        
        # Apply noise only where mask is True
        augmented_features[mask] += noise[mask]
    
    return augmented_features

def load_split_case_ids(split_file, split='train'):
    """
    Load case IDs from a split CSV file for a specific split (train/val/test).
    
    Args:
        split_file: Path to the CSV file
        split: 'train', 'val', or 'test'
    
    Returns:
        List of case IDs in the specified split
    """
    df = pd.read_csv(split_file)
    if split in df.columns:
        case_ids = df[split].dropna().tolist()
        return case_ids
    else:
        raise ValueError(f"Split '{split}' not found in {split_file}")

def augment_features_for_split(
    original_features_path,
    split_file,
    save_path,
    task_name,
    feature_extractor,
    augmentation_type,
    fold_number,
    noise_level=0.01,
    augment_ratio=1.0,
    case_ratio=1.0
):
    """
    Augment features for training cases in a specific fold.
    
    Args:
        original_features_path: Path to original .pt files
        split_file: Path to the splits CSV file
        save_path: Base path to save augmented features
        task_name: 'LGG_vs_HGG' or '5_class'
        feature_extractor: Name of the feature extractor (e.g., 'conch_v1')
        augmentation_type: Type of augmentation (e.g., 'gaussian_noise')
        fold_number: Fold number
        noise_level: Standard deviation of Gaussian noise
        augment_ratio: Proportion of elements to augment (0.0 to 1.0)
        case_ratio: Proportion of training cases to augment (0.0 to 1.0)
                    1.0 = all cases, 0.5 = 50% of cases, etc.
    """
    # Load train case IDs
    train_case_ids = load_split_case_ids(split_file, split='train')
    
    # Select random subset of cases to augment
    if case_ratio < 1.0:
        n_cases_to_augment = int(len(train_case_ids) * case_ratio)
        train_case_ids = random.sample(train_case_ids, n_cases_to_augment)
    
    print(f"\nProcessing fold {fold_number} for task {task_name} with {augmentation_type}")
    print(f"Number of cases to augment: {len(train_case_ids)} ({case_ratio*100:.1f}% of training set)")
    print(f"Augmenting {augment_ratio*100:.1f}% of elements with noise_level={noise_level}")
    
    # Create output directory
    output_dir = os.path.join(save_path, task_name, feature_extractor, augmentation_type, f'fold_{fold_number}', 'pt_files')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each training case
    augmented_count = 0
    for case_id in train_case_ids:
        feature_file = os.path.join(original_features_path, f'{case_id}.pt')
        
        if not os.path.exists(feature_file):
            print(f"Warning: Feature file not found for {case_id}")
            continue
        
        # Load original features
        features = torch.load(feature_file, weights_only=True)
        
        # Add noise to create one augmented version
        augmented_features = add_gaussian_noise(features, noise_level=noise_level, augment_ratio=augment_ratio)
        
        # Save augmented features
        output_file = os.path.join(output_dir, f'{case_id}.pt')
        torch.save(augmented_features, output_file)
        augmented_count += 1
    
    print(f"Created {augmented_count} augmented feature files in {output_dir}")
    return augmented_count

def augment_all_folds(
    original_features_path,
    splits_path,
    save_path,
    task_name,
    feature_extractor,
    augmentation_type='gaussian_noise',
    n_folds=50,
    noise_level=0.01,
    augment_ratio=1.0,
    case_ratio=1.0
):
    """
    Augment features for all folds in a task.
    
    Args:
        original_features_path: Path to original .pt files
        splits_path: Path to directory containing split CSV files
        save_path: Base path to save augmented features
        task_name: 'LGG_vs_HGG' or '5_class'
        feature_extractor: Name of the feature extractor
        augmentation_type: Type of augmentation (e.g., 'gaussian_noise', 'dropout', etc.)
        n_folds: Number of folds to process
        noise_level: Standard deviation of Gaussian noise
        augment_ratio: Proportion of elements to augment (0.0 to 1.0)
                      1.0 = all elements, 0.5 = 50% of elements, etc.
        case_ratio: Proportion of training cases to augment (0.0 to 1.0)
                    1.0 = all cases, 0.5 = 50% of cases, etc.
    """
    total_augmented = 0
    
    for fold in range(n_folds):
        split_file = os.path.join(splits_path, f'splits_{fold}.csv')
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}")
            continue
        
        count = augment_features_for_split(
            original_features_path=original_features_path,
            split_file=split_file,
            save_path=save_path,
            task_name=task_name,
            feature_extractor=feature_extractor,
            augmentation_type=augmentation_type,
            fold_number=fold,
            noise_level=noise_level,
            augment_ratio=augment_ratio,
            case_ratio=case_ratio
        )
        total_augmented += count
    
    print(f"\n{'='*60}")
    print(f"Total augmented files created for {task_name} using {augmentation_type}: {total_augmented}")
    print(f"{'='*60}\n")

# %% LOAD FEATURES & DEFINE PATHS
path_to_HE_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1/pt_files'
path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1/pt_files'

path_to_LGG_vs_HGG_splits = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_LGG_vs_HGG_0.7_0.1_0.2_100'
path_to_5_class_splits = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_5_class_0.7_0.1_0.2_100'

path_to_save_HE_augmented_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/augmented_features'
path_to_save_KI67_augmented_features = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/augmented_features'

feature_extractor = 'conch_v1'

# %% RUN AUGMENTATION for LGG vs HGG task
# Augment HE features for LGG vs HGG task
augment_all_folds(
    original_features_path=path_to_HE_features,
    splits_path=path_to_LGG_vs_HGG_splits,
    save_path=path_to_save_HE_augmented_features,
    task_name='LGG_vs_HGG',
    feature_extractor=feature_extractor,
    augmentation_type='gaussian_noise',
    n_folds=50,
    noise_level=0.05, # noise level for Gaussian noise
    augment_ratio=0.3,  # Augment 30% of elements in each feature vector
    case_ratio=0.5      # Augment 50% of training cases
)

# Augment KI67 features for LGG vs HGG task
augment_all_folds(
    original_features_path=path_to_KI67_features,
    splits_path=path_to_LGG_vs_HGG_splits,
    save_path=path_to_save_HE_augmented_features,
    task_name='LGG_vs_HGG',
    feature_extractor=feature_extractor,
    augmentation_type='gaussian_noise',
    n_folds=50,
    noise_level=0.05, # noise level for Gaussian noise
    augment_ratio=0.3,  # Augment 30% of elements in each feature vector
    case_ratio=0.5      # Augment 50% of training cases
)

# %% RUN AUGMENTATION for 5-class task
# Augment HE features for 5-class task
augment_all_folds(
    original_features_path=path_to_HE_features,
    splits_path=path_to_5_class_splits,
    save_path=path_to_save_HE_augmented_features,
    task_name='5_class',
    feature_extractor=feature_extractor,
    augmentation_type='gaussian_noise',
    n_folds=50,
    noise_level=0.05, # noise level for Gaussian noise
    augment_ratio=0.3,  # Augment 30% of elements in each feature vector
    case_ratio=0.5      # Augment 50% of training cases
)

# Augment KI67 features for 5-class task
augment_all_folds(
    original_features_path=path_to_KI67_features,
    splits_path=path_to_5_class_splits,
    save_path=path_to_save_HE_augmented_features,
    task_name='5_class',
    feature_extractor=feature_extractor,
    augmentation_type='gaussian_noise',
    n_folds=50,
    noise_level=0.05, # noise level for Gaussian noise
    augment_ratio=0.3,  # Augment 30% of elements in each feature vector
    case_ratio=0.5      # Augment 50% of training cases
)

# %%
