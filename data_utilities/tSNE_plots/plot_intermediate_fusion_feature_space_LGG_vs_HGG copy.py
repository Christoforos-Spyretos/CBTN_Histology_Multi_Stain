# %% IMPORTS
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.image as mpimg

# %% CREATE OUTPUT DIRECTORIES
base_output_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG'
splits_csv = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_LGG_vs_HGG_0.7_0.1_0.2_100/splits_9.csv'
# pca_output_dir = os.path.join(base_output_dir, 'PCA')
tsne_output_dir = os.path.join(base_output_dir, 'tSNE')

# os.makedirs(pca_output_dir, exist_ok=True)
os.makedirs(tsne_output_dir, exist_ok=True)

# %% UTILITY FUNCTIONS 
def load_split_case_ids(splits_csv_path):
    """Load case IDs for each split from the splits CSV."""
    df_splits = pd.read_csv(splits_csv_path)
    
    train_ids = df_splits['train'].dropna().tolist()
    val_ids = df_splits['val'].dropna().tolist()
    test_ids = df_splits['test'].dropna().tolist()
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

def collect_learned_features_and_labels(df, path_to_features, labels_to_include=None, case_ids_to_include=None):

    """
    Collect learned (subject-level attention) features that are stored as dictionaries.
    These features have shape (512, 1) and are already aggregated at the subject level.
    """

    all_features = []
    all_labels = []
    all_case_ids = []
    
    if labels_to_include is not None:
        df_filtered = df[df['label'].isin(labels_to_include)]
        print(f"Filtering to include only labels: {labels_to_include}")
        print(f"Filtered dataset size: {len(df_filtered)} cases (from {len(df)} total)")
    else:
        df_filtered = df
        print(f"Processing all {len(df)} cases...")
    
    if case_ids_to_include is not None:
        df_filtered = df_filtered[df_filtered['case_id'].isin(case_ids_to_include)]
        print(f"Filtering to include only {len(case_ids_to_include)} case IDs from split")
        print(f"Filtered dataset size after case ID filter: {len(df_filtered)} cases")
    
    for idx, row in df_filtered.iterrows():
        case_id = row['case_id']
        label = row['label']
        
        pt_file_path = os.path.join(path_to_features, f"{case_id}.pt")
        
        if os.path.exists(pt_file_path):
            try:
                data = torch.load(pt_file_path, weights_only=False)
                
                # handle dictionary format (any key: subject_attention, concatenated, element_wise_mult, cross_attended, etc.)
                if isinstance(data, dict):
                    features = list(data.values())[0]
                    if hasattr(features, 'squeeze'):
                        features = features.squeeze()
                else:
                    # fallback for tensor formats (both 1D and 2D)
                    features = data
                    # if it's a 2D tensor with multiple patches, aggregate via mean
                    if hasattr(features, 'shape') and len(features.shape) == 2 and features.shape[0] > 1:
                        features = torch.mean(features, dim=0)
                    elif hasattr(features, 'squeeze'):
                        # for already aggregated features, just squeeze extra dimensions
                        features = features.squeeze()
                
                # convert to numpy if it's a torch tensor
                if hasattr(features, 'numpy'):
                    features = features.numpy()
                
                all_features.append(features)
                all_labels.append(label)
                all_case_ids.append(case_id)
                
                if (len(all_features)) % 50 == 0:
                    print(f"Processed {len(all_features)} cases...")
                    
            except Exception as e:
                print(f"Error loading features for case {case_id}: {e}")
        else:
            print(f"Features file not found for case {case_id}")
    
    print(f"Successfully loaded features for {len(all_features)} cases")
    return np.array(all_features), np.array(all_labels), np.array(all_case_ids)

# def plot_pca(features, labels, title, save_path, labels_to_include=None):
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(features)
#     plt.figure(figsize=(10, 8))
#     custom_palette = {"LGG": "#5C92B1", "HGG": "#D32F2F"}
#     sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=False)
#     # plt.title(title, fontsize=18, fontweight='bold')
#     plt.xlabel('PCA Component 1', fontsize=14)
#     plt.ylabel('PCA Component 2', fontsize=14)
#     plt.tick_params(axis='both', which='major', labelsize=12)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()

def plot_tsne(features, labels, title, save_path, labels_to_include=None):
    if len(features) < 2:
        print(f"Skipping t-SNE for '{title}': not enough samples ({len(features)})")
        return
    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    custom_palette = {"LGG": "#5C92B1", "HGG": "#D32F2F"}
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=False)
    # plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# %% PCA and t-SNE PLOTS FOR INTERMEDIATE CONCATENATED LEARNED H&E + Ki-67 FEATURES
# Load splits
split_case_ids = load_split_case_ids(splits_csv)

# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')

labels_to_include = ['LGG', 'HGG']

for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for Concatenation features ===")

    path_to_concat_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CONCATENATION_HE_KI67/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/fold_9/pt_files'

    features_matrix, label_array, case_ids = collect_learned_features_and_labels(
        df, path_to_concat_features, labels_to_include, split_case_ids[split_name]
    )

    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")

    plot_tsne(features_matrix, label_array, f't-SNE of Concatenation H&E & Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_Concatenation_HE_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR INTERMEDIATE ELEMENT-WISE MULTIPLICATION LEARNED H&E + Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for Element-wise Multiplication features ===")

    path_to_elemwise_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ELEMENT_WISE_MULT_HE_KI67/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/fold_9/pt_files'

    features_matrix, label_array, case_ids = collect_learned_features_and_labels(
        df, path_to_elemwise_features, labels_to_include, split_case_ids[split_name]
    )

    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")

    plot_tsne(features_matrix, label_array, f't-SNE of Element-wise Multiplication H&E & Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_ElemWiseMult_HE_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR INTERMEDIATE H&E-GUIDED CROSS ATTENTION LEARNED H&E + Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for H&E-guided Cross Attention features ===")

    path_to_he_cross_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_HE_inform_KI67/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/fold_9/pt_files'

    features_matrix, label_array, case_ids = collect_learned_features_and_labels(
        df, path_to_he_cross_features, labels_to_include, split_case_ids[split_name]
    )

    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")

    plot_tsne(features_matrix, label_array, f't-SNE of H&E-guided Cross Attention Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_CrossAttn_HE_inform_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR INTERMEDIATE Ki-67-GUIDED CROSS ATTENTION LEARNED H&E + Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for Ki-67-guided Cross Attention features ===")

    path_to_ki67_cross_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_KI67_inform_HE/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/fold_9/pt_files'

    features_matrix, label_array, case_ids = collect_learned_features_and_labels(
        df, path_to_ki67_cross_features, labels_to_include, split_case_ids[split_name]
    )

    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")

    plot_tsne(features_matrix, label_array, f't-SNE of Ki-67-guided Cross Attention Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_CrossAttn_KI67_inform_HE_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %%