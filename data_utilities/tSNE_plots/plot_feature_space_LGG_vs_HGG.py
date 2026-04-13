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
import matplotlib.patches as mpatches

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

def collect_features_and_labels(df, path_to_features, labels_to_include=None, case_ids_to_include=None):
    
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
                features = torch.load(pt_file_path, weights_only=False)
                
                if isinstance(features, dict):
                    features = list(features.values())[0]
                    if hasattr(features, 'squeeze'):
                        features = features.squeeze()
                    aggregated_features = torch.tensor(features) if not isinstance(features, torch.Tensor) else features
                    if len(aggregated_features.shape) == 2:
                        aggregated_features = torch.mean(aggregated_features, dim=0)
                elif len(features.shape) == 2:
                    aggregated_features = torch.mean(features, dim=0)
                else:
                    aggregated_features = features
                
                all_features.append(aggregated_features.numpy())
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
    
#     if labels_to_include is not None:
#         filtered_info = f" (Filtered: {', '.join(map(str, labels_to_include))})"
#         title += filtered_info
    
#     plt.figure(figsize=(10, 8))
#     # Define custom palette: LGG (blue), HGG (red)
#     custom_palette = {"LGG": "#5C92B1", "HGG": "#D32F2F"}
#     sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=False)
#     # plt.title(title, fontsize=18, fontweight='bold')
#     plt.xlabel('PCA Component 1', fontsize=14)
#     plt.ylabel('PCA Component 2', fontsize=14)
#     # Legend removed from main plot
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

    # Add filtered labels info to title if applicable
    if labels_to_include is not None:
        filtered_info = f" (Filtered: {', '.join(map(str, labels_to_include))})"
        title += filtered_info

    plt.figure(figsize=(10, 8))
    # Define custom palette: LGG (blue), HGG (red)
    custom_palette = {"LGG": "#5C92B1", "HGG": "#D32F2F"}
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=False)
    # plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    # Legend removed from main plot
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# %% PCA and t-SNE PLOTS FOR H&E FEATURES
# Load splits
split_case_ids = load_split_case_ids(splits_csv)

# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')
# path to features
path_to_HE_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1_5/pt_files'

labels_to_include = ['LGG', 'HGG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for H&E features ===")
    
    features_matrix, labels, case_ids = collect_features_and_labels(
        df, path_to_HE_features, labels_to_include, split_case_ids[split_name]
    )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # plot_pca(features_matrix, labels, f'PCA of H&E Features ({split_name})', 
    #          os.path.join(pca_output_dir, f'PCA_raw_HE_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, labels, f't-SNE of H&E Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_raw_HE_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR Ki-67 FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')
# path to features
path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1_5/pt_files'

labels_to_include = ['LGG', 'HGG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for Ki-67 features ===")
    
    features_matrix, labels, case_ids = collect_features_and_labels(
        df, path_to_KI67_features, labels_to_include, split_case_ids[split_name]
        )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # plot_pca(features_matrix, labels, f'PCA of Ki-67 Features ({split_name})', 
    #          os.path.join(pca_output_dir, f'PCA_raw_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, labels, f't-SNE of Ki-67 Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_raw_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% LOAD SPLITS
split_case_ids = load_split_case_ids(splits_csv)

# %% PCA and t-SNE PLOTS FOR LEARNED H&E FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')

labels_to_include = ['LGG', 'HGG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for learned H&E features ===")
    
    path_to_learned_HE_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_HE/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/pt_files/fold_9'
    features_matrix, label_array, case_ids = collect_features_and_labels(
        df, path_to_learned_HE_features, labels_to_include, split_case_ids[split_name]
    )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")
    
    # plot_pca(features_matrix, label_array, f'PCA of Learned H&E Features ({split_name})', 
    #          os.path.join(pca_output_dir, f'PCA_learned_HE_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, label_array, f't-SNE of Learned H&E Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_learned_HE_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR LEARNED Ki-67 FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')

labels_to_include = ['LGG', 'HGG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for learned Ki-67 features ===")
    
    path_to_learned_KI67_features = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_KI67/70%_split/LGG_vs_HGG/features/conch_v1_5/{split_name}/pt_files/fold_9'
    features_matrix, label_array, case_ids = collect_features_and_labels(
        df, path_to_learned_KI67_features, labels_to_include, split_case_ids[split_name]
    )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {label_array.shape}")
    print(f"Unique labels: {np.unique(label_array)}")
    
    # plot_pca(features_matrix, label_array, f'PCA of Learned Ki-67 Features ({split_name})', 
    #          os.path.join(pca_output_dir, f'PCA_learned_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, label_array, f't-SNE of Learned Ki-67 Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_learned_KI67_features_LGG_vs_HGG_{split_name}.png'), labels_to_include)

# %% LEGEND BOX
def plot_legend_box(save_path):
    custom_palette = {"LGG": "#5C92B1", "HGG": "#D32F2F"}
    legend_labels = ["LGG", "HGG"]
    handles = [mpatches.Patch(facecolor=custom_palette[label], edgecolor='black', linewidth=1, label=label) for label in legend_labels]
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.axis('off')
    leg = fig.legend(handles, legend_labels, title=None, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=2, fontsize=14, frameon=True)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close(fig)

plot_legend_box(os.path.join(base_output_dir, 'legend_box_LGG_vs_HGG.png'))

# %%