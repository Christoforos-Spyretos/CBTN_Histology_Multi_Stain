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
base_output_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/raw_features_plot/5_class'
splits_csv = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_5_class_0.7_0.1_0.2_100/splits_12.csv'
pca_output_dir = os.path.join(base_output_dir, 'PCA')
tsne_output_dir = os.path.join(base_output_dir, 'tSNE')

os.makedirs(pca_output_dir, exist_ok=True)
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
                
                if len(features.shape) == 2:
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

def plot_pca(features, labels, title, save_path, labels_to_include=None):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    if labels_to_include is not None:
        filtered_info = f" (Filtered: {', '.join(map(str, labels_to_include))})"
        title += filtered_info
    
    plt.figure(figsize=(10, 8))
    custom_palette = {
         "LGG": "#5C92B1",   # blue
         "HGG": "#D32F2F",   # red
         "MB": "#FF00FF",    # magenta (vivid pink)
         "EP": "#388E3C",    # green
         "GG": "#FF8000"     # bright orange
         }
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette=custom_palette, s=50)
    # plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    # plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_tsne(features, labels, title, save_path, labels_to_include=None):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Add filtered labels info to title if applicable
    if labels_to_include is not None:
        filtered_info = f" (Filtered: {', '.join(map(str, labels_to_include))})"
        title += filtered_info

    plt.figure(figsize=(10, 8))
    custom_palette = {
         "LGG": "#5C92B1",   # blue
         "HGG": "#D32F2F",   # red
         "MB": "#FF00FF",    # magenta (vivid pink)
         "EP": "#388E3C",    # green
         "GG": "#FF8000"     # bright orange
         }
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=custom_palette, s=50)
    # plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    # plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# %% PCA and t-SNE PLOTS FOR HE FEATURES
# Load splits
split_case_ids = load_split_case_ids(splits_csv)

# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv')
# path to features
path_to_HE_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1_5/pt_files'

labels_to_include = ['LGG', 'HGG', 'EP', 'MB', 'GG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for H&E features ===")
    
    features_matrix, labels, case_ids = collect_features_and_labels(
        df, path_to_HE_features, labels_to_include, split_case_ids[split_name]
    )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    plot_pca(features_matrix, labels, f'PCA of H&E Features ({split_name})', 
             os.path.join(pca_output_dir, f'PCA_raw_HE_features_5_class_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, labels, f't-SNE of H&E Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_raw_HE_features_5_class_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR KI67 FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv')
# path to features
path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1_5/pt_files'

labels_to_include = ['LGG', 'HGG', 'EP', 'MB', 'GG']

# Plot for each split (train and test)
for split_name in ['train', 'test']:
    print(f"\n=== Processing {split_name.upper()} split for Ki-67 features ===")
    
    features_matrix, labels, case_ids = collect_features_and_labels(
        df, path_to_KI67_features, labels_to_include, split_case_ids[split_name]
    )
    
    print(f"Feature matrix shape: {features_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    plot_pca(features_matrix, labels, f'PCA of Ki-67 Features ({split_name})', 
             os.path.join(pca_output_dir, f'PCA_raw_KI67_features_5_class_{split_name}.png'), labels_to_include)
    plot_tsne(features_matrix, labels, f't-SNE of Ki-67 Features ({split_name})', 
              os.path.join(tsne_output_dir, f'tSNE_raw_KI67_features_5_class_{split_name}.png'), labels_to_include)

# %% PCA and t-SNE PLOTS FOR MERGED HE AND KI67 FEATURES
# # csv file
# df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv')
# # path to features
# path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE_KI67/features/conch_v1_5/pt_files'

# labels = ['LGG', 'HGG', 'EP', 'MB', 'GG']

# features_matrix, labels, case_ids = collect_features_and_labels(df, path_to_KI67_features, labels)

# print(f"Feature matrix shape: {features_matrix.shape}")
# print(f"Labels shape: {labels.shape}")
# print(f"Unique labels: {np.unique(labels)}")

# plot_pca(features_matrix, labels, 'PCA of Merged H&E & Ki-67 Features', os.path.join(pca_output_dir, 'PCA_raw_Merged_HE_KI67_features_5_class.png'))
# plot_tsne(features_matrix, labels, 't-SNE of Merged H&E & Ki-67 Features', os.path.join(tsne_output_dir, 'tSNE_raw_Merged_HE_KI67_features_5_class.png'))

# %% t-SNE PLOTS FOR HE FEATURES WITH LEGEND
# def plot_tsne_with_legend(features, labels, title, save_path, labels_to_include=None):
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
#     tsne_result = tsne.fit_transform(features)
#     fig, ax = plt.subplots(figsize=(10, 8))
#     custom_palette = {
#         "LGG": "#5C92B1", "HGG": "#D32F2F", "MB": "#8e24aa", "EP": "#388E3C", "GG": "#A15D73"
#     }
#     scatter = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=True, ax=ax)
#     ax.set_xlabel('t-SNE Component 1', fontsize=14)
#     ax.set_ylabel('t-SNE Component 2', fontsize=14)
#     legend_labels = ["LGG", "HGG", "MB", "EP", "GG"]
#     handles = [mpatches.Patch(facecolor=custom_palette[label], edgecolor='black', linewidth=1, label=label) for label in legend_labels]
#     ax.legend_.remove() if hasattr(ax, 'legend_') and ax.legend_ else None
#     leg = fig.legend(handles, legend_labels, title=None, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=14, frameon=True)
#     leg.get_frame().set_edgecolor('black')
#     leg.get_frame().set_linewidth(1)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     plt.tight_layout(rect=[0,0.08,1,1])
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.show()

# plot_tsne_with_legend(features_matrix, labels, 't-SNE of Learned 5-Class Features (with legend)', os.path.join(tsne_output_dir, 'tSNE_learned_5class_features_with_legend.png'))

# %%
def plot_legend_box(save_path):
    custom_palette = {
        "LGG": "#5C92B1", "HGG": "#D32F2F", "Medulloblastoma": "#FF00FF", "Ependymoma": "#388E3C", "Ganglioglioma": "#FF8000"
    }
    legend_labels = ["LGG", "HGG", "Medulloblastoma", "Ependymoma", "Ganglioglioma"]
    handles = [mpatches.Patch(facecolor=custom_palette[label], edgecolor='black', linewidth=1, label=label) for label in legend_labels]
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.axis('off')
    leg = fig.legend(handles, legend_labels, title=None, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=5, fontsize=14, frameon=True)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi = 300)
    plt.close(fig)

plot_legend_box(os.path.join(base_output_dir, 'legend_box_5class.png'))
# %%