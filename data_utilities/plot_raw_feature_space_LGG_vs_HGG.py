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
base_output_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/raw_features_plot/LGG_vs_HGG'
pca_output_dir = os.path.join(base_output_dir, 'PCA')
tsne_output_dir = os.path.join(base_output_dir, 'tSNE')

os.makedirs(pca_output_dir, exist_ok=True)
os.makedirs(tsne_output_dir, exist_ok=True)

# %% UTILITY FUNCTIONS 
def collect_features_and_labels(df, path_to_features, labels_to_include=None):
    
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
    
    for idx, row in df_filtered.iterrows():
        case_id = row['case_id']
        label = row['label']
        
        pt_file_path = os.path.join(path_to_features, f"{case_id}.pt")
        
        if os.path.exists(pt_file_path):
            try:
                features = torch.load(pt_file_path)
                
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
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='tab10', s=50)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_tsne(features, labels, title, save_path, labels_to_include=None):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Add filtered labels info to title if applicable
    if labels_to_include is not None:
        filtered_info = f" (Filtered: {', '.join(map(str, labels_to_include))})"
        title += filtered_info

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='tab10', s=50)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# %% PCA and t-SNE PLOTS FOR HE FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')
# path to features
path_to_HE_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1/pt_files'

labels = ['LGG', 'HGG']

features_matrix, labels, case_ids = collect_features_and_labels(df, path_to_HE_features, labels)

print(f"Feature matrix shape: {features_matrix.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

plot_pca(features_matrix, labels, 'PCA of HE Features', os.path.join(pca_output_dir, 'PCA_raw_HE_features_LGG_vs_HGG.png'))
plot_tsne(features_matrix, labels, 't-SNE of HE Features', os.path.join(tsne_output_dir, 'tSNE_raw_HE_features_LGG_vs_HGG.png'))

# %% PCA and t-SNE PLOTS FOR KI67 FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')
# path to features
path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1/pt_files'

labels = ['LGG', 'HGG']

features_matrix, labels, case_ids = collect_features_and_labels(df, path_to_KI67_features, labels)

print(f"Feature matrix shape: {features_matrix.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

plot_pca(features_matrix, labels, 'PCA of KI67 Features', os.path.join(pca_output_dir, 'PCA_raw_KI67_features_LGG_vs_HGG.png'))
plot_tsne(features_matrix, labels, 't-SNE of KI67 Features', os.path.join(tsne_output_dir, 'tSNE_raw_KI67_features_LGG_vs_HGG.png'))

# %% PCA and t-SNE PLOTS FOR MERGED HE AND KI67 FEATURES
# csv file
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')
# path to features
path_to_KI67_features = '/local/data3/chrsp39/CBTN_v2/Merged_HE_KI67/features/conch_v1/pt_files'

labels = ['LGG', 'HGG']

features_matrix, labels, case_ids = collect_features_and_labels(df, path_to_KI67_features, labels)

print(f"Feature matrix shape: {features_matrix.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

plot_pca(features_matrix, labels, 'PCA of Merged HE & KI67 Features', os.path.join(pca_output_dir, 'PCA_raw_Merged_HE_KI67_features_LGG_vs_HGG.png'))
plot_tsne(features_matrix, labels, 't-SNE of Merged HE & KI67 Features', os.path.join(tsne_output_dir, 'tSNE_raw_Merged_HE_KI67_features_LGG_vs_HGG.png'))

# %%