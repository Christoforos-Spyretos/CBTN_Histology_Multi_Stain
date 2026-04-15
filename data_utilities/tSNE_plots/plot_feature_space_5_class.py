# %% IMPORTS
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patches as mpatches

# %% SETUP
base_output_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class'
splits_csv = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_5_class_0.7_0.1_0.2_100/splits_12.csv'
tsne_output_dir = os.path.join(base_output_dir, 'tSNE')
os.makedirs(tsne_output_dir, exist_ok=True)

dataset_csv = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv'
labels_to_include = ['LGG', 'HGG', 'EP', 'MB', 'GG']

# %% UTILITY FUNCTIONS
def load_split_case_ids(splits_csv_path):
    df_splits = pd.read_csv(splits_csv_path)
    return {
        'train': df_splits['train'].dropna().tolist(),
        'val':   df_splits['val'].dropna().tolist(),
        'test':  df_splits['test'].dropna().tolist(),
    }

def collect_features_and_labels(df, path_to_features, labels_to_include=None, case_ids_to_include=None):
    all_features, all_labels, all_case_ids = [], [], []

    df_filtered = df[df['label'].isin(labels_to_include)] if labels_to_include else df.copy()
    if case_ids_to_include is not None:
        df_filtered = df_filtered[df_filtered['case_id'].isin(case_ids_to_include)]
    print(f"Filtered dataset size: {len(df_filtered)} cases")

    for _, row in df_filtered.iterrows():
        case_id, label = row['case_id'], row['label']
        pt_file_path = os.path.join(path_to_features, f"{case_id}.pt")

        if not os.path.exists(pt_file_path):
            print(f"Features file not found for case {case_id}")
            continue

        try:
            data = torch.load(pt_file_path, weights_only=False)

            if isinstance(data, dict):
                features = list(data.values())[0]
                if hasattr(features, 'squeeze'):
                    features = features.squeeze()
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features)
            else:
                features = data

            if hasattr(features, 'shape') and len(features.shape) == 2 and features.shape[0] > 1:
                features = torch.mean(features, dim=0)
            elif hasattr(features, 'squeeze'):
                features = features.squeeze()

            if hasattr(features, 'numpy'):
                features = features.numpy()

            all_features.append(features)
            all_labels.append(label)
            all_case_ids.append(case_id)

            if len(all_features) % 50 == 0:
                print(f"Processed {len(all_features)} cases...")

        except Exception as e:
            print(f"Error loading features for case {case_id}: {e}")

    print(f"Successfully loaded features for {len(all_features)} cases")
    return np.array(all_features), np.array(all_labels), np.array(all_case_ids)

def plot_tsne(features, labels, title, save_path):
    if len(features) < 2:
        print(f"Skipping t-SNE for '{title}': not enough samples ({len(features)})")
        return
    perplexity = min(30, len(features) - 1)
    tsne_result = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42).fit_transform(features)
    plt.figure(figsize=(10, 8))
    custom_palette = {
        "LGG": "#5C92B1",   # blue
        "HGG": "#D32F2F",   # red
        "MB": "#FF00FF",    # magenta (vivid pink)
        "EP": "#388E3C",    # green
        "GG": "#FF8000"     # bright orange
    }
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=custom_palette, s=50, legend=False)
    plt.xlabel('t-SNE Component 1', fontsize=40)
    plt.ylabel('t-SNE Component 2', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# %% LOAD SPLITS AND DATASET
split_case_ids = load_split_case_ids(splits_csv)
df = pd.read_csv(dataset_csv)

# %% t-SNE FOR RAW H&E FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Raw H&E features — {split_name.upper()} ===")
    features_matrix, labels, _ = collect_features_and_labels(
        df, '/local/data3/chrsp39/CBTN_v2/Merged_HE/features/conch_v1_5/pt_files',
        labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of H&E Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_raw_HE_features_5_class_{split_name}.png'))

# %% t-SNE FOR RAW Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Raw Ki-67 features — {split_name.upper()} ===")
    features_matrix, labels, _ = collect_features_and_labels(
        df, '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1_5/pt_files',
        labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_raw_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR LEARNED H&E FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Learned H&E features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_HE/70%_split/5_class/features/conch_v1_5/{split_name}/pt_files/fold_12'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Learned H&E Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_HE_features_5_class_{split_name}.png'))

# %% t-SNE FOR LEARNED Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Learned Ki-67 features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_KI67/70%_split/5_class/features/conch_v1_5/{split_name}/pt_files/fold_12'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Learned Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR RAW EARLY FUSION H&E + Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Raw Early Fusion H&E + Ki-67 features — {split_name.upper()} ===")
    features_matrix, labels, _ = collect_features_and_labels(
        df, '/local/data3/chrsp39/CBTN_v2/Merged_HE_KI67/features/conch_v1_5/pt_files',
        labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Early Fusion H&E + Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_raw_Early_Fusion_HE_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR LEARNED EARLY FUSION H&E + Ki-67 FEATURES
for split_name in ['train', 'test']:
    print(f"\n=== Learned Early Fusion H&E + Ki-67 features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_Early_Fusion_HE_KI67/70%_split/5_class/features/conch_v1_5/{split_name}/pt_files/fold_12'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Learned Early Fusion H&E + Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_Early_Fusion_HE_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR INTERMEDIATE FUSION — CONCATENATION
for split_name in ['train', 'test']:
    print(f"\n=== Concatenation features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CONCATENATION_HE_KI67/70%_split/5_class/features/conch_v1_5/{split_name}/fold_12/pt_files'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Concatenation H&E + Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_Concatenation_HE_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR INTERMEDIATE FUSION — ELEMENT-WISE MULTIPLICATION
for split_name in ['train', 'test']:
    print(f"\n=== Element-wise Multiplication features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ELEMENT_WISE_MULT_HE_KI67/70%_split/5_class/features/conch_v1_5/{split_name}/fold_12/pt_files'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Element-wise Multiplication H&E + Ki-67 Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_ElemWiseMult_HE_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR INTERMEDIATE FUSION — H&E-GUIDED CROSS ATTENTION
for split_name in ['train', 'test']:
    print(f"\n=== H&E-guided Cross Attention features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_HE_inform_KI67/70%_split/5_class/features/conch_v1_5/{split_name}/fold_12/pt_files'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of H&E-guided Cross Attention Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_CrossAttn_HE_inform_KI67_features_5_class_{split_name}.png'))

# %% t-SNE FOR INTERMEDIATE FUSION — Ki-67-GUIDED CROSS ATTENTION
for split_name in ['train', 'test']:
    print(f"\n=== Ki-67-guided Cross Attention features — {split_name.upper()} ===")
    path = f'/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_KI67_inform_HE/70%_split/5_class/features/conch_v1_5/{split_name}/fold_12/pt_files'
    features_matrix, labels, _ = collect_features_and_labels(df, path, labels_to_include, split_case_ids[split_name])
    plot_tsne(features_matrix, labels, f't-SNE of Ki-67-guided Cross Attention Features ({split_name})',
              os.path.join(tsne_output_dir, f'tSNE_learned_CrossAttn_KI67_inform_HE_features_5_class_{split_name}.png'))

# %% LEGEND BOX
def plot_legend_box(save_path):
    custom_palette = {
        "LGG": "#5C92B1", "HGG": "#D32F2F", "Medulloblastoma": "#FF00FF", "Ependymoma": "#388E3C", "Ganglioglioma": "#FF8000"
    }
    legend_labels = ["LGG", "HGG", "Medulloblastoma", "Ependymoma", "Ganglioglioma"]
    handles = [mpatches.Patch(facecolor=custom_palette[label], edgecolor='black', linewidth=1, label=label) for label in legend_labels]
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.axis('off')
    leg = fig.legend(handles, legend_labels, title=None, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=5, fontsize=24, frameon=True)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close(fig)

plot_legend_box(os.path.join(base_output_dir, 'legend_box_5_class.png'))
# %%