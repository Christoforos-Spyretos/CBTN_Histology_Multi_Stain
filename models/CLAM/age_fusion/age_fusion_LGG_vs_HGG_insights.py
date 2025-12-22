# %% IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import  torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.manifold import TSNE
import seaborn as sns

# %% AGE INFORMATION 
# load the dataframes
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv')
histological_diagnosis_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Histological Diagnoses', engine='openpyxl')
participants_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Participants', engine='openpyxl')

# map the age at diagnosis to df based on case_id
df['age_at_diagnosis_(days)'] = df['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Age at Diagnosis (Days)'].to_dict())

df['gender'] = df['case_id'].map(participants_df_from_portal.set_index('External Id')['Gender'].to_dict())

# convert age from days to years
df['age_at_diagnosis_(years)'] = df['age_at_diagnosis_(days)'] / 365.25

# descriptive statistics for age at diagnosis (years) for each label
labels = df['label'].unique()
for label in labels:
    label_df = df[df['label'] == label]
    age_data = label_df['age_at_diagnosis_(years)'].dropna()
    print(f"\nDescriptive Statistics for Age at Diagnosis (Years) - {label}:")
    print(f"Count: {age_data.count()}")
    print(f"Mean: {age_data.mean():.2f}")
    print(f"Median: {age_data.median():.2f}")
    print(f"Standard Deviation: {age_data.std():.2f}")
    print(f"Minimum: {age_data.min():.2f}")
    print(f"Maximum: {age_data.max():.2f}")
    print(f"25th Percentile: {age_data.quantile(0.25):.2f}")
    print(f"75th Percentile: {age_data.quantile(0.75):.2f}")

# %% FEATURES INFORMATION
# paths to pt files
HE_features_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_HE/features/conch_v1/pt_files"
KI67_features_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_KI67/features/conch_v1/pt_files"
Merged_features_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_HE_KI67/features/conch_v1/pt_files"

# %% LOAD AND ANALYZE FEATURE RANGES
def load_features_for_cases(feature_path, df):
    """Load features for all cases from pt files"""
    features_list = []
    labels_list = []
    case_ids_list = []
    
    for idx, row in df.iterrows():
        case_id = row['case_id']
        label = row['label']
        feature_file = os.path.join(feature_path, f"{case_id}.pt")
        
        if os.path.exists(feature_file):
            try:
                # load the feature file
                data = torch.load(feature_file, map_location='cpu', weights_only=False)
                
                if isinstance(data, dict):
                    if 'subject_attention' in data:
                        feature = data['subject_attention']
                    elif 'features' in data:
                        feature = data['features']
                    elif 'feat' in data:
                        feature = data['feat']
                    else:
                        # take first tensor value
                        for value in data.values():
                            if isinstance(value, (torch.Tensor, np.ndarray)):
                                feature = value
                                break
                else:
                    feature = data
                
                # convert to numpy
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                
                # if it's 2D like (512, 1) or (n, d), flatten or take mean
                if len(feature.shape) > 1:
                    if feature.shape[1] == 1:
                        # (n, 1) -> (n,)
                        feature = feature.squeeze()
                    else:
                        # (n, d) -> (d,) by taking mean
                        feature = feature.mean(axis=0)
                
                # ensure it's 1D
                if len(feature.shape) == 0:
                    feature = feature.reshape(1)
                
                features_list.append(feature)
                labels_list.append(label)
                case_ids_list.append(case_id)
            except Exception as e:
                print(f"Error loading {feature_file}: {e}")
                import traceback
                traceback.print_exc()
    
    # if features_list:
    #     features_array = np.stack(features_list)
    #     print(f"Loaded {len(features_list)} samples with shape {features_array.shape}")
    #     return features_array, labels_list, case_ids_list
    # else:
    #     print(f"No features loaded from {feature_path}")
    return None, None, None

# Load all features
print("Loading HE features...")
HE_features, HE_labels, HE_case_ids = load_features_for_cases(HE_features_path, df)

print("Loading KI67 features...")
KI67_features, KI67_labels, KI67_case_ids = load_features_for_cases(KI67_features_path, df)

print("Loading Merged features...")
Merged_features, Merged_labels, Merged_case_ids = load_features_for_cases(Merged_features_path, df)

print(f"\nLoaded features:")
print(f"HE: {HE_features.shape if HE_features is not None else 'None'}")
print(f"KI67: {KI67_features.shape if KI67_features is not None else 'None'}")
print(f"Merged: {Merged_features.shape if Merged_features is not None else 'None'}")

if HE_features is not None:
    print(f"\nHE Features:")
    print(f"  Shape: {HE_features.shape}")
    print(f"  Min: {HE_features.min():.6f}, Max: {HE_features.max():.6f}")
    print(f"  Mean: {HE_features.mean():.6f}, Std: {HE_features.std():.6f}")
if KI67_features is not None:
    print(f"\nKI67 Features:")
    print(f"  Shape: {KI67_features.shape}")
    print(f"  Min: {KI67_features.min():.6f}, Max: {KI67_features.max():.6f}")
    print(f"  Mean: {KI67_features.mean():.6f}, Std: {KI67_features.std():.6f}")
if Merged_features is not None:
    print(f"\nMerged Features:")
    print(f"  Shape: {Merged_features.shape}")
    print(f"  Min: {Merged_features.min():.6f}, Max: {Merged_features.max():.6f}")
    print(f"  Mean: {Merged_features.mean():.6f}, Std: {Merged_features.std():.6f}")

def analyze_feature_ranges(features, labels, feature_name):
    if features is None:
        print(f"\n{feature_name}: No data available")
        return None
    
    print(f"{feature_name} - Feature Value Range Analysis")
    
    # overall statistics
    overall_min = features.min()
    overall_max = features.max()
    overall_mean = features.mean()
    overall_std = features.std()
    
    print(f"\nOverall (all labels combined):")
    print(f"  Min: {overall_min:.6f}")
    print(f"  Max: {overall_max:.6f}")
    print(f"  Mean: {overall_mean:.6f}")
    print(f"  Std: {overall_std:.6f}")
    print(f"  Range: [{overall_min:.6f}, {overall_max:.6f}]")
    
    # per-label statistics
    unique_labels = np.unique(labels)
    print(f"\nPer-label statistics:")
    
    for label in unique_labels:
        mask = np.array(labels) == label
        label_features = features[mask]
        
        label_min = label_features.min()
        label_max = label_features.max()
        label_mean = label_features.mean()
        label_std = label_features.std()
        
        print(f"\n  Label: {label} (n={mask.sum()} samples)")
        print(f"    Min: {label_min:.6f}")
        print(f"    Max: {label_max:.6f}")
        print(f"    Mean: {label_mean:.6f}")
        print(f"    Std: {label_std:.6f}")
    
    return {
        'overall_min': overall_min,
        'overall_max': overall_max,
        'overall_mean': overall_mean,
        'overall_std': overall_std
    }

he_stats = analyze_feature_ranges(HE_features, HE_labels, "HE Features")
ki67_stats = analyze_feature_ranges(KI67_features, KI67_labels, "KI67 Features")
merged_stats = analyze_feature_ranges(Merged_features, Merged_labels, "Merged HE & KI67 Features")

# %% AGE NORMALIZATION USING Z-SCORE STANDARDIZATION
age_min = df['age_at_diagnosis_(years)'].min()
age_max = df['age_at_diagnosis_(years)'].max()
age_mean = df['age_at_diagnosis_(years)'].mean()
age_std = df['age_at_diagnosis_(years)'].std()

print(f"\nAge Statistics:")
print(f"  Min: {age_min:.2f} years")
print(f"  Max: {age_max:.2f} years")
print(f"  Mean: {age_mean:.2f} years")
print(f"  Std: {age_std:.2f} years")
print(f"  Range: [{age_min:.2f}, {age_max:.2f}]")

print("\nZ-score (Standardization):")
print(f"  Formula: age_normalized = (age - {age_mean:.2f}) / {age_std:.2f}")
df['age_normalized'] = (df['age_at_diagnosis_(years)'] - age_mean) / age_std
print(f"  Result range: [{df['age_normalized'].min():.6f}, {df['age_normalized'].max():.6f}]")

if he_stats is not None:
    print(f"\nFeature range: [0, ~{he_stats['overall_max']:.2f}]")
print(f"Age Z-score range: [{df['age_normalized'].min():.2f}, {df['age_normalized'].max():.2f}]")
# Save the dataframe with normalized age
output_csv = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset_with_normalized_age.csv'
df.to_csv(output_csv, index=False)
output_csv = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset_with_normalized_age.csv'
df.to_csv(output_csv, index=False)

# %% VISUALIZE FEATURE VALUE DISTRIBUTIONS BY LABEL
def plot_feature_distributions_by_label(features, labels, feature_name):
    if features is None:
        print(f"{feature_name}: No data available")
        return
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # create figure with one subplot per label
    fig, axes = plt.subplots(1, n_labels, figsize=(7*n_labels, 5))
    if n_labels == 1:
        axes = [axes]
    
    for idx, label in enumerate(unique_labels):
        # get all features for this label
        mask = np.array(labels) == label
        label_features = features[mask]
        
        # flatten all values from all samples for this label
        all_values = label_features.flatten()
        
        # plot histogram
        axes[idx].hist(all_values, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        axes[idx].set_xlabel('Feature Value', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{feature_name}\nLabel: {label}\n({mask.sum()} samples, {label_features.shape[1]} dims)', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        stats_text = (f'Min: {all_values.min():.4f}\n'
                     f'Max: {all_values.max():.4f}\n'
                     f'Mean: {all_values.mean():.4f}\n'
                     f'Std: {all_values.std():.4f}\n')
        axes[idx].text(0.98, 0.97, stats_text, transform=axes[idx].transAxes,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                      fontsize=9, family='monospace')
        
        axes[idx].legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    filename = f'feature_distribution_{feature_name.replace(" ", "_")}_by_label.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# plot distributions for each feature type
if HE_features is not None:
    plot_feature_distributions_by_label(HE_features, HE_labels, 'HE Features')

if KI67_features is not None:
    plot_feature_distributions_by_label(KI67_features, KI67_labels, 'KI67 Features')

if Merged_features is not None:
    plot_feature_distributions_by_label(Merged_features, Merged_labels, 'Merged HE & KI67 Features')

# %% COMBINED COMPARISON PLOT
def plot_combined_comparison(features_dict, labels_dict):
    """Plot combined comparison of feature distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # plot 1: Overlaid histograms by feature type (all labels combined)
    ax = axes[0, 0]
    for feature_name, features in features_dict.items():
        if features is not None:
            ax.hist(features.flatten(), bins=100, alpha=0.4, label=feature_name, density=True)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1, color='g', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-1, color='b', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.set_title('All Feature Types - Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # plot 2: Box plots by feature type
    ax = axes[0, 1]
    data_to_plot = []
    labels_plot = []
    for feature_name, features in features_dict.items():
        if features is not None:
            data_to_plot.append(features.flatten())
            labels_plot.append(feature_name)
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=1, color='g', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=-1, color='b', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Value Ranges (Box Plots)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # plot 3: Original age distribution
    ax = axes[1, 0]
    age_data = df['age_at_diagnosis_(years)'].dropna()
    ax.hist(age_data, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax.set_xlabel('Age at Diagnosis (years)')
    ax.set_ylabel('Frequency')
    ax.set_title('Original Age Distribution')
    ax.grid(True, alpha=0.3)
    
    # plot 4: Normalized age distribution (Z-score)
    ax = axes[1, 1]
    ax.hist(df['age_normalized'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='coral')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.6, label='Mean (0)')
    ax.axvline(x=df['age_normalized'].mean(), color='darkred', linestyle='-', linewidth=1, alpha=0.4)
    ax.set_xlabel('Z-score Normalized Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Age Distribution (Z-score Standardized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # add statistics text
    stats_text = f'Min: {df["age_normalized"].min():.2f}\nMax: {df["age_normalized"].max():.2f}\nMean: {df["age_normalized"].mean():.2f}\nStd: {df["age_normalized"].std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig('feature_and_age_normalization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: feature_and_age_normalization_analysis.png")

features_dict = {
    'HE': HE_features,
    'KI67': KI67_features,
    'Merged': Merged_features
}

labels_dict = {
    'HE': HE_labels,
    'KI67': KI67_labels,
    'Merged': Merged_labels
}

plot_combined_comparison(features_dict, labels_dict)

# %% ADD AGE TO FEATURES AND SAVE NEW .PT FILES
def add_age_to_features(feature_path, output_path, df):
    """Add normalized age to features and save as new .pt files"""
    os.makedirs(output_path, exist_ok=True)
    
    saved_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        case_id = row['case_id']
        age_normalized = row['age_normalized']
        
        feature_file = os.path.join(feature_path, f"{case_id}.pt")
        
        if os.path.exists(feature_file) and not pd.isna(age_normalized):
            try:
                # Load original feature
                data = torch.load(feature_file, map_location='cpu', weights_only=False)
                
                if isinstance(data, dict) and 'subject_attention' in data:
                    feature = data['subject_attention']
                else:
                    continue
                
                # Convert to numpy
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                
                # Squeeze to 1D (512,)
                if len(feature.shape) > 1:
                    feature = feature.squeeze()
                
                # Append age as additional feature (512,) -> (513,)
                feature_with_age = np.append(feature, age_normalized)
                
                # Save new feature dict with age appended
                new_data = {
                    'subject_attention': feature_with_age,
                    'age_normalized': age_normalized
                }
                
                output_file = os.path.join(output_path, f"{case_id}.pt")
                torch.save(new_data, output_file)
                saved_count += 1
                    
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"\nSuccessfully saved {saved_count} files with age features")
    print(f"Skipped {skipped_count} files (missing features or age)")
    return saved_count

# Create output directories
HE_output_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_HE/features_with_age/conch_v1/pt_files"
KI67_output_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_KI67/features_with_age/conch_v1/pt_files"
Merged_output_path = "/local/data3/chrsp39/CBTN_v2/Learned_Subject_Level_Features/LGG_vs_HGG/Merged_HE_KI67/features_with_age/conch_v1/pt_files"

print("\nAdding age to HE features...")
add_age_to_features(HE_features_path, HE_output_path, df)

print("\nAdding age to KI67 features...")
add_age_to_features(KI67_features_path, KI67_output_path, df)

print("\nAdding age to Merged features...")
add_age_to_features(Merged_features_path, Merged_output_path, df)

# %% PLOT t-SNE WITH AGE-AUGMENTED FEATURES
def load_features_with_age(feature_path, df, labels_to_include=None):
    """Load features with age appended (513 dimensions)"""
    all_features = []
    all_labels = []
    all_case_ids = []
    
    if labels_to_include is not None:
        df_filtered = df[df['label'].isin(labels_to_include)]
    else:
        df_filtered = df
    
    for idx, row in df_filtered.iterrows():
        case_id = row['case_id']
        label = row['label']
        
        pt_file_path = os.path.join(feature_path, f"{case_id}.pt")
        
        if os.path.exists(pt_file_path):
            try:
                data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
                
                if isinstance(data, dict) and 'subject_attention' in data:
                    features = data['subject_attention']
                    
                    # Convert to numpy
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    
                    all_features.append(features)
                    all_labels.append(label)
                    all_case_ids.append(case_id)
                    
            except Exception as e:
                print(f"Error loading {case_id}: {e}")
    
    print(f"  Loaded {len(all_features)} cases")
    return np.array(all_features), np.array(all_labels), np.array(all_case_ids)

def plot_tsne_with_age(features, labels, title, save_path):
    """Plot t-SNE of age-augmented features"""
    
    print(f"  Running t-SNE on {features.shape[0]} samples with {features.shape[1]} features...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.6, s=50)
    
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=12, title_fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")

labels_to_include = ['LGG', 'HGG']

# Plot t-SNE for HE features with age
print("\nProcessing HE features with age...")
HE_features_age, HE_labels_age, HE_case_ids_age = load_features_with_age(
    HE_output_path, df, labels_to_include)
if len(HE_features_age) > 0:
    print(f"  Feature shape: {HE_features_age.shape}")
    plot_tsne_with_age(HE_features_age, HE_labels_age, 
                       't-SNE of HE Features with Age',
                       'tSNE_HE_features_with_age_LGG_vs_HGG.png')

# Plot t-SNE for KI67 features with age
print("\nProcessing KI67 features with age...")
KI67_features_age, KI67_labels_age, KI67_case_ids_age = load_features_with_age(
    KI67_output_path, df, labels_to_include)
if len(KI67_features_age) > 0:
    print(f"  Feature shape: {KI67_features_age.shape}")
    plot_tsne_with_age(KI67_features_age, KI67_labels_age,
                       't-SNE of KI67 Features with Age',
                       'tSNE_KI67_features_with_age_LGG_vs_HGG.png')

# Plot t-SNE for Merged features with age
print("\nProcessing Merged features with age...")
Merged_features_age, Merged_labels_age, Merged_case_ids_age = load_features_with_age(
    Merged_output_path, df, labels_to_include)
if len(Merged_features_age) > 0:
    print(f"  Feature shape: {Merged_features_age.shape}")
    plot_tsne_with_age(Merged_features_age, Merged_labels_age,
                       't-SNE of Merged HE & KI67 Features with Age',
                       'tSNE_Merged_features_with_age_LGG_vs_HGG.png')

# %%

