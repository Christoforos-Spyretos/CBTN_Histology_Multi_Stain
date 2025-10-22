# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% UTILITY FUNCTIONS
def plot_logits_histogram(df, model_name, save_path=None):
    """
    Plot histogram of logits for LGG (label 0) and HGG (label 1)
    
    Args:
        df: DataFrame with columns ['Y', 'logits_0', 'logits_1']
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LGG samples (Y=0)
    lgg_samples = df[df['Y'] == 0]
    # HGG samples (Y=1)
    hgg_samples = df[df['Y'] == 1]
    
    # Plot logits_0 (LGG logits)
    ax = axes[0]
    ax.hist(lgg_samples['logits_0'], bins=30, alpha=0.6, label='True LGG', color='blue', edgecolor='black')
    ax.hist(hgg_samples['logits_0'], bins=30, alpha=0.6, label='True HGG', color='red', edgecolor='black')
    ax.set_xlabel('Logits for LGG (Class 0)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of LGG Logits\n{model_name}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot logits_1 (HGG logits)
    ax = axes[1]
    ax.hist(lgg_samples['logits_1'], bins=30, alpha=0.6, label='True LGG', color='blue', edgecolor='black')
    ax.hist(hgg_samples['logits_1'], bins=30, alpha=0.6, label='True HGG', color='red', edgecolor='black')
    ax.set_xlabel('Logits for HGG (Class 1)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of HGG Logits\n{model_name}', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{model_name} - Logits Statistics:")
    print(f"LGG samples (n={len(lgg_samples)}):")
    print(f"  logits_0 (LGG): mean={lgg_samples['logits_0'].mean():.3f}, std={lgg_samples['logits_0'].std():.3f}")
    print(f"  logits_1 (HGG): mean={lgg_samples['logits_1'].mean():.3f}, std={lgg_samples['logits_1'].std():.3f}")
    print(f"HGG samples (n={len(hgg_samples)}):")
    print(f"  logits_0 (LGG): mean={hgg_samples['logits_0'].mean():.3f}, std={hgg_samples['logits_0'].std():.3f}")
    print(f"  logits_1 (HGG): mean={hgg_samples['logits_1'].mean():.3f}, std={hgg_samples['logits_1'].std():.3f}")
    print("-" * 80)

# %% LOAD LOGITS DATA
# descriptive dataframe
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')

# logits data from best fold of KI67
HE_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1/fold_10.csv')
KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1/fold_10.csv')
early_fusion_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1/fold_10.csv')
late_fusion_LM_SL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/fold_10.csv')
late_fusion_LM_OHL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/fold_10.csv')
late_fusion_LM_THL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1/fold_10.csv')
late_fusion_LM_AL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/fold_10.csv')

# %% LOGITS DISTRIBUTION FOR HE
plot_logits_histogram(HE_logits, 'HE Model', save_path='logits_distribution_HE_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR KI67
plot_logits_histogram(KI67_logits, 'KI67 Model', save_path='logits_distribution_KI67_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR EARLY FUSION HE AND KI67
plot_logits_histogram(early_fusion_HE_KI67_logits, 'Early Fusion', save_path='logits_distribution_early_fusion_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION SIMPLE MODEL
plot_logits_histogram(late_fusion_LM_SL_HE_KI67_logits, 'Late Fusion - Single Layer', save_path='logits_distribution_late_fusion_SL_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION ONE HIDDEN LAYER MODEL
plot_logits_histogram(late_fusion_LM_OHL_HE_KI67_logits, 'Late Fusion - One Hidden Layer', save_path='logits_distribution_late_fusion_OHL_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION TWO HIDDEN LAYER MODEL
plot_logits_histogram(late_fusion_LM_THL_HE_KI67_logits, 'Late Fusion - Two Hidden Layers', save_path='logits_distribution_late_fusion_THL_LGG_vs_HGG.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION ATTENTION BASED MODEL
plot_logits_histogram(late_fusion_LM_AL_HE_KI67_logits, 'Late Fusion - Attention Layer', save_path='logits_distribution_late_fusion_AL_LGG_vs_HGG.png')

# %%
