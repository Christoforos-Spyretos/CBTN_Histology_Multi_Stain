# %% IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% UTILITY FUNCTIONS
def plot_logits_histogram(df, model_name, save_path=None):
    """
    Plot histogram of logits for 5 classes: LGG, HGG, MB, EP, GG
    
    Args:
        df: DataFrame with columns ['Y', 'logits_0', 'logits_1', 'logits_2', 'logits_3', 'logits_4']
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    # Class names
    class_names = ['LGG', 'HGG', 'MB', 'EP', 'GG']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot logits for each class
    for class_idx in range(5):
        ax = axes[class_idx]
        logit_col = f'logits_{class_idx}'
        
        # Plot histogram for each true class
        for true_class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
            class_samples = df[df['Y'] == true_class_idx]
            if len(class_samples) > 0:
                ax.hist(class_samples[logit_col], bins=30, alpha=0.5, 
                       label=f'True {class_name}', color=color, edgecolor='black')
        
        ax.set_xlabel(f'Logits for {class_names[class_idx]} (Class {class_idx})', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {class_names[class_idx]} Logits\n{model_name}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide the 6th subplot (we only have 5 classes)
    axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{model_name} - Logits Statistics:")
    for class_idx, class_name in enumerate(class_names):
        class_samples = df[df['Y'] == class_idx]
        if len(class_samples) > 0:
            print(f"\n{class_name} samples (n={len(class_samples)}):")
            for logit_idx in range(5):
                logit_col = f'logits_{logit_idx}'
                mean_val = class_samples[logit_col].mean()
                std_val = class_samples[logit_col].std()
                print(f"  {logit_col} ({class_names[logit_idx]}): mean={mean_val:.3f}, std={std_val:.3f}")
    print("-" * 80)

# %% LOAD LOGITS DATA
# descriptive dataframe
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')

# logits data from best fold of KI67
HE_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1/fold_47.csv')
KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1/fold_47.csv')
early_fusion_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_conch_v1/fold_47.csv')
late_fusion_LM_SL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1/fold_47.csv')
late_fusion_LM_OHL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1/fold_47.csv')
late_fusion_LM_THL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1/fold_47.csv')
late_fusion_LM_AL_HE_KI67_logits = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1/fold_47.csv')

# %% LOGITS DISTRIBUTION FOR HE
plot_logits_histogram(HE_logits, 'HE Model', save_path='logits_distribution_HE_5_class.png')

# %% LOGITS DISTRIBUTION FOR KI67
plot_logits_histogram(KI67_logits, 'KI67 Model', save_path='logits_distribution_KI67_5_class.png')

# %% LOGITS DISTRIBUTION FOR EARLY FUSION HE AND KI67
plot_logits_histogram(early_fusion_HE_KI67_logits, 'Early Fusion', save_path='logits_distribution_early_fusion_5_class.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION SIMPLE MODEL
plot_logits_histogram(late_fusion_LM_SL_HE_KI67_logits, 'Late Fusion - Single Layer', save_path='logits_distribution_late_fusion_SL_5_class.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION ONE HIDDEN LAYER MODEL
plot_logits_histogram(late_fusion_LM_OHL_HE_KI67_logits, 'Late Fusion - One Hidden Layer', save_path='logits_distribution_late_fusion_OHL_5_class.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION TWO HIDDEN LAYER MODEL
plot_logits_histogram(late_fusion_LM_THL_HE_KI67_logits, 'Late Fusion - Two Hidden Layers', save_path='logits_distribution_late_fusion_THL_5_class.png')

# %% LOGITS DISTRIBUTION FOR LATE FUSION ATTENTION BASED MODEL
plot_logits_histogram(late_fusion_LM_AL_HE_KI67_logits, 'Late Fusion - Attention Layer', save_path='logits_distribution_late_fusion_AL_5_class.png')

# %%
