# %% IMPORTS
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
import os
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, auc, roc_curve, f1_score

# %% UTILITIES
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# %% PATH TO RESULTS
HE_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_conch.csv'
KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_conch.csv'
GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
EF_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
AP_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
AL_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
MAJ_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
MAJ_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
MAJ_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
MLP_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
MLP_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
MLP_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
  

# %% LGG vs HGG combined

models = [
    HE_conch,
    KI67_conch,
    # GFAP_conch,
    EF_HE_KI67_conch,
    # EF_HE_GFAP_conch,
    # EF_HE_KI67_GFAP_conch,
    AP_HE_KI67_conch,
    # AP_HE_GFAP_conch,
    # AP_HE_KI67_GFAP_conch,
    AL_HE_KI67_conch,
    # AL_HE_GFAP_conch,
    # AL_HE_KI67_GFAP_conch,
    MAJ_HE_KI67_conch,
    # MAJ_HE_GFAP_conch,
    # MAJ_HE_KI67_GFAP_conch,
    # MLP_HE_KI67_conch,
    # MLP_HE_GFAP_conch,
    MLP_HE_KI67_GFAP_conch
]


model_names = [
    "HE_conch",
    "KI67_conch",
    # "GFAP_conch",
    "EF_HE_KI67_conch",
    # "EF_HE_GFAP_conch",
    # "EF_HE_KI67_GFAP_conch",
    "AP_HE_KI67_conch",
    # "AP_HE_GFAP_conch",
    # "AP_HE_KI67_GFAP_conch",
    "AL_HE_KI67_conch",
    # "AL_HE_GFAP_conch",
    # "AL_HE_KI67_GFAP_conch",
    "MAJ_HE_KI67_conch",
    # "MAJ_HE_GFAP_conch",
    # "MAJ_HE_KI67_GFAP_conch",
    # "MLP_HE_KI67_conch",
    # "MLP_HE_GFAP_conch",
    "MLP_HE_KI67_GFAP_conch"  
]

# %%
# Load balanced accuracy scores from all models
performance = []

metric = 'Balanced_Accuracy' # Balanced_Accuracy, MCC, AUC,F1-Score

for model_path in models:
    df = pd.read_csv(model_path)
    # Extract the Balanced Accuracy column
    performance.append(df[metric].values)

# %% RUN PERMUTATION TEST
# Perform pairwise permutation tests
results = []
num_comparisons = 21
adjusted_alpha = 0.05 / num_comparisons  # Bonferroni correction

for (i, j) in itertools.combinations(range(len(models)), 2):
    model_a_name, model_b_name = model_names[i], model_names[j]
    perf_a, perf_b = performance[i], performance[j]
    
    # Perform permutation test
    res = permutation_test(
        (perf_a, perf_b), statistic, vectorized=True, 
        permutation_type='samples', n_resamples=10000, alternative='two-sided'
    )
    
    results.append({
        "Model A": model_a_name,
        "Model B": model_b_name,
        "Statistic": res.statistic,
        "p-value": res.pvalue,
        "Significant": res.pvalue < adjusted_alpha
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# %%
# Save results to a CSV file
results_df.to_csv('LGG_vs_HGG_class_conch_ba_perm_test.csv', index=False)

# %%
# import matplotlib.pyplot as plt 

# mean_diff = np.mean(np.array(aucs_a) - np.array(aucs_b))

# # Plot the distributions
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.hist(aucs_a, bins=20, alpha=0.7, label='AUCs A')
# plt.hist(aucs_b, bins=20, alpha=0.7, label='AUCs B')
# plt.legend()

# %%
# perm_diffs = []

# for n in range(nbr_permutations):
#     idxs = np.random.randint(0,49,25)
#     p_aucs_a = np.array(aucs_a)
#     p_aucs_a[idxs] = np.array(aucs_b)[idxs]
#     p_aucs_b = np.array(aucs_b)
#     p_aucs_b[idxs] = np.array(aucs_a)[idxs]
#     perm_diffs.append(statistic(p_aucs_a,p_aucs_b,axis = 0))

# plt.hist(perm_diffs, bins=20)

# binary = np.array(np.abs(perm_diffs)) > np.abs(res.statistic)

# perc = sum(binary)/len(binary)
# %%
