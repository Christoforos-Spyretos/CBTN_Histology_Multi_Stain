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
HE_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1.csv'
KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1.csv'

EF_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1.csv'

IM_CA_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1.csv'

PA_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1.csv'
LA_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1.csv'
MJ_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_MJ_HE_KI67_small_clam_sb_conch_v1.csv'
LM_SL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1.csv'
LM_OHL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1.csv'
LM_AL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv'

# %% LGG vs HGG combined
models = [
    # HE_conch,
    KI67_conch,

    # EF_HE_KI67_conch,

    # IM_CA_HE_KI67_conch,

    # PA_HE_KI67_conch,
    # LA_HE_KI67_conch,
    # MJ_HE_KI67_conch,
    # LM_SL_HE_KI67_conch,
    LM_OHL_HE_KI67_conch,
    # LM_AL_HE_KI67_conch
]


model_names = [
    # "HE_conch",
    "KI67_conch",

    # "EF_HE_KI67_conch",

    # "IM_CA_HE_KI67_conch",

    # "PA_HE_KI67_conch",
    # "LA_HE_KI67_conch",
    # "MJ_HE_KI67_conch",
    # "LM_SL_HE_KI67_conch",
    "LM_OHL_HE_KI67_conch",
    # "LM_AL_HE_KI67_conch"
]

# %%
# Load balanced accuracy scores from all models
performance = []

metrics = ['BA', 'MCC', 'AUC', 'F1-Score']

for model_path in models:
    df = pd.read_csv(model_path)
    # Extract the Balanced Accuracy column
    performance.append({metric: df[metric].values for metric in metrics})

# %% RUN PERMUTATION TEST

# Perform pairwise permutation tests for all metrics
results = []
num_models = len(models)
num_metrics = len(metrics)
num_comparisons = (num_models * (num_models - 1) // 2) * num_metrics
adjusted_alpha = 0.05 / num_comparisons  # Bonferroni correction

for (i, j) in itertools.combinations(range(num_models), 2):
    model_a_name, model_b_name = model_names[i], model_names[j]
    perf_a, perf_b = performance[i], performance[j]
    for metric in metrics:
        # Perform permutation test for each metric
        res = permutation_test(
            (perf_a[metric], perf_b[metric]), statistic, vectorized=True,
            permutation_type='samples', n_resamples=10000, alternative='two-sided'
        )
        results.append({
            "Model A": model_a_name,
            "Model B": model_b_name,
            "Metric": metric,
            "Statistic": res.statistic,
            "p-value": res.pvalue,
            "Significant": res.pvalue < adjusted_alpha
        })

# Display results
results_df = pd.DataFrame(results)
print(results_df)

# %%
# Save results to a CSV file
results_df.to_csv('LGG_vs_HGG_class_conch_perm_test.csv', index=False)

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
