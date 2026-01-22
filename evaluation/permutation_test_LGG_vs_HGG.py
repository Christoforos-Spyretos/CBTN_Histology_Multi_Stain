# %% IMPORTS
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
import os
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, auc, roc_curve, f1_score

# %% UTILITIES
# Set random seed for reproducibility
np.random.seed(42)

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# %% PATH TO RESULTS
HE_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1.csv'
HE_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1_5.csv'
HE_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_resnet50.csv'
HE_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_uni.csv'
HE_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_uni2-h.csv'

KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1.csv'
KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5.csv'
KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_resnet50.csv'
KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_uni.csv'
KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_uni2-h.csv'

EF_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1.csv'
EF_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1_5.csv'
EF_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_resnet50.csv'
EF_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_uni.csv'
EF_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_uni2-h.csv'

IM_CA_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1.csv'
IM_CA_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1_5.csv'
IM_CA_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_resnet50.csv'
IM_CA_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_uni.csv'
IM_CA_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_uni2-h.csv'

PA_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1.csv'
LA_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1.csv'
LM_SL_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1.csv'
LM_OHL_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1.csv'
LM_THL_LM_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1.csv'
LM_AL_HE_KI67_conch_v1 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv'

PA_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5.csv'
LA_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_SL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_OHL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_THL_LM_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_AL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5.csv'

PA_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_resnet50.csv'
LA_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_resnet50.csv'
LM_SL_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_resnet50.csv'
LM_OHL_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_resnet50.csv'
LM_THL_LM_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_resnet50.csv'
LM_AL_HE_KI67_resnet50 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_resnet50.csv'

PA_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_uni.csv'
LA_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_uni.csv'
LM_SL_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_uni.csv'
LM_OHL_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_uni.csv'
LM_THL_LM_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_uni.csv'
LM_AL_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_uni.csv'

PA_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_uni2-h.csv'
LA_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_uni2-h.csv'
LM_SL_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_uni2-h.csv'
LM_OHL_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_uni2-h.csv'
LM_THL_LM_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_uni2-h.csv'
LM_AL_HE_KI67_uni2_h = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_uni2-h.csv'


# %% LGG vs HGG combined
models = [
    # HE_conch_v1,
    HE_conch_v1_5,
    # HE_resnet50,
    # HE_uni,
    # HE_uni2_h,

    # KI67_conch_v1,
    KI67_conch_v1_5,
    # KI67_resnet50,
    # KI67_uni,
    # KI67_uni2_h,

    # EF_HE_KI67_conch_v1,
    EF_HE_KI67_conch_v1_5,
    # EF_HE_KI67_resnet50,
    # EF_HE_KI67_uni,
    # EF_HE_KI67_uni2_h,

    # IM_CA_HE_KI67_conch_v1,
    IM_CA_HE_KI67_conch_v1_5,
    # IM_CA_HE_KI67_resnet50,
    # IM_CA_HE_KI67_uni,
    # IM_CA_HE_KI67_uni2_h,

    # PA_HE_KI67_conch_v1,
    # LA_HE_KI67_conch_v1,
    # LM_SL_HE_KI67_conch_v1,
    # LM_OHL_HE_KI67_conch_v1,
    # LM_THL_LM_HE_KI67_conch_v1,
    # LM_AL_HE_KI67_conch_v1,

    PA_HE_KI67_conch_v1_5,
    # LA_HE_KI67_conch_v1_5,
    # LM_SL_HE_KI67_conch_v1_5,
    # LM_OHL_HE_KI67_conch_v1_5,
    # LM_THL_LM_HE_KI67_conch_v1_5,
    # LM_AL_HE_KI67_conch_v1_5,

    # PA_HE_KI67_resnet50,
    # LA_HE_KI67_resnet50,
    # LM_SL_HE_KI67_resnet50,
    # LM_OHL_HE_KI67_resnet50,
    # LM_THL_LM_HE_KI67_resnet50,
    # LM_AL_HE_KI67_resnet50,

    # PA_HE_KI67_uni,
    # LA_HE_KI67_uni,
    # LM_SL_HE_KI67_uni,
    # LM_OHL_HE_KI67_uni,
    # LM_THL_LM_HE_KI67_uni,
    # LM_AL_HE_KI67_uni,

    # PA_HE_KI67_uni2_h,
    # LA_HE_KI67_uni2_h,
    # LM_SL_HE_KI67_uni2_h,
    # LM_OHL_HE_KI67_uni2_h,
    # LM_THL_LM_HE_KI67_uni2_h,
    # LM_AL_HE_KI67_uni2_h
]


model_names = [
    # "HE_conch_v1",
    "HE_conch_v1_5",
    # "HE_resnet50",
    # "HE_uni",
    # "HE_uni2-h",

    # "KI67_conch_v1",
    "KI67_conch_v1_5",
    # "KI67_resnet50",
    # "KI67_uni",
    # "KI67_uni2-h",

    # "EF_HE_KI67_conch_v1",
    "EF_HE_KI67_conch_v1_5",
    # "EF_HE_KI67_resnet50",
    # "EF_HE_KI67_uni",
    # "EF_HE_KI67_uni2-h",

    # "IM_CA_HE_KI67_conch_v1",
    "IM_CA_HE_KI67_conch_v1_5",
    # "IM_CA_HE_KI67_resnet50",
    # "IM_CA_HE_KI67_uni",
    # "IM_CA_HE_KI67_uni2-h",

    # "PA_HE_KI67_conch_v1",
    # "LA_HE_KI67_conch_v1",
    # "LM_SL_HE_KI67_conch_v1",
    # "LM_OHL_HE_KI67_conch_v1",
    # "LM_THL_LM_HE_KI67_conch_v1",
    # "LM_AL_HE_KI67_conch_v1",

    "PA_HE_KI67_conch_v1_5",
    # "LA_HE_KI67_conch_v1_5",
    # "LM_SL_HE_KI67_conch_v1_5",
    # "LM_OHL_HE_KI67_conch_v1_5",
    # "LM_THL_LM_HE_KI67_conch_v1_5",
    # "LM_AL_HE_KI67_conch_v1_5",

    # "PA_HE_KI67_resnet50",
    # "LA_HE_KI67_resnet50",
    # "LM_SL_HE_KI67_resnet50",
    # "LM_OHL_HE_KI67_resnet50",
    # "LM_THL_LM_HE_KI67_resnet50",
    # "LM_AL_HE_KI67_resnet50",

    # "PA_HE_KI67_uni",
    # "LA_HE_KI67_uni",
    # "LM_SL_HE_KI67_uni",
    # "LM_OHL_HE_KI67_uni",
    # "LM_THL_LM_HE_KI67_uni",
    # "LM_AL_HE_KI67_uni",

    # "PA_HE_KI67_uni2-h",
    # "LA_HE_KI67_uni2-h",
    # "LM_SL_HE_KI67_uni2-h",
    # "LM_OHL_HE_KI67_uni2-h",
    # "LM_THL_LM_HE_KI67_uni2-h",
    # "LM_AL_HE_KI67_uni2-h"
]

# %%
# Load balanced accuracy scores from all models
performance = []

# metrics = ['BA', 'MCC', 'AUC', 'F1-Score']
metrics = ['BA', 'MCC']

for model_path in models:
    df = pd.read_csv(model_path)
    # Extract the Balanced Accuracy column
    performance.append({metric: df[metric].values for metric in metrics})

# %% DOUBLE SIDED PERMUTATION TEST
print("DOUBLE SIDED PERMUTATION TEST RESULTS")
# Perform pairwise permutation tests for all metrics
results_two_sided = []
num_models = len(models)
num_metrics = len(metrics)
num_comparisons = (num_models * (num_models - 1) // 2)
adjusted_alpha = 0.05 / num_comparisons  # Bonferroni correction
print(f"Adjusted alpha level after Bonferroni correction: {adjusted_alpha}")

for (i, j) in itertools.combinations(range(num_models), 2):
    model_a_name, model_b_name = model_names[i], model_names[j]
    perf_a, perf_b = performance[i], performance[j]
    for metric in metrics:
        # Perform permutation test for each metric
        res = permutation_test(
            (perf_a[metric], perf_b[metric]), statistic, vectorized=True,
            permutation_type='samples', n_resamples=10000, alternative='two-sided'
        )
        results_two_sided.append({
            "Model A": model_a_name,
            "Model B": model_b_name,
            "Metric": metric,
            "Statistic": res.statistic,
            "p-value": res.pvalue,
            "Significant": res.pvalue < adjusted_alpha
        })

results_df_two_sided = pd.DataFrame(results_two_sided)
print(results_df_two_sided)

# %% ONE SIDED PERMUTATION TEST
print("ONE SIDED PERMUTATION TEST RESULTS")
print(f"Adjusted alpha level after Bonferroni correction: {adjusted_alpha}")
results_one_sided = []

for (i, j) in itertools.combinations(range(num_models), 2):
    model_a_name, model_b_name = model_names[i], model_names[j]
    perf_a, perf_b = performance[i], performance[j]
    for metric in metrics:
        # Perform permutation test for each metric
        res = permutation_test(
            (perf_a[metric], perf_b[metric]), statistic, vectorized=True,
            permutation_type='samples', n_resamples=10000, alternative='greater'
        )
        results_one_sided.append({
            "Model A": model_a_name,
            "Model B": model_b_name,
            "Metric": metric,
            "Statistic": res.statistic,
            "p-value": res.pvalue,
            "Significant": res.pvalue < adjusted_alpha
        })

results_df_one_sided = pd.DataFrame(results_one_sided)
print(results_df_one_sided)

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
