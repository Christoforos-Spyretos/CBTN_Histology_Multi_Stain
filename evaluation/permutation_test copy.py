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

# %% PATHS TO RESULTS
# HE = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_HE_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'
# KI67 = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_KI67_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'
# GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_GFAP_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'
# HE_KI67 = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_HE_KI67_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'
# HE_GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_HE_GFAP_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'
# HE_KI67_GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_HE_KI67_GFAP_LGG_vs_HGG_bounded_clam_sb_small_uni_vit'

HE = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_class_bounded_small_clam_sb_uni_vit'
KI67 = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_KI67_class_bounded_small_clam_sb_uni_vit'
GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_GFAP_class_bounded_small_clam_sb_uni_vit'
HE_KI67 = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_KI67_class_bounded_small_clam_sb_uni_vit'
HE_GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_GFAP_class_bounded_small_clam_sb_uni_vit'
HE_KI67_GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_KI67_GFAP_class_bounded_small_clam_sb_uni_vit'

results_path_a = HE_KI67
results_path_b = HE_GFAP

contents_a = os.listdir(HE_KI67)
contents_b = os.listdir(HE_GFAP)

folds = [f'fold_{i}' for i in range(50)]

folds_dict_a = {} 

for content in contents_a:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(results_path_a + '/' + content)
        folds_dict_a[name] = df

for fold in folds:
    if fold in folds_dict_a:
        current_fold = folds_dict_a[fold]
        current_fold.rename(columns={
            "p_0": "ASTR_LGG_prob",
            "p_1": "ASTR_HGG_prob",
            "p_2": "EP_prob",
            "p_3": "MED_prob",
            "p_4": "GANG_prob"
            }, inplace=True)
        current_fold['true_label'] = current_fold['Y']
        current_fold.replace({'true_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG'
            }}, inplace=True)
        current_fold['predicted_label'] = current_fold['Y_hat']
        current_fold.replace({'predicted_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG'
            }}, inplace=True)
        
fold_values_a = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in folds_dict_a:
        fold_values_a[f'fold_{i + 1}'] = folds_dict_a[fold_key]

folds_dict_b = {} 

for content in contents_b:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(results_path_b + '/' + content)
        folds_dict_b[name] = df

for fold in folds:
    if fold in folds_dict_a:
        current_fold = folds_dict_b[fold]
        current_fold.rename(columns={
            "p_0": "ASTR_LGG_prob",
            "p_1": "ASTR_HGG_prob",
            "p_2": "EP_prob",
            "p_3": "MED_prob",
            "p_4": "GANG_prob"
            }, inplace=True)
        current_fold['true_label'] = current_fold['Y']
        current_fold.replace({'true_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG'
            }}, inplace=True)
        current_fold['predicted_label'] = current_fold['Y_hat']
        current_fold.replace({'predicted_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG'
            }}, inplace=True)
        
fold_values_b = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in folds_dict_b:
        fold_values_b[f'fold_{i + 1}'] = folds_dict_b[fold_key]

# %%
balanced_accuracies_a = []
balanced_accuracies_b = []

for fold in folds:
    fold_a = folds_dict_a[fold]
    b_acc_a = balanced_accuracy_score(fold_a['true_label'], fold_a['predicted_label'])
    balanced_accuracies_a.append(b_acc_a)

    fold_b = folds_dict_b[fold]
    b_acc_b = balanced_accuracy_score(fold_b['true_label'], fold_b['predicted_label'])
    balanced_accuracies_b.append(b_acc_b)

mccs_a = []
mccs_b = []

for fold in folds:
    fold_a = folds_dict_a[fold]
    mcc_a = matthews_corrcoef(fold_a['true_label'], fold_a['predicted_label'])
    mccs_a.append(mcc_a)

    fold_b = folds_dict_b[fold]
    mcc_b = matthews_corrcoef(fold_b['true_label'], fold_b['predicted_label'])
    mccs_b.append(mcc_b)

aucs_a = []
aucs_b = []

for fold_name in folds:
    if fold_name in folds_dict_a:
        fold_a = folds_dict_a[fold_name]
        fold_probs_a = fold_a.iloc[:, 3:8]
        auc_a = roc_auc_score(fold_a['Y'], fold_probs_a, average='weighted', multi_class='ovr')
        aucs_a.append(auc_a)

    if fold_name in folds_dict_b:
        fold_b = folds_dict_b[fold_name]
        fold_probs_b = fold_b.iloc[:, 3:8]
        auc_b = roc_auc_score(fold_b['Y'], fold_probs_b, average='weighted', multi_class='ovr')
        aucs_b.append(auc_b)

# aucs_hgg_a = []
# aucs_lgg_a = []
# aucs_hgg_b = []
# aucs_lgg_b = []

# def map_labels(label, target_class):
#     return 1 if label == target_class else 0

# for fold_name in folds:
#     if fold_name in folds_dict_a:
#         fold_a = folds_dict_a[fold_name]

#         true_labels_hgg_a = fold_a['true_label'].map(lambda x: map_labels(x, 'ASTR_HGG'))
#         predicted_probs_hgg_a = fold_a['ASTR_HGG_prob']
#         fpr_hgg_a, tpr_hgg_a, _ = roc_curve(true_labels_hgg_a, predicted_probs_hgg_a)
#         auc_hgg_a = auc(fpr_hgg_a, tpr_hgg_a)
#         aucs_hgg_a.append(auc_hgg_a)

#         true_labels_lgg_a = fold_a['true_label'].map(lambda x: map_labels(x, 'ASTR_LGG'))
#         predicted_probs_lgg_a = fold_a['ASTR_LGG_prob']
#         fpr_lgg_a, tpr_lgg_a, _ = roc_curve(true_labels_lgg_a, predicted_probs_lgg_a)
#         auc_lgg_a = auc(fpr_lgg_a, tpr_lgg_a)
#         aucs_lgg_a.append(auc_lgg_a)

#     if fold_name in folds_dict_b:
#         fold_b = folds_dict_b[fold_name]

#         true_labels_hgg_b = fold_b['true_label'].map(lambda x: map_labels(x, 'ASTR_HGG'))
#         predicted_probs_hgg_b = fold_b['ASTR_HGG_prob']
#         fpr_hgg_b, tpr_hgg_b, _ = roc_curve(true_labels_hgg_b, predicted_probs_hgg_b)
#         auc_hgg_b = auc(fpr_hgg_b, tpr_hgg_b)
#         aucs_hgg_b.append(auc_hgg_b)

#         true_labels_lgg_b = fold_b['true_label'].map(lambda x: map_labels(x, 'ASTR_LGG'))
#         predicted_probs_lgg_b = fold_b['ASTR_LGG_prob']
#         fpr_lgg_b, tpr_lgg_b, _ = roc_curve(true_labels_lgg_b, predicted_probs_lgg_b)
#         auc_lgg_b = auc(fpr_lgg_b, tpr_lgg_b)
#         aucs_lgg_b.append(auc_lgg_b)

# # Compute the mean AUC for each fold
# aucs_a = [(auc_hgg + auc_lgg) / 2 for auc_hgg, auc_lgg in zip(aucs_hgg_a, aucs_lgg_a)]
# aucs_b = [(auc_hgg + auc_lgg) / 2 for auc_hgg, auc_lgg in zip(aucs_hgg_b, aucs_lgg_b)]

weighted_f1s_a = []
weighted_f1s_b = []

for fold in folds:
    fold_a = folds_dict_a[fold]
    f1_a = f1_score(fold_a['true_label'], fold_a['predicted_label'], average='weighted')
    weighted_f1s_a.append(f1_a)

    if fold in folds_dict_b:
        fold_b = folds_dict_b[fold]
        f1_b = f1_score(fold_b['true_label'], fold_b['predicted_label'], average='weighted')
        weighted_f1s_b.append(f1_b)

# %% RUN PERMUTATION TEST
# settings for the permutation test
permutation_type = 'samples'
alternative = 'two-sided'
nbr_permutations = 10000

# significance value
a = 0.05

# Bonferroni correction
n_comparisons = 6
new_a = a/n_comparisons

population_a = aucs_a
population_b = aucs_b
        
res = permutation_test((population_a, population_b), statistic, vectorized=True, permutation_type=permutation_type,
                n_resamples=nbr_permutations, alternative=alternative, axis=0)
    
print(f"Bonferroni corrected significance level: {new_a}")
print(f"p-value: {res.pvalue}")
print(f"Statistics: {res.statistic}")

if res.pvalue < new_a:
    print("Conclusion: Reject the null hypothesis (significant difference)")
else:
    print("Conclusion: Fail to reject the null hypothesis (no significant difference)")

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
