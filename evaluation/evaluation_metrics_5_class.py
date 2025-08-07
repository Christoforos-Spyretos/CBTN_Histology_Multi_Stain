# TO DO
'''
1) Overall confusion matrix
2) Add graphs
3) Organise code base on number of classes
4) Better Printing
'''

# %% IMPORTS
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, matthews_corrcoef, auc, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# %% LOAD RESULTS
# path to results
results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Single_Stain/small_clam_sb/conch/EVAL_5_class_Merged_GFAP_small_clam_sb_conch'
contents = os.listdir(results_path)

folds_dict = {} 

for content in contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(results_path + '/' + content)
        folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]

for fold in folds:
    if fold in folds_dict:
        current_fold = folds_dict[fold]
        current_fold.rename(columns={
            "p_0": "ASTR_LGG_prob",
            "p_1": "ASTR_HGG_prob",
            "p_2": "EP_prob",
            "p_3": "MED_prob",
            "p_4": "GANG_prob",
            }, inplace=True)
        
        current_fold['true_label'] = current_fold['Y']

        current_fold.replace({'true_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG',
            }}, inplace=True)
        
        current_fold['predicted_label'] = current_fold['Y_hat']

        current_fold.replace({'predicted_label': {
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'EP',
            3.0: 'MED',
            4.0: 'GANG',
            }}, inplace=True)

classes = ['ASTR_LGG','ASTR_HGG','EP','MED','GANG']

fold_values = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in folds_dict:
        fold_values[f'fold_{i + 1}'] = folds_dict[fold_key]

# %% create a csv file with the summary of the results
summary = pd.DataFrame(columns=[
    'Task', 'Fusion', 'Modality', 'Feature_Extractor', 'Aggregation_Method',
    'Fold', 'Balanced_Accuracy', 'MCC', 'AUC', 'F1-Score'
])

for fold in folds:
    new_row = pd.DataFrame({
        'Task': ['5_class'],
        'Fusion': ['Single_Stain'],
        'Modality': ['GFAP'],
        'Feature_Extractor': ['conch'],
        'Aggregation_Method': ['small_clam_sb'],
        'Fold': [str(fold)],
        'Balanced_Accuracy': [0],
        'MCC': [0],
        'AUC': [0],
        'F1-Score': [0],
    })
    summary = pd.concat([summary, new_row], ignore_index=True)

# save the summary dataframe to a csv file
save_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results'
save_name = 'EVAL_5_class_Merged_GFAP_small_clam_sb_conch.csv'
summary.to_csv(os.path.join(save_path, save_name), index=False)

# summary = folds_dict['summary']

# %% ACCURACY
#------------------- Balanced Accuracy Across Repetitions -------------------#
accuracies = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    acc = balanced_accuracy_score(fold['true_label'], fold['predicted_label'])
    accuracies.append(acc)
    # save the balanced accuracy in the summary dataframe
    summary.loc[summary['Fold'] == fold_key, 'Balanced_Accuracy'] = acc

# print("Balanced accuracies across repetitions:")
# for i, acc in enumerate(accuracies):
#     print(f"Fold {i + 1}: {acc:.2f}")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\n")

print("Mean balanced accuracy and standard deviation across repetitions:")
print(f"Balanced accuracy:{mean_accuracy:.2f}")
print(f"Standard deviation:{std_accuracy:.2f}")

confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(accuracies)-1, loc=mean_accuracy, scale=st.sem(accuracies))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

#------------------------- Accuracy Per Class -------------------------#
class_accuracies = []

for fold in folds:
    fold = folds_dict[fold]
    fold_accuracies = []

    for class_ in classes:
        index = np.where(fold['true_label'] == class_)
        acc = accuracy_score(fold['true_label'].iloc[index], fold['predicted_label'].iloc[index])
        fold_accuracies.append(acc)

    class_accuracies.append(fold_accuracies)

print("\n")

for i, class_ in enumerate(classes):
    class_fold_accuracies = [fold_accuracies[i] for fold_accuracies in class_accuracies]
    mean_accuracy = np.mean(class_fold_accuracies)
    std_accuracy = np.std(class_fold_accuracies)
    confidence_level = 0.95
    ci_lower, ci_upper = st.t.interval(confidence_level, len(class_fold_accuracies)-1, loc=mean_accuracy, scale=st.sem(class_fold_accuracies))
    
    print(f"Class {class_}: Mean accuracy: {mean_accuracy:.2f}, Standard deviation: {std_accuracy:.2f}, "
          f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")
    
# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# %% MCC
#--------------------------------- MCC Across Repetitions---------------------------------#
mccs = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    mcc = matthews_corrcoef(fold['true_label'], fold['predicted_label'])
    mccs.append(mcc)
    # save the MCC in the summary dataframe
    summary.loc[summary['Fold'] == fold_key, 'MCC'] = mcc

# print("MCCs across repetitions:")
# for i, mcc in enumerate(mccs):
#     print(f"Fold {i + 1}: {mcc:.2f}")

mean_mcc= np.mean(mccs)
std_mcc = np.std(mccs)

print("\n")
    
print("Mean MCC and standard deviation across repetitions:")
print(f"Mean MCC: {mean_mcc:.2f}")
print(f"Standard deviation: {std_mcc:.2f}")
confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(mccs)-1, loc=mean_mcc, scale=st.sem(mccs))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# %% AUCROC
#--------------------------------- AUCROC Across Repetitions ---------------------------------#
print("\n")
aucs = []

def map_labels(label, target_class):
    return 1 if label == target_class else 0

for fold_key in folds:
    if fold_key in folds_dict:
        fold = folds_dict[fold_key]

        fold_probs = fold[['ASTR_LGG_prob', 'ASTR_HGG_prob', 'EP_prob', 'MED_prob', 'GANG_prob']].values
        auc_ = roc_auc_score(fold['Y'].values, fold_probs, average='weighted', multi_class='ovr')
        aucs.append(auc_)
        summary.loc[summary['Fold'] == fold_key, 'AUC'] = auc_


# print("AUCs across repetitions:")
# for i, auc in enumerate(aucs):
#     print(f"Fold {i + 1}: {auc:.2f}")

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

print("\n")
print("Mean AUC and standard deviation across repetitions:")
print(f"Mean AUC: {mean_auc:.2f}")
print(f"Standard deviation: {std_auc:.2f}")
confidence_level = 0.95

ci_lower, ci_upper = st.t.interval(confidence_level, len(aucs)-1, loc=mean_auc, scale=st.sem(aucs))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

#------------------------- AUCROC Per Class -------------------------#
auc_scores = {class_name: [] for class_name in [
    'ASTR_LGG',
    'ASTR_HGG',
    'EP',
    'MED',
    'GANG'
]}

for fold_name in folds:
    if fold_name in folds_dict:
        fold = folds_dict[fold_name]
        
        fold_probs = fold.iloc[:, 3:8]
        y_true = fold['Y']

        for idx, class_name in enumerate(auc_scores.keys()):
            auc_ = roc_auc_score(y_true == idx, fold_probs.iloc[:, idx], average='weighted', multi_class='ovr')
            auc_scores[class_name].append(auc_)

mean_auc_scores = {class_name: np.mean(scores) for class_name, scores in auc_scores.items()}
std_auc_scores = {class_name: np.std(scores) for class_name, scores in auc_scores.items()}

print("\n")
print("Mean AUCs and standard deviations per class:")
for class_name in auc_scores.keys():
    mean_auc = mean_auc_scores[class_name]
    std_auc = std_auc_scores[class_name]
    print(f"{class_name}: Mean AUC = {mean_auc:.2f}, Standard Deviation = {std_auc:.2f}")

# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# %% F1-SCORES
#----------------------------- Weighted F1-scores Across Repetitions -----------------------------#
f1s = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    f1 = f1_score(fold['true_label'], fold['predicted_label'], average='weighted')
    f1s.append(f1)
    # save the F1-score in the summary dataframe
    summary.loc[summary['Fold'] == fold_key, 'F1-Score'] = f1

# print("Weighted f1-scores across repetitions:")
# for i, f1 in enumerate(f1s):
#     print(f"Fold {i + 1}: {f1:.2f}")

mean_f1 = np.mean(f1s)
std_f1 = np.std(f1s)

print("\n")
    
print("Mean weighted f1-score and standard deviation across repetitions:")
print(f"Mean weighted f1-score: {mean_f1:.2f}")
print(f"Standard deviation: {std_f1:.2f}")
confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(f1s)-1, loc=mean_f1, scale=st.sem(f1s))
print(f"95% Confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}]")

#--------------------------------- F1-scores Per Class ---------------------------------#
f1_scores = {class_name: [] for class_name in classes}

for fold in folds:
    if fold in folds_dict:
        true_labels = folds_dict[fold]['true_label']
        pred_labels = folds_dict[fold]['predicted_label']

        for class_name in classes:
            true_binary = [1 if label == class_name else 0 for label in true_labels]
            pred_binary = [1 if label == class_name else 0 for label in pred_labels]
            precision = precision_score(true_binary, pred_binary, zero_division=0)
            recall = recall_score(true_binary, pred_binary, zero_division=0)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores[class_name].append(f1)

average_f1_scores = {class_name: np.mean(scores) for class_name, scores in f1_scores.items()}
std_f1_scores = {class_name: np.std(scores) for class_name, scores in f1_scores.items()}

print("Average F1-scores and standard deviations per class:")
for class_name in f1_scores.keys():
    avg_f1 = average_f1_scores[class_name]
    std_f1 = std_f1_scores[class_name]
    confidence_level = 0.95
    ci_lower, ci_upper = st.t.interval(confidence_level, len(f1_scores[class_name])-1, loc=avg_f1, scale=st.sem(f1_scores[class_name]))
    print(f"{class_name}: Average F1-score = {avg_f1:.2f}, Standard Deviation = {std_f1:.2f}, "
          f"95% Confidence Interval = [{ci_lower:.2f}, {ci_upper:.2f}]")
    
# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# %%