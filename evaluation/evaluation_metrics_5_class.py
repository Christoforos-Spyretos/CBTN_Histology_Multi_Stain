# %% IMPORTS
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, matthews_corrcoef, auc, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# %% LOAD RESULTS
# path to results
results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1'
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
            "p_0": "LGG_prob",
            "p_1": "HGG_prob",
            "p_2": "MB_prob",
            "p_3": "EP_prob",
            "p_4": "GG_prob"

            }, inplace=True)

        current_fold['true_label'] = current_fold['Y']

        current_fold.replace({'true_label': {
            0.0: 'LGG',
            1.0: 'HGG',
            2.0: 'MB',
            3.0: 'EP',
            4.0: 'GG'
            }}, inplace=True)

        current_fold['predicted_label'] = current_fold['Y_hat']

        current_fold.replace({'predicted_label': {
            0.0: 'LGG',
            1.0: 'HGG',
            2.0: 'MB',
            3.0: 'EP',
            4.0: 'GG'
            }}, inplace=True)

classes = ['LGG','HGG','MB','EP','GG']

fold_values = {}
 
for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in folds_dict:
        fold_values[f'fold_{i + 1}'] = folds_dict[fold_key]

# %% create a csv file with the summary of the results
summary = pd.DataFrame(columns=[
    'Task', 'Modality', 'Feature_Encoder', 'Aggregation',
    'Fusion', 'Fold','BA', 'MCC', 'AUC', 'F1-Score'
])

for fold in folds:
    new_row = pd.DataFrame({
        'Task': ['5_class'],
        'Modality': ['HE_KI67'],
        'Feature_Encoder': ['conch_v1'], 
        'Aggregation': ['small_clam_sb'],
        'Fusion': ['Late_Fusion_LM_AL'],
        'Fold': [str(fold)],
        'BA': [0],
        'MCC': [0],
        'AUC': [0],
        'F1-Score': [0],
    })
    summary = pd.concat([summary, new_row], ignore_index=True)

# save the summary dataframe to a csv file
save_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class'
save_name = 'EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv'
summary.to_csv(os.path.join(save_path, save_name), index=False)


# %% BALANCED ACCURACY
# calculate balanced accuracy for each fold
balanced_accuracies = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    balanced_accuracy = balanced_accuracy_score(fold['true_label'], fold['predicted_label'])
    balanced_accuracies.append(balanced_accuracy)
    # update the summary DataFrame with BA scores
    summary.loc[summary['Fold'] == fold_key, 'BA'] = balanced_accuracy

# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# print("Balanced accuracies across repetitions:")
# for i, balanced_accuracy in enumerate(balanced_accuracies):
#     print(f"Fold {i + 1}: {balanced_accuracy:.2f}")

mean_balanced_accuracy = np.mean(balanced_accuracies)
std_balanced_accuracy = np.std(balanced_accuracies)

print("\n")

print("Mean balanced accuracy and standard deviation across repetitions:")
print(f"Balanced accuracy:{mean_balanced_accuracy:.2f}")
print(f"Standard deviation:{std_balanced_accuracy:.2f}")

confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, 
                                   len(balanced_accuracies)-1, 
                                   loc=mean_balanced_accuracy, 
                                   scale=st.sem(balanced_accuracies))

print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

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

# %% MCC
mcc_scores = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    mcc = matthews_corrcoef(fold['true_label'], fold['predicted_label'])
    mcc_scores.append(mcc)
    # Update the summary DataFrame with MCC scores
    summary.loc[summary['Fold'] == fold_key, 'MCC'] = mcc

# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

# print("MCCs across repetitions:")
# for i, mcc in enumerate(mcc_scores):
#     print(f"Fold {i + 1}: {mcc:.2f}")

mean_mcc= np.mean(mcc_scores)
std_mcc = np.std(mcc_scores)

print("\n")
    
print("Mean MCC and standard deviation across repetitions:")
print(f"Mean MCC: {mean_mcc:.2f}")
print(f"Standard deviation: {std_mcc:.2f}")
confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(mcc_scores)-1, loc=mean_mcc, scale=st.sem(mcc_scores))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

# %% AUC
print("\n")
aucs = []

def map_labels(label, target_class):
    return 1 if label == target_class else 0

for fold_key in folds:
    if fold_key in folds_dict:
        fold = folds_dict[fold_key]

        fold_probs = fold[['LGG_prob', 'HGG_prob', 'MB_prob', 'EP_prob', 'GG_prob']].values
        auc_ = roc_auc_score(fold['Y'].values, fold_probs, average='weighted', multi_class='ovr')
        aucs.append(auc_)
        summary.loc[summary['Fold'] == fold_key, 'AUC'] = auc_

# save summary
summary.to_csv(os.path.join(save_path, save_name), index=False)

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


# AUC for each class (one-vs-rest) - FIXED
auc_scores = {class_name: [] for class_name in [
    'LGG',
    'HGG',
    'MB',
    'EP',
    'GG'
]}

class_names = list(auc_scores.keys())

for fold_name in folds:
    if fold_name in folds_dict:
        fold = folds_dict[fold_name]
        y_true = fold['Y'].values
        fold_probs = fold[['LGG_prob', 'HGG_prob', 'MB_prob', 'EP_prob', 'GG_prob']].values

        for idx, class_name in enumerate(class_names):
            y_true_binary = (y_true == idx).astype(int)
            y_score = fold_probs[:, idx]
            try:
                auc_ = roc_auc_score(y_true_binary, y_score)
            except ValueError:
                auc_ = np.nan  # In case only one class present in y_true_binary
            auc_scores[class_name].append(auc_)

mean_auc_scores = {class_name: np.nanmean(scores) for class_name, scores in auc_scores.items()}
std_auc_scores = {class_name: np.nanstd(scores) for class_name, scores in auc_scores.items()}

print("\n")
print("Mean AUCs and standard deviations per class:")
for class_name in auc_scores.keys():
    mean_auc = mean_auc_scores[class_name]
    std_auc = std_auc_scores[class_name]
    print(f"{class_name}: Mean AUC = {mean_auc:.2f}, Standard Deviation = {std_auc:.2f}")
    
# %% WEIGHTED F1-SCORE
f1_scores = []
for fold_key in folds:
    fold = folds_dict[fold_key]
    f1 = f1_score(fold['true_label'], fold['predicted_label'], average='weighted')
    f1_scores.append(f1)
    # update summary with F1 scores
    summary.loc[summary['Fold'] == fold_key, 'F1-Score'] = f1

summary.to_csv(os.path.join(save_path, save_name), index=False)

# print("F1-scores across repetitions:")
# for i, f1 in enumerate(f1_scores):
#     print(f"Fold {i + 1}: {f1:.2f}")

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("\n")

print("Mean F1-score and standard deviation across repetitions:")
print(f"Mean F1-score: {mean_f1:.2f}")
print(f"Standard deviation: {std_f1:.2f}")
confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(f1_scores)-1, loc=mean_f1, scale=st.sem(f1_scores))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

# weighted F1-score for each class
class_f1_scores = []
for fold_key in folds:
    fold = folds_dict[fold_key]
    class_f1_fold = []
    
    for class_ in classes:
        index = np.where(fold['true_label'] == class_)
        f1 = f1_score(fold['true_label'].iloc[index], fold['predicted_label'].iloc[index], average='weighted')
        class_f1_fold.append(f1)

    class_f1_scores.append(class_f1_fold)

print("\n")
for i, class_ in enumerate(classes):
    class_f1_fold_scores = [fold_f1[i] for fold_f1 in class_f1_scores]
    mean_class_f1 = np.mean(class_f1_fold_scores)
    std_class_f1 = np.std(class_f1_fold_scores)

    print(f"Class {class_}: Mean F1-score: {mean_class_f1:.2f}, Standard deviation: {std_class_f1:.2f}, ")

# %% MEAN CONFUSION MATRIX
confusion_matrices = []
for fold_key in folds:
    fold = folds_dict[fold_key]
    cm = confusion_matrix(fold['true_label'], fold['predicted_label'], labels=classes)
    confusion_matrices.append(cm)

mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

cm_df = pd.DataFrame(mean_confusion_matrix, 
                     index=[f"True {class_}" for class_ in classes],
                     columns=[f"Pred {class_}" for class_ in classes])

# plot the mean confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, values_format='.1f')
plt.title('Mean Confusion Matrix')
plt.show()

# %%