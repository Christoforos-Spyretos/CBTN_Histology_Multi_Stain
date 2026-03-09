# %% IMPORTS
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay, matthews_corrcoef, auc, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

# %% LOAD RESULTS
# path to results
results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_EWM_HE_KI67_small_clam_sb_conch_v1_5'
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
            "p_1": "HGG_prob"
            }, inplace=True)

        current_fold['true_label'] = current_fold['Y']

        current_fold.replace({'true_label': {
            0.0: 'LGG',
            1.0: 'HGG',
            }}, inplace=True)
        
        current_fold['predicted_label'] = current_fold['Y_hat']

        current_fold.replace({'predicted_label': {
            0.0: 'LGG',
            1.0: 'HGG',
            }}, inplace=True)

classes = ['LGG','HGG']

fold_values = {}
 
for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in folds_dict:
        fold_values[f'fold_{i + 1}'] = folds_dict[fold_key]

# %% create a csv file with the summary of the results
summary = pd.DataFrame(columns=[
    'Task', 'Modality', 'Feature_Encoder', 'Aggregation',
    'Fusion', 'Fold', 'BA', 'MCC', 'AUC', 'F1-Score'
])

for fold in folds:
    new_row = pd.DataFrame({
        'Task': ['LGG_vs_HGG'],
        'Modality': ['HE_KI67'],
        'Feature_Encoder': ['conch_v1_5'], 
        'Aggregation': ['small_clam_sb'],
        'Fusion': ['Intermediate_Fusion_EWM'],
        'Fold': [str(fold)],
        'BA': [0],
        'MCC': [0],
        'AUC': [0],
        'F1-Score': [0],
    })
    summary = pd.concat([summary, new_row], ignore_index=True)

# save the summary dataframe to a csv file
save_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/50%_split/LGG_vs_HGG'
os.makedirs(save_path, exist_ok=True)
save_name = 'EVAL_LGG_vs_HGG_Intermediate_Fusion_EWM_HE_KI67_small_clam_sb_conch_v1_5.csv'

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

print("Balanced accuracies across repetitions:")
for i, balanced_accuracy in enumerate(balanced_accuracies):
    print(f"Fold {i + 1}: {balanced_accuracy:.2f}")

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

print("MCCs across repetitions:")
for i, mcc in enumerate(mcc_scores):
    print(f"Fold {i + 1}: {mcc:.2f}")

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
auc_scores = []

for fold_key in folds:
    fold = folds_dict[fold_key]
    # For binary AUC, use the probability of the positive class (HGG)
    y_true = fold['true_label'].map({'LGG': 0, 'HGG': 1}).values
    y_probs = fold['HGG_prob'].values
    auc = roc_auc_score(y_true, y_probs)
    auc_scores.append(auc)
    # update summary with AUC scores
    summary.loc[summary['Fold'] == fold_key, 'AUC'] = auc

summary.to_csv(os.path.join(save_path, save_name), index=False)

print("AUCs across repetitions:")
for i, auc in enumerate(auc_scores):
    print(f"Fold {i + 1}: {auc:.2f}")

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

print("\n")

print("Mean AUC (macro-averaged) and standard deviation across repetitions:")
print(f"Mean AUC: {mean_auc:.2f}")
print(f"Standard deviation: {std_auc:.2f}")
confidence_level = 0.95
ci_lower, ci_upper = st.t.interval(confidence_level, len(auc_scores)-1, loc=mean_auc, scale=st.sem(auc_scores))
print(f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")

# AUC for each class (one-vs-rest)
class_auc_scores = []
for fold_key in folds:
    fold = folds_dict[fold_key]
    class_auc_fold = []
    
    for i, class_ in enumerate(classes):
        # Create binary labels (current class vs all others)
        y_true_binary = (fold['true_label'] == class_).astype(int)
        y_prob_class = fold[f'{class_}_prob']
        
        # Only calculate AUC if both classes are present
        if len(np.unique(y_true_binary)) > 1:
            auc = roc_auc_score(y_true_binary, y_prob_class)
        else:
            auc = np.nan  # Cannot calculate AUC with only one class
        class_auc_fold.append(auc)

    class_auc_scores.append(class_auc_fold)

print("\n")
for i, class_ in enumerate(classes):
    class_auc_fold_scores = [fold_auc[i] for fold_auc in class_auc_scores]
    # Filter out NaN values for calculation
    valid_scores = [score for score in class_auc_fold_scores if not np.isnan(score)]
    
    if len(valid_scores) > 0:
        mean_class_auc = np.mean(valid_scores)
        std_class_auc = np.std(valid_scores)
        confidence_level = 0.95
        ci_lower, ci_upper = st.t.interval(confidence_level, len(valid_scores)-1, loc=mean_class_auc, scale=st.sem(valid_scores))
        
        print(f"Class {class_} (One-vs-Rest): Mean AUC: {mean_class_auc:.2f}, Standard deviation: {std_class_auc:.2f}, "
              f"95% Confidence interval:[{ci_lower:.2f}, {ci_upper:.2f}]")
    else:
        print(f"Class {class_}: No valid AUC scores (class may not appear in some folds)")
    
# %% WEIGHTED F1-SCORE
f1_scores = []
for fold_key in folds:
    fold = folds_dict[fold_key]
    f1 = f1_score(fold['true_label'], fold['predicted_label'], average='weighted')
    f1_scores.append(f1)
    # update summary with F1 scores
    summary.loc[summary['Fold'] == fold_key, 'F1-Score'] = f1

summary.to_csv(os.path.join(save_path, save_name), index=False)

print("F1-scores across repetitions:")
for i, f1 in enumerate(f1_scores):
    print(f"Fold {i + 1}: {f1:.2f}")

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

# Replace class names for display with 'glioma' beneath, centered for y-axis
# display_names_x = ['Low-grade\nglioma' if c == 'LGG' else 'High-grade\nglioma' for c in classes]
display_names_x = ['LGG' if c == 'LGG' else 'HGG' for c in classes]

# display_names_y = ['Low-grade\n\u2003\u2003\u2003glioma' if c == 'LGG' else 'High-grade\n\u2003\u2003\u2003glioma' for c in classes]
display_names_y = ['LGG' if c == 'LGG' else 'HGG' for c in classes]
cm_df = pd.DataFrame(mean_confusion_matrix, index=[f"True {name}" for name in display_names_y], columns=[f"Pred {name}" for name in display_names_x])

# plot the mean confusion matrix
plt.figure(figsize=(10, 8))
# plot confusion matrix and manually set number font size
disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=display_names_x)
ax = disp.plot(cmap=plt.cm.Blues, values_format='.1f', colorbar=False)
# Increase font size of numbers in the matrix
for text in ax.ax_.texts:
    text.set_fontsize(14)
# Center the column labels
plt.setp(ax.ax_.xaxis.get_majorticklabels(), ha='center', va='center', fontsize=14)
ax.ax_.tick_params(axis='x', pad=10)
ax.ax_.tick_params(axis='y', pad=20)
# Align the row labels (y-axis) to the center horizontally and vertically
plt.setp(ax.ax_.yaxis.get_majorticklabels(), ha='center', va='center', fontsize=14)

# for HE
# ax.ax_.set_xlabel('Predicted label', labelpad=30, fontsize=14, color = 'white')
# ax.ax_.set_ylabel('True label', labelpad=30, fontsize=14)

# for KI67
# ax.ax_.set_yticklabels([])
# ax.ax_.set_yticks([])
# ax.ax_.set_xlabel('Predicted label', labelpad=30, fontsize=14)
# ax.ax_.set_ylabel('', labelpad=30, fontsize=14)

# for fusion
ax.ax_.set_yticklabels([])
ax.ax_.set_yticks([])
ax.ax_.set_xlabel('Predicted label', labelpad=30, fontsize=14, color = 'white')
ax.ax_.set_ylabel('', labelpad=30, fontsize=14)
# plt.title('Mean Confusion Matrix')
plt.savefig(os.path.join('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG', 'LGG_vs_HGG_HE_KI67_IM_CONCAT_CM.png'), bbox_inches='tight', dpi=300)
plt.show()

# %%