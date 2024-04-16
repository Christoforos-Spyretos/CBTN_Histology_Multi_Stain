# %% IMPORTS
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef, auc, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# %% LOAD RESULTS
# path to results
results_path = 'models/CLAM/eval_results/EVAL_KI67_6_class_merged_attempt_1'

# load folds
contents = os.listdir(results_path)

folds_dict = {} 

for content in contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(results_path + '/' + content)
        folds_dict[name] = df

# %% 
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 
         'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9']       

for fold in folds:
    if fold in folds_dict:
        current_fold = folds_dict[fold]
        current_fold.rename(columns={
            "Y": "true_label", 
            "Y_hat": "predicted_label",
            "p_0": "ASTR_LGG_prob",
            "p_1": "ASTR_HGG_prob",
            "p_2": "MED_prob",
            "p_3": "EP_prob",
            "p_4": "ATRT_prob",
            "p_5": "DIPG_prob"
            }, inplace=True)

        current_fold['true_label'].replace({
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'MED',
            3.0: 'EP',
            4.0: 'ATRT',
            5.0: 'DIPG'
            }, inplace=True)
        
        current_fold['predicted_label'].replace({
            0.0: 'ASTR_LGG',
            1.0: 'ASTR_HGG',
            2.0: 'MED',
            3.0: 'EP',
            4.0: 'ATRT',
            5.0: 'DIPG'
            }, inplace=True)
        
for fold in folds:
    if fold in folds_dict:
        fold_1 = folds_dict['fold_0']
        fold_2 = folds_dict['fold_1']
        fold_3 = folds_dict['fold_2']
        fold_4 = folds_dict['fold_3']
        fold_5 = folds_dict['fold_4']
        fold_6 = folds_dict['fold_5']
        fold_7 = folds_dict['fold_6']
        fold_8 = folds_dict['fold_7']
        fold_9 = folds_dict['fold_8']
        fold_10 = folds_dict['fold_9']

summary = folds_dict['summary']

classes = [
    'ASTR_LGG',
    'ASTR_HGG',
    'MED',
    'EP',
    'ATRT',
    'DIPG'
]

# %% MEAN ACCURACY AND MCC ACROSS FOLDS
#--------------------------------- Accuracy ---------------------------------#
accuracies = []

for fold in folds:
    fold = folds_dict[fold]
    acc = accuracy_score(fold['true_label'], fold['predicted_label'])
    accuracies.append(acc)

print("Accuracies across folds:")
for i, acc in enumerate(accuracies):
    print(f"Fold {i + 1}: {acc:.4f}")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\n")

print("Mean accuracy and standard deviation across folds:")
print(f"Accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation: {std_accuracy:.4f}")

#------------------------- Mean Accuracy Per Class -------------------------#
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
    
    print(f"Class {class_}:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print("\n")

#--------------------------------- MCC ---------------------------------#
mccs = []

for fold in folds:
    fold = folds_dict[fold]
    mcc = matthews_corrcoef(fold['true_label'], fold['predicted_label'])
    mccs.append(mcc)

print("MCCs across folds:")
for i, mcc in enumerate(mccs):
    print(f"Fold {i + 1}: {mcc:.4f}")

mean_mcc= np.mean(mccs)
std_mcc = np.std(mccs)

print("\n")
    
print("Mean MCC and standard deviation across folds:")
print(f"Mean MCC: {mean_mcc:.4f}")
print(f"Standard deviation: {std_mcc:.4f}")

#--------------------------------- Mean AUC ---------------------------------#
print("\n")
aucs = []
print("AUCs across folds:")
for index, row in summary.iterrows():
    auc_test= row['test_auc']
    aucs.append(auc_test)
    print(f"Fold {index + 1}:{auc_test:.4f}")

print("\n")

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
print("Mean AUC and standard deviation across folds:")
print(f"Mean AUC: {mean_auc:.4f}")
print(f"Standard deviation: {std_auc:.4f}")

# %% CLASSIFICATION REPORT
i = 0
for fold in folds:
    i += 1
    fold = folds_dict[fold]

    fold_report = classification_report(
        fold['true_label'], 
        fold['predicted_label'], 
        target_names=[
            'ASTR_LGG',
            'ASTR_HGG',
            'MED',
            'EP',
            'ATRT',
            'DIPG'
            ])
    print(f"Classification report of fold {i}")
    print(fold_report)
    print("\n")

# %%
i = 0
for fold in folds:
    i += 1
    fold = folds_dict[fold]

    fold_cm = confusion_matrix(
        fold['true_label'],
        fold['predicted_label'],
        labels=[
            'ASTR_LGG',
            'ASTR_HGG',
            'MED',
            'EP',
            'ATRT',
            'DIPG'
            ])

    fold_cmd = ConfusionMatrixDisplay(
        confusion_matrix=fold_cm, 
        display_labels=[
            'ASTR_LGG',
            'ASTR_HGG',
            'MED',
            'EP',
            'ATRT',
            'DIPG'
            ])

    plt.figure(figsize=(10, 8))
    fold_cmd.plot(cmap='Blues')
    plt.xticks(rotation=45)
    fold_cmd.ax_.set(xlabel='Predicted', ylabel='True')
    plt.title(f'Confusion Matrix of fold {i}')
    plt.show()

# %% ROC CURVE PER FOLD
# class_labels = [
#     'ASTR_LGG',
#     'ASTR_HGG',
#     'MED',
#     'EP',
#     # 'ATRT',
#     # 'DIPG'
# ]

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# plt.figure(figsize=(10, 8))

# i = 0
# for fold_ in folds:
#     i += 1
#     fold = folds_dict[fold_]

#     fold_probs = fold.iloc[:, 3:9] # for 4 classes
#     fold_probs = fold.iloc[:, 3:11] # for 6 classes
#     column_names = list(fold_probs.columns)

#     for class_label in class_labels:
#         for column_name in column_names:
#             fpr[fold_], tpr[fold_], _ = roc_curve(fold['true_label'], fold_probs[column_name], pos_label=class_label)
#             roc_auc[fold_] = auc(fpr[fold_], tpr[fold_])
#             # roc_auc[fold_] = roc_auc_score(fold['true_label'] == class_label, fold_probs[column_name])
#             roc_auc[fold_] = roc_auc_score(fold['true_label'] == class_label, fold_probs[column_name], multi_class='over')
            
#     plt.plot(fpr[fold_], tpr[fold_], label=f'Fold {i} (AUC = {roc_auc[fold_]:.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=1.0, color='darkred', label='Random Classifier')  
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True) 
# plt.title("ROC Curve per Fold")
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='best')
# plt.show()

# %% ROC CURVE PER CLASS 
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# plt.figure(figsize=(10, 8))

# all_folds = pd.concat([folds_dict[fold_] for fold_ in folds], ignore_index=True)

# fold_probs = all_folds.iloc[:, 3:9] # for 4 classes
# fold_probs = all_folds.iloc[:, 3:11] # for 6 classes
# column_names = list(fold_probs.columns)

# for class_label in class_labels:
#     for column_name in column_names:
#         fpr[class_label], tpr[class_label], _ = roc_curve(all_folds['true_label'], all_folds[column_name], pos_label=class_label)
#         roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
    
#     plt.plot(fpr[class_label], tpr[class_label], label=f'{class_label} (AUC = {roc_auc[class_label]:.2f})')

# plt.plot([0, 1], [0, 1], 'k--', lw=1.0, color='darkred', label='Random Classifier')  
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True) 
# plt.title("ROC Curve per Class")
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='best')
# plt.show()

# %% PRECISION-RECALL CURVE PER FOLD
# class_labels = [
#     'ASTR_LGG',
#     'ASTR_HGG',
#     'MED',
#     'EP',
#     # 'ATRT',
#     # 'DIPG'
# ]

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# plt.figure(figsize=(10, 8))

# i = 0
# for fold_ in folds:
#     i += 1
#     fold = folds_dict[fold_]

#     fold_probs = fold.iloc[:, 3:9] # for 4 classes
#     fold_probs = fold.iloc[:, 3:11] # for 6 classes
#     column_names = list(fold_probs.columns)

#     for class_label in class_labels:
#         for column_name in column_names:
#             precision[fold_], recall[fold_], _ = precision_recall_curve(fold['true_label'], fold_probs[column_name], pos_label=class_label)
            
#     plt.plot(recall[fold_], precision[fold_], label=f'Fold {i}')

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True) 
# plt.title('Precision vs Recall Curve per Fold')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='best')
# plt.show()

# %% PRECISION-RECALL CURVE PER CLASS
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# plt.figure(figsize=(10, 8))

# all_folds = pd.concat([folds_dict[fold_] for fold_ in folds], ignore_index=True)
    
# fold_probs = fold.iloc[:, 3:9] # for 4 classes
# fold_probs = fold.iloc[:, 3:11] # for 6 classes
# column_names = list(fold_probs.columns)

# for class_label in class_labels:
#     for column_name in column_names:
#         precision[class_label], recall[class_label], _ = precision_recall_curve(all_folds['true_label'], all_folds[column_name], pos_label=class_label)
    
#     plt.plot(recall[class_label], precision[class_label], label=f'{class_label}')

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.grid(True) 
# plt.title("Precision vs Recall Curve per Class")
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='best')
# plt.show()

# %%
