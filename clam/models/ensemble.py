# %% IMPORTS
import pandas as pd
import numpy as np 
import torch
import os

# %% LOAD CSV FILES
HE = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_class_bounded_small_clam_sb_uni_vit'
KI67 = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_ΚΙ67_class_bounded_small_clam_sb_uni_vit'
GFAP = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_GFAP_class_bounded_small_clam_sb_uni_vit'

# load HE folds
HE_contents = os.listdir(HE)

HE_folds_dict = {} 

for content in HE_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE + '/' + content)
        HE_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]

HE_fold_values = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in HE_folds_dict:
        HE_fold_values[f'fold_{i + 1}'] = HE_folds_dict[fold_key]

# load KI67 folds
KI67_contents = os.listdir(KI67)

KI67_folds_dict = {} 

for content in KI67_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(KI67 + '/' + content)
        KI67_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]
        
KI67_fold_values = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in KI67_folds_dict:
        KI67_fold_values[f'fold_{i + 1}'] = KI67_folds_dict[fold_key]

# load GFAP folds
GFAP_contents = os.listdir(GFAP)
GFAP_folds_dict = {} 

for content in GFAP_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(GFAP + '/' + content)
        GFAP_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]
        
GFAP_fold_values = {}

for i in range(50):
    fold_key = f'fold_{i}'
    if fold_key in GFAP_folds_dict:
        GFAP_fold_values[f'fold_{i + 1}'] = GFAP_folds_dict[fold_key]

classes = [
    'ASTR_LGG',
    'ASTR_HGG',
    'EP',
    'MED',
    'GANG'
]

# %%
probs_HE = {}
logits_HE = {}

for fold, df in HE_fold_values.items():
    probs_HE[fold] = df[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']]
    logits_HE[fold] = np.log(probs_HE[fold])

probs_KI67 = {}
logits_KI67 = {}

for fold, df in KI67_fold_values.items():
    probs_KI67[fold] = df[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']]
    logits_KI67[fold] = np.log(probs_KI67[fold])

probs_GFAP = {}
logits_GFAP = {}

for fold, df in GFAP_fold_values.items():
    probs_GFAP[fold] = df[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']]
    logits_GFAP[fold] = np.log(probs_GFAP[fold])

# %%
# ensemble probs of H&E and KI67
HE_KI67_ensemble_logits_values = {}

for fold in HE_fold_values.keys():
    HE_KI67_ensemble_logits_values[fold] = (logits_HE[fold] + logits_KI67[fold]) / 2

HE_KI67_ensemble_logits_numpy = {fold: HE_KI67_ensemble_logits_values[fold].to_numpy() for fold in HE_KI67_ensemble_logits_values}

HE_KI67_ensemble_softmax = {}
for fold, logits in HE_KI67_ensemble_logits_numpy.items():
    HE_KI67_ensemble_softmax[fold] = torch.nn.functional.softmax(torch.tensor(logits), dim=1)

# ensemble probs of H&E and GFAP
HE_GFAP_ensemble_logits_values = {}

for fold in HE_fold_values.keys():
    HE_GFAP_ensemble_logits_values[fold] = (logits_HE[fold] + logits_GFAP[fold]) / 2

HE_GFAP_ensemble_logits_numpy = {fold: HE_GFAP_ensemble_logits_values[fold].to_numpy() for fold in HE_GFAP_ensemble_logits_values}

HE_GFAP_ensemble_softmax = {}
for fold, logits in HE_GFAP_ensemble_logits_numpy.items():
    HE_GFAP_ensemble_softmax[fold] = torch.nn.functional.softmax(torch.tensor(logits), dim=1)

# ensemble probs of H&E, KI67 and GFAP
HE_KI67_GFAP_ensemble_logits_values = {}

for fold in HE_fold_values.keys():
    HE_KI67_GFAP_ensemble_logits_values[fold] = (logits_HE[fold] + logits_KI67[fold] + logits_GFAP[fold]) / 2

HE_KI67_GFAP_ensemble_logits_numpy = {fold: HE_KI67_GFAP_ensemble_logits_values[fold].to_numpy() for fold in HE_KI67_GFAP_ensemble_logits_values}

HE_KI67_GFAP_ensemble_softmax = {}
for fold, logits in HE_KI67_GFAP_ensemble_logits_numpy.items():
    HE_KI67_GFAP_ensemble_softmax[fold] = torch.nn.functional.softmax(torch.tensor(logits), dim=1)

# %% 
HE_KI67_ensemble_softmax_array = np.concatenate([HE_KI67_ensemble_softmax[fold].numpy() for fold in HE_KI67_ensemble_softmax], axis=0)
HE_KI67_ensemble_softmax_df = pd.DataFrame(HE_KI67_ensemble_softmax_array, columns=['p_0', 'p_1', 'p_2', 'p_3', 'p_4'])

HE_GFAP_ensemble_softmax_array = np.concatenate([HE_GFAP_ensemble_softmax[fold].numpy() for fold in HE_GFAP_ensemble_softmax], axis=0)
HE_GFAP_ensemble_softmax_df = pd.DataFrame(HE_GFAP_ensemble_softmax_array, columns=['p_0', 'p_1', 'p_2', 'p_3', 'p_4'])

HE_KI67_GFAP_ensemble_softmax_array = np.concatenate([HE_KI67_GFAP_ensemble_softmax[fold].numpy() for fold in HE_KI67_GFAP_ensemble_softmax], axis=0)
HE_KI67_GFAP_ensemble_softmax_df = pd.DataFrame(HE_KI67_GFAP_ensemble_softmax_array, columns=['p_0', 'p_1', 'p_2', 'p_3', 'p_4'])

# %%
HE_KI67_save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_ensemble_results/HE_KI67_ensemble_results"
HE_GFAP_save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_ensemble_results/HE_GFAP_ensemble_results"
HE_KI67_GFAP_save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_ensemble_results/HE_KI67_GFAP_ensemble_results"

os.makedirs(HE_KI67_save_path, exist_ok=True)
os.makedirs(HE_GFAP_save_path, exist_ok=True)
os.makedirs(HE_KI67_GFAP_save_path, exist_ok=True)

for fold in HE_fold_values.keys():
    HE_KI67_fold = HE_fold_values[fold][['slide_id', 'Y']] 
    HE_KI67_fold['Y_hat'] = ""  
    HE_KI67_fold = pd.concat([HE_KI67_fold, HE_KI67_ensemble_softmax_df], axis=1) 
    max_prob_indices = HE_KI67_ensemble_softmax_df.idxmax(axis=1)
    class_labels = max_prob_indices.apply(lambda x: int(x.split('_')[-1]))
    HE_KI67_fold['Y_hat'] = class_labels

    HE_KI67_fold.to_csv(f"{HE_KI67_save_path}/{fold}.csv", index=False)


for fold in HE_fold_values.keys():
    HE_GFAP_fold = HE_fold_values[fold][['slide_id', 'Y']] 
    HE_GFAP_fold['Y_hat'] = ""  
    HE_GFAP_fold = pd.concat([HE_GFAP_fold, HE_GFAP_ensemble_softmax_df], axis=1) 
    max_prob_indices = HE_GFAP_ensemble_softmax_df.idxmax(axis=1)
    class_labels = max_prob_indices.apply(lambda x: int(x.split('_')[-1]))
    HE_GFAP_fold['Y_hat'] = class_labels

    HE_KI67_fold.to_csv(f"{HE_GFAP_save_path}/{fold}.csv", index=False)


for fold in HE_fold_values.keys():
    HE_KI67_GFAP_fold = HE_fold_values[fold][['slide_id', 'Y']] 
    HE_KI67_GFAP_fold['Y_hat'] = ""  
    HE_KI67_GFAP_fold = pd.concat([HE_KI67_GFAP_fold, HE_KI67_GFAP_ensemble_softmax_df], axis=1) 
    max_prob_indices = HE_KI67_GFAP_ensemble_softmax_df.idxmax(axis=1)
    class_labels = max_prob_indices.apply(lambda x: int(x.split('_')[-1]))
    HE_KI67_GFAP_fold['Y_hat'] = class_labels

    HE_KI67_fold.to_csv(f"{HE_KI67_GFAP_save_path}/{fold}.csv", index=False)

# %%
