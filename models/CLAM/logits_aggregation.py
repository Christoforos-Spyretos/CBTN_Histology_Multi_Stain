# %% IMPORTS
import pandas as pd
import os
import torch
import torch.nn.functional as F

# %% LOAD RESULTS

# path to results
Merged_HE_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_conch'
Merged_KI67_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_conch'
Merged_GFAP_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_GFAP_small_clam_sb_conch'

HE_contents = os.listdir(Merged_HE_path)
HE_folds_dict = {} 

for content in HE_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_HE_path + '/' + content)
        HE_folds_dict[name] = df

KI67_contents = os.listdir(Merged_KI67_path)
KI67_folds_dict = {}

for content in KI67_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_KI67_path + '/' + content)
        KI67_folds_dict[name] = df  

GFAP_contents = os.listdir(Merged_GFAP_path)
GFAP_folds_dict = {}

for content in GFAP_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_GFAP_path + '/' + content)
        GFAP_folds_dict[name] = df

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED_HE_KI67
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    logits_0 = torch.tensor((HE_fold['logits_0'] + KI67_fold['logits_0']) / 2)
    logits_1 = torch.tensor((HE_fold['logits_1'] + KI67_fold['logits_1']) / 2)
    logits = torch.stack([torch.tensor(logits_0), torch.tensor(logits_1)], dim=0)  # Shape: [2, 71]
    probs = F.softmax(logits, dim=0)  
    Y_hat = torch.argmax(probs, dim=0) 
    Y_hat = Y_hat.cpu().numpy()
    p_0 = probs[0].cpu().numpy()
    p_1 = probs[1].cpu().numpy()
    Y = HE_fold['Y']
    Merged_HE_KI67_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1})
    # Define save path
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/LGG_vs_HGG/Late_Fusion/logits_aggregation/small_clam_sb/conch/EVAL_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_KI67_fold.to_csv(save_path, index=False)


# %% MERGED_HE_GFAP
for fold in folds:
    HE_fold = HE_folds_dict[fold]   
    GFAP_fold = GFAP_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    # convert logits to tensors
    logits_0 = torch.tensor((HE_fold['logits_0'] + KI67_fold['logits_0']) / 2)
    logits_1 = torch.tensor((HE_fold['logits_1'] + KI67_fold['logits_1']) / 2)
    logits = torch.stack([torch.tensor(logits_0), torch.tensor(logits_1)], dim=0)  # Shape: [2, 71]
    probs = F.softmax(logits, dim=0)
    Y_hat = torch.argmax(probs, dim=0)
    Y_hat = Y_hat.cpu().numpy()
    p_0 = probs[0].cpu().numpy()
    p_1 = probs[1].cpu().numpy()
    Y = HE_fold['Y']
    Merged_HE_GFAP_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1})
    # Define save path
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/LGG_vs_HGG/Late_Fusion/logits_aggregation/small_clam_sb/conch/EVAL_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_GFAP_fold.to_csv(save_path, index=False)

# %% MERGED_HE_KI67_GFAP
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    GFAP_fold = GFAP_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    # convert logits to tensors
    logits_0 = torch.tensor((HE_fold['logits_0'] + KI67_fold['logits_0'] + GFAP_fold['logits_0']) / 3)
    logits_1 = torch.tensor((HE_fold['logits_1'] + KI67_fold['logits_1'] + GFAP_fold['logits_1']) / 3)
    logits = torch.stack([torch.tensor(logits_0), torch.tensor(logits_1)], dim=0)
    probs = F.softmax(logits, dim=0)
    Y_hat = torch.argmax(probs, dim=0)
    Y_hat = Y_hat.cpu().numpy()
    p_0 = probs[0].cpu().numpy()
    p_1 = probs[1].cpu().numpy()
    Y = HE_fold['Y']
    Merged_HE_KI67_GFAP_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1})
    # save results
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/LGG_vs_HGG/Late_Fusion/logits_aggregation/small_clam_sb/conch/EVAL_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_KI67_GFAP_fold.to_csv(save_path, index=False)   
# %%


