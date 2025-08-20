# %% IMPORTS
import pandas as pd
import os
import torch
import torch.nn.functional as F

# %% LOAD RESULTS
# path to results
Merged_HE_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_conch_v1'
Merged_KI67_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_conch_v1'

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

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED_HE_KI67
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    logits_0 = torch.tensor((HE_fold['logits_0'] + KI67_fold['logits_0']) / 2)
    logits_1 = torch.tensor((HE_fold['logits_1'] + KI67_fold['logits_1']) / 2)
    logits_2 = torch.tensor((HE_fold['logits_2'] + KI67_fold['logits_2']) / 2)
    logits_3 = torch.tensor((HE_fold['logits_3'] + KI67_fold['logits_3']) / 2)
    logits_4 = torch.tensor((HE_fold['logits_4'] + KI67_fold['logits_4']) / 2)
    logits = torch.stack([torch.tensor(logits_0), torch.tensor(logits_1), torch.tensor(logits_2), torch.tensor(logits_3), torch.tensor(logits_4)], dim=0)
    probs = F.softmax(logits, dim=0)  
    Y_hat = torch.argmax(probs, dim=0) 
    Y_hat = Y_hat.cpu().numpy()
    p_0 = probs[0].cpu().numpy()
    p_1 = probs[1].cpu().numpy()
    p_2 = probs[2].cpu().numpy()
    p_3 = probs[3].cpu().numpy()
    p_4 = probs[4].cpu().numpy()
    Y = HE_fold['Y']
    Merged_HE_KI67_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1,
        'p_2': p_2,
        'p_3': p_3,
        'p_4': p_4})
    # Define save path
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_KI67_fold.to_csv(save_path, index=False)
    
# %%


