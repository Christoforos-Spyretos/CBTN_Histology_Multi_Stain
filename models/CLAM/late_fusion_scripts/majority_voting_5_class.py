# %% IMPORTS
import pandas as pd
import os
from collections import Counter

# %% LOAD RESULTS
# path to results
Merged_HE_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Single_Stain/small_clam_sb/conch/EVAL_5_class_Merged_HE_small_clam_sb_conch'
Merged_KI67_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Single_Stain/small_clam_sb/conch/EVAL_5_class_Merged_KI67_small_clam_sb_conch'
Merged_GFAP_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Single_Stain/small_clam_sb/conch/EVAL_5_class_Merged_GFAP_small_clam_sb_conch'

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

    HE_Y_hat = HE_fold['Y_hat']
    HE_probs = HE_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    KI67_Y_hat = KI67_fold['Y_hat']
    KI67_probs = KI67_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    slide_id = HE_fold['slide_id']
    Y = HE_fold['Y']
    Y_hat_new = []
    p_new = []

    for i in range(len(Y)):
        votes = [HE_Y_hat[i], KI67_Y_hat[i]]
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]

        if vote_counts[majority_vote] == 1:  # tie
            probs = [HE_probs[i], KI67_probs[i]]
            max_prob_idx = max(range(len(probs)), key=lambda idx: max(probs[idx]))
            Y_hat_new.append(votes[max_prob_idx])
            p_new.append(probs[max_prob_idx])
        else:
            Y_hat_new.append(majority_vote)
            if majority_vote == HE_Y_hat[i]:
                p_new.append(HE_probs[i])
            else:
                p_new.append(KI67_probs[i])

    Merged_HE_KI67_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat_new, 
        'p_0': [p[0] for p in p_new], 
        'p_1': [p[1] for p in p_new],
        'p_2': [p[2] for p in p_new],
        'p_3': [p[3] for p in p_new],
        'p_4': [p[4] for p in p_new]
    })
    
    # ensure directory exists
    save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Late_Fusion/majority_voting/small_clam_sb/conch/EVAL_5_class_Merged_HE_KI67_small_clam_sb_conch'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{fold}.csv'
    
    # save results
    Merged_HE_KI67_fold.to_csv(save_path)

# %% MERGED_HE_GFAP
for fold in folds:
    HE_fold = HE_folds_dict[fold]   
    GFAP_fold = GFAP_folds_dict[fold]

    HE_Y_hat = HE_fold['Y_hat']
    HE_probs = HE_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    GFAP_Y_hat = GFAP_fold['Y_hat']
    GFAP_probs = GFAP_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    slide_id = HE_fold['slide_id']
    Y = HE_fold['Y']
    Y_hat_new = []
    p_new = []

    for i in range(len(Y)):
        votes = [HE_Y_hat[i], GFAP_Y_hat[i]]
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]

        if vote_counts[majority_vote] == 1:  # tie
            probs = [HE_probs[i], GFAP_probs[i]]
            max_prob_idx = max(range(len(probs)), key=lambda idx: max(probs[idx]))
            Y_hat_new.append(votes[max_prob_idx])
            p_new.append(probs[max_prob_idx])
        else:
            Y_hat_new.append(majority_vote)
            if majority_vote == HE_Y_hat[i]:
                p_new.append(HE_probs[i])
            else:
                p_new.append(GFAP_probs[i])

    Merged_HE_GFAP_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat_new, 
        'p_0': [p[0] for p in p_new], 
        'p_1': [p[1] for p in p_new],
        'p_2': [p[2] for p in p_new],
        'p_3': [p[3] for p in p_new],
        'p_4': [p[4] for p in p_new]
    })
    
    # ensure directory exists
    save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Late_Fusion/majority_voting/small_clam_sb/conch/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_conch'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{fold}.csv'
    
    # save results
    Merged_HE_GFAP_fold.to_csv(save_path)

# %% MERGED_HE_KI67_GFAP
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    GFAP_fold = GFAP_folds_dict[fold]

    HE_Y_hat = HE_fold['Y_hat']
    HE_probs = HE_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    KI67_Y_hat = KI67_fold['Y_hat']
    KI67_probs = KI67_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    GFAP_Y_hat = GFAP_fold['Y_hat']
    GFAP_probs = GFAP_fold[['p_0', 'p_1', 'p_2', 'p_3', 'p_4']].values

    slide_id = HE_fold['slide_id']
    Y = HE_fold['Y']
    Y_hat_new = []
    p_new = []

    for i in range(len(Y)):
        votes = [HE_Y_hat[i], KI67_Y_hat[i], GFAP_Y_hat[i]]
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]

        if vote_counts[majority_vote] == 1:  # tie
            probs = [HE_probs[i], KI67_probs[i], GFAP_probs[i]]
            max_prob_idx = max(range(len(probs)), key=lambda idx: max(probs[idx]))
            Y_hat_new.append(votes[max_prob_idx])
            p_new.append(probs[max_prob_idx])
        else:
            Y_hat_new.append(majority_vote)
            if majority_vote == HE_Y_hat[i]:
                p_new.append(HE_probs[i])
            elif majority_vote == KI67_Y_hat[i]:
                p_new.append(KI67_probs[i])
            else:
                p_new.append(GFAP_probs[i])

    Merged_HE_KI67_GFAP_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat_new, 
        'p_0': [p[0] for p in p_new], 
        'p_1': [p[1] for p in p_new],
        'p_2': [p[2] for p in p_new],
        'p_3': [p[3] for p in p_new],
        'p_4': [p[4] for p in p_new]
    })
    
    # ensure directory exists
    save_dir = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/Experiments_Evaluation/5_class/Late_Fusion/majority_voting/small_clam_sb/conch/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{fold}.csv'
    
    # save results
    Merged_HE_KI67_GFAP_fold.to_csv(save_path)

# %%