# %% IMPORTS
import pandas as pd
import os
from collections import Counter

# %% LOAD RESULTS
# path to results
Merged_HE_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_conch_v1'
Merged_KI67_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_conch_v1'

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

    HE_Y_hat = HE_fold['Y_hat']
    HE_probs = HE_fold[['p_0', 'p_1']].values

    KI67_Y_hat = KI67_fold['Y_hat']
    KI67_probs = KI67_fold[['p_0', 'p_1']].values

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
        'p_1': [p[1] for p in p_new]
    })
    
    # ensure directory exists
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Late_Fusion_MJ_HE_KI67_small_clam_sb_conch_v1/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_KI67_fold.to_csv(save_path, index=False)

# %%