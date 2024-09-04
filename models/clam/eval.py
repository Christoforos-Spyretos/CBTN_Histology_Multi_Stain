from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--n_classes', type=str, default=6,
                    help='number of classes')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'abmil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', 
                    type=str, 
                    choices=[
                        'KI67_7_class_tumor_subtyping',
                        'HE_7_class_tumor_subtyping_bounded_to_KI67',
                        'Merged_KI67_7_class_tumor_subtyping',
                        'Merged_HE_7_class_tumor_subtyping_bounded_to_KI67',
                        'KI67_7_class_tumor_subtyping_bounded_to_HE',
                        'Merged_KI67_7_class_tumor_subtyping_bounded_to_HE',
                        'Merged_Histology_7_class_bounded_tumor_subtyping',
                        'HE_7_class_tumor_subtyping',
                        'HE_10_class_tumor_subtyping',
                        'Merged_HE_7_class_tumor_subtyping',
                        'Merged_Histology_7_class_tumor_subtyping',
                        'Merged_HE_10_class_tumor_subtyping',
                        'Merged_Histology_10_class_tumor_subtyping',
                        'Histology_5_class_bounded',
                        'Histology_3_class_bounded',
                        'Histology_5_class_HE_KI67_bounded',
                        'Histology_7_class_HE_KI67_bounded',
                        'HE_KI67_LGG_vs_HGG',
                        'LGG_vs_HGG'
                        ])
parser.add_argument('--feature_type', 
                    type=str, 
                    choices=[
                        'resnet50_features', 
                        'resnet50_features_7_class',
                        'hipt_vit_features',
                        'hipt_vit_features_7_class',
                        'uni_vit_features',
                        'uni_vit_features_7_class',
                        'conch_vit_features',
                        'conch_vit_features_7_class'
                        ])
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'model_size': args.model_size,
            'feature_type': args.feature_type
            }

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

if args.task == 'KI67_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/KI67/KI67_7_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_KI67_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/Merged_KI67_7_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'HE_7_class_tumor_subtyping_bounded_to_KI67':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset_bounded_to_KI67.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'Merged_HE_7_class_tumor_subtyping_bounded_to_KI67':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_7_class_dataset_bounded_to_KI67.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'KI67_7_class_tumor_subtyping_bounded_to_HE':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/KI67/KI67_7_class_dataset_bounded_to_HE.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'Merged_KI67_7_class_tumor_subtyping_bounded_to_HE':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/Merged_KI67_7_class_dataset_bounded_to_HE.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_Histology_7_class_bounded_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'HE_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_HE_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_7_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_Histology_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'HE_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_10_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_HE_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_10_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Merged_Histology_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_10_class_dataset.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Histology_5_class_bounded':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/Merged_HE_5_class_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Histology_3_class_bounded':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_3_class_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Histology_5_class_HE_KI67_bounded':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/Merged_HE_KI67_5_class_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'Histology_7_class_HE_KI67_bounded':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_KI67/Merged_KI67_7_class_dataset_bounded_to_HE.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'HE_KI67_LGG_vs_HGG':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/Merged_HE_KI67_HGG_LGG_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'LGG_vs_HGG':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_HGG_LGG_dataset_bounded.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_type),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1},
                            patient_strat= False,
                            ignore=[])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
