from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lr_scheduler', default=None, choices=['cosine'])
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--use_class_weights', default=True)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
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
                        'Histology_7_class_bounded_HE_KI67',
                        'Histology_3_class_bounded',
                        'Histology_5_class_bounded_HE_KI67',
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
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'lr_scheduler':args.lr_scheduler,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'feature_type': args.feature_type,
            'use_class_weights':args.use_class_weights,
            'opt': args.opt}

if args.model_type in ['abmil','clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'HE_7_class_tumor_subtyping_bounded_to_KI67':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset_bounded_to_KI67.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_HE_7_class_tumor_subtyping_bounded_to_KI67':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/Merged_HE_7_class_dataset_bounded_to_KI67.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'KI67_7_class_tumor_subtyping_bounded_to_HE':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/KI67/KI67_7_class_dataset_bounded_to_HE.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_KI67_7_class_tumor_subtyping_bounded_to_HE':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/Merged_KI67_7_class_dataset_bounded_to_HE.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_Histology_7_class_bounded_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset_bounded.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'HE_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_HE_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_7_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'KI67_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/KI67/KI67_7_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_KI67_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/Merged_KI67_7_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_Histology_7_class_tumor_subtyping':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False, 
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6},
    patient_strat=False,
    ignore=[])

elif args.task == 'HE_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_10_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False,
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_HE_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_10_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False,
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
    patient_strat=False,
    ignore=[])

elif args.task == 'Merged_Histology_10_class_tumor_subtyping':
    args.n_classes=10
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_10_class_dataset.csv',
    data_dir= os.path.join(args.data_root_dir, args.feature_type),
    shuffle = False,
    seed = args.seed, 
    print_info = True,
    label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2, 'EP':3, 'GANG':4, 'MEN':5, 'ATRT':6, 'DIPG': 7, 'DNET': 8, 'CRAN': 9},
    patient_strat=False,
    ignore=[])

elif args.task == 'Histology_5_class_bounded':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/Merged_HE_5_class_dataset_bounded.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'Histology_7_class_bounded_HE_KI67':
    args.n_classes=7
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_KI67/Merged_KI67_7_class_dataset_bounded_to_HE.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4, 'MEN':5, 'ATRT':6},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'Histology_5_class_bounded_HE_KI67':
    args.n_classes=5
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/Merged_HE_KI67_5_class_dataset_bounded.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'EP':2, 'MED':3, 'GANG':4},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'Histology_3_class_bounded':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/Merged_HE_KI67_HGG_LGG_dataset_bounded.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1, 'MED':2},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'HE_KI67_LGG_vs_HGG':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/Merged_HE_KI67_HGG_LGG_dataset_bounded.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1},
                                  patient_strat=False,
                                  ignore=[])
    
elif args.task == 'LGG_vs_HGG':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_HGG_LGG_dataset_bounded.csv',
                                  data_dir= os.path.join(args.data_root_dir, args.feature_type),
                                  shuffle = False,
                                  seed = args.seed, 
                                  print_info = True,
                                  label_dict = {'ASTR_LGG':0, 'ASTR_HGG':1},
                                  patient_strat=False,
                                  ignore=[])
    
    if args.model_type in ['abmil','clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")