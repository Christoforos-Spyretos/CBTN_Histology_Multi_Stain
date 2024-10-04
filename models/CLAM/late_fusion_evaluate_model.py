# imports
from __future__ import print_function
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import h5py
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

# pytorch imports
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd

# internal imports
from utils.utils import *
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset_modality_n, save_splits
from models.CLAM.utils.late_fusion_eval_utils import *

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_experiment_name(cfg):
    return '_'.join([str(cfg.seed),
                    str(cfg.stain_modalities.description),
                    str(cfg.n_classes),
                    str(cfg.label_dict),
                    str(cfg.csv_path),
                    str(cfg.patient_strat),
                    str(cfg.shuffle),
                    str(cfg.print_info),
                    str(cfg.late_fusion_method),
                    str(cfg.drop_out),
                    str(cfg.embed_dim),
                    str(cfg.results_dir),
                    str(cfg.splits_dir),
                    str(cfg.model_size),
                    str(cfg.k),
                    str(cfg.k_start),
                    str(cfg.k_end),
                    str(cfg.fold),
                    str(cfg.micro_average),
                    str(cfg.split),
                    str(cfg.task),
                    str(cfg.feature_type)])
       
@hydra.main(version_base="1.3.2", 
			config_path= '/home/chrsp39/CBTN_Histology_Multi_Modal/configs/evaluation', 
			config_name='late_fusion_evaluate_model')

def main(cfg:DictConfig):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(cfg.seed)

    experiment_name = build_experiment_name(cfg)

    save_dir = os.path.join('./eval_results', 'EVAL_' + str(cfg.stain_modalities.save_exp_code))

    models_dirs = []
    for i in enumerate(cfg.stain_modalities.models_exp_codes):
        models_dir = os.path.join(cfg.results_dir, str(cfg.stain_modalities[f"models_exp_code_{i}"]))
        models_dirs.append(models_dir)

    os.makedirs(save_dir, exist_ok=True)

    if cfg.splits_dir is None:
        cfg.splits_dir = models_dirs[0]

    for i, model_dir in enumerate(models_dirs):
        assert os.path.isdir(model_dir)
    assert os.path.isdir(cfg.splits_dir)

    settings = {'task': cfg.task,
                'split': cfg.splits_dir,
                'save_dir': save_dir, 
                'models_dirs': models_dirs,
                'model_type': cfg.model_type,
                'model_size': cfg.model_size,
                'feature_type': cfg.feature_type
                }

    if cfg.ignore is None:
        cfg.ignore = [] # Set to an empty list if None

    with open(save_dir + '/eval_experiment_{}.txt'.format(cfg.stain_modalities.save_exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print(settings)

    datasets = []
    for i, data_dir in enumerate(cfg.stain_modalities.data_root_dir):
        dataset = Generic_MIL_Dataset_modality_n(csv_path=cfg.csv_path,
                                                data_dir_n=data_dir,
                                                shuffle=cfg.shuffle, 
                                                print_info=cfg.print_info,
                                                label_dict=cfg.label_dict,
                                                patient_strat=cfg.patient_strat,
                                                ignore=cfg.ignore)
        datasets.append(dataset)

    if cfg.k_start == -1:
        start = 0
    else:
        start = cfg.k_start
    if cfg.k_end == -1:
        end = cfg.k
    else:
        end = cfg.k_end

    if cfg.fold == -1:
        folds = range(start, end)
    else:
        folds = range(cfg.fold, cfg.fold+1)

    ckpt_paths = []
    for i, model_dir in enumerate(cfg.stain_modalities.models_exp_codes):
        ckpt_paths.append([os.path.join(model_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds])

    datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

    all_results = []
    all_auc = []
    all_acc = []

    for ckpt_idx in range(len(ckpt_paths[0])):
        if datasets_id[cfg.split] < 0:
            split_dataset = datasets[0]
        else:
            csv_path = '{}/splits_{}.csv'.format(cfg.splits_dir, folds[ckpt_idx])
            datasets = datasets[0].return_splits(from_id=False, csv_path=csv_path)
            split_datasets = []
            
            for i in len(datasets):
                split_datasets.append(datasets[datasets_id[cfg.split]])

        models, patient_results, test_error, auc, df  = eval(split_datasets, cfg, ckpt_paths)
        
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(folds) != cfg.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(save_dir, save_name))

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")