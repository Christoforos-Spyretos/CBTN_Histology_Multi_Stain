# imports
from __future__ import print_function
import argparse
import pdb
import os
import math
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from sklearn.utils.class_weight import compute_class_weight

# internal imports
from intermediate_fusion.classifier_utils import train
from torch.utils.data import Dataset
from utils.file_utils import save_pkl, load_pkl

# seeding
def seed_torch(seed=7, device=None):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_experiment_name(cfg):
    return '_'.join([
        
        str(cfg.seed),
                     
        str(cfg.task),
        str(cfg.n_classes),
        str(cfg.label_dict),

        str(cfg.csv_path),
        str(cfg.k),
        str(cfg.k_start),
        str(cfg.k_end),
        str(cfg.split_dir),
        str(cfg.patient_strat),
        str(cfg.shuffle),
        str(cfg.print_info), 
        str(cfg.label_frac), 

        str(cfg.feature_type),
        str(cfg.embed_dim), 

        str(cfg.max_epochs), 
        str(cfg.early_stopping),
        str(cfg.min_epochs),
        str(cfg.patience),
        str(cfg.stop_epoch),
        str(cfg.lr), 
        str(cfg.lr_scheduler),
        str(cfg.opt),
        str(cfg.reg),
        str(cfg.use_class_weights),
        str(cfg.ignore),
        str(cfg.testing),
        
        str(cfg.log_data),
        str(cfg.save_dir),
        ])

@hydra.main(version_base="1.3.2", 
			config_path= '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion', 
			config_name='run_classifier')

def main(cfg:DictConfig):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(cfg.seed)

    experiment_name = build_experiment_name(cfg)

    settings = {
        'seed': cfg.seed,

        'task': cfg.task,
        'n_classes': cfg.n_classes,
        'label_dict': cfg.label_dict,

        'csv_path': cfg.csv_path,
        'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir,
        'test_dir': cfg.test_dir,
        'k': cfg.k,
        'k_start': str(cfg.k_start),
        'k_end': str(cfg.k_end),
        'split_dir': cfg.split_dir,
        'patient_strat': str(cfg.patient_strat),
        'shuffle': str(cfg.shuffle),
        'print_info': str(cfg.print_info),
        'label_frac': str(cfg.label_frac),

        'feature_type': cfg.feature_type,
        'embed_dim': cfg.embed_dim,

        'max_epochs': cfg.max_epochs,
        'early_stopping': cfg.early_stopping,
        'lr': cfg.lr,
        'lr_scheduler': cfg.lr_scheduler,
        'opt': cfg.opt,
        'reg': cfg.reg,
        'use_class_weights': cfg.use_class_weights,
        'ignore': cfg.ignore,
        'weighted_sample': cfg.weighted_sample,


        'log_data': cfg.log_data,
        'save_dir': cfg.save_dir,
    }

    # create results directory if necessary
    if not os.path.isdir(cfg.save_dir):
        os.mkdir(cfg.save_dir)

    with open_dict(cfg):
        cfg.n_classes = cfg.n_classes

    print('\nLoad Dataset')
    if cfg.task:
        print(f'Task description: {cfg.task}')

    if cfg.ignore is None:
        cfg.ignore = []

    # Prepare data directory templates for each split
    train_dir = cfg.train_dir
    val_dir = cfg.val_dir
    test_dir = cfg.test_dir

    cfg.save_dir = os.path.join(cfg.save_dir, str(cfg.exp_code) + '_s{}'.format(cfg.seed))
    if not os.path.isdir(cfg.save_dir):
        os.mkdir(cfg.save_dir)

    if cfg.split_dir is None:
        cfg.split_dir = os.path.join('splits', cfg.task+'_{}'.format(int(cfg.label_frac*100)))
    else:
        cfg.split_dir = os.path.join('splits', cfg.split_dir)

    print('split_dir: ', cfg.split_dir)
    assert os.path.isdir(cfg.split_dir)

    settings.update({'split_dir': cfg.split_dir})

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val)) 

    if cfg.k_start == -1:
        start = 0
    else:
        start = cfg.k_start
    if cfg.k_end == -1:
        end = cfg.k
    else:
        end = cfg.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)


    # Minimal custom dataset for .pt files and split CSVs
    class SimplePTDataset(Dataset):
        def __init__(self, split_ids, all_data_df, pt_dir, label_dict):
            self.df = all_data_df[all_data_df['slide_id'].isin(split_ids)].reset_index(drop=True)
            self.pt_dir = pt_dir
            self.label_dict = label_dict

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            slide_id = row['slide_id']
            label = self.label_dict[row['label']] if isinstance(row['label'], str) else row['label']
            pt_path = os.path.join(self.pt_dir, "pt_files", f"{slide_id}.pt")
            obj = torch.load(pt_path, weights_only=False)
            if isinstance(obj, dict):
                if 'cross_attended' in obj:
                    arr = obj['cross_attended']
                    if isinstance(arr, np.ndarray):
                        features = torch.from_numpy(arr)
                    else:
                        features = arr
                else:
                    raise KeyError(f"'cross_attended' key not found in {pt_path}. Available keys: {list(obj.keys())}")
            else:
                features = obj
            return features, label

        def get_class_weights(self):
            labels = self.df['label']
            # If labels are strings, map them to integers using label_dict
            if isinstance(labels.iloc[0], str):
                labels = labels.map(self.label_dict)
            unique_classes = np.array(sorted(set(labels)))
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=labels
            )
            return class_weights

    all_data_df = pd.read_csv(cfg.csv_path)

    for i in folds:
        split_csv = f"{cfg.split_dir}/splits_{i}.csv"
        split_df = pd.read_csv(split_csv)
        train_ids = split_df['train'].dropna().tolist()
        val_ids = split_df['val'].dropna().tolist()
        test_ids = split_df['test'].dropna().tolist()

        train_dir_i = cfg.train_dir.format(fold=i)
        val_dir_i = cfg.val_dir.format(fold=i)
        test_dir_i = cfg.test_dir.format(fold=i)

        train_dataset = SimplePTDataset(train_ids, all_data_df, train_dir_i, cfg.label_dict)
        val_dataset = SimplePTDataset(val_ids, all_data_df, val_dir_i, cfg.label_dict)
        test_dataset = SimplePTDataset(test_ids, all_data_df, test_dir_i, cfg.label_dict)

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, cfg)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(cfg.save_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != cfg.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(cfg.save_dir, save_name))

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")    