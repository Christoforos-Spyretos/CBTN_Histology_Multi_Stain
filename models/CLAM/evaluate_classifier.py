
# imports
from __future__ import print_function
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import os
import pandas as pd

# internal imports
from intermediate_fusion.classifier_utils import Classifier
from intermediate_fusion.classifier_eval_utils import eval_classifier

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
        str(cfg.models_dir),
        str(cfg.feature_type),
        str(cfg.embed_dim),
        str(cfg.save_dir)
    ])

@hydra.main(version_base="1.3.2", 
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion',
            config_name='evaluate_classifier')

def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device)

    experiment_name = build_experiment_name(cfg)
    save_dir = os.path.join('./eval_results', 'EVAL_' + str(cfg.save_dir))
    models_dir = os.path.join(cfg.models_dir)
    print(models_dir)
    os.makedirs(save_dir, exist_ok=True)

    if cfg.splits_dir is None:
        cfg.splits_dir = models_dir

    assert os.path.isdir(models_dir)
    assert os.path.isdir(cfg.splits_dir)

    settings = {
        'task': cfg.task,
        'split': cfg.splits_dir,
        'save_dir': save_dir,
        'models_dir': models_dir,
        'feature_type': cfg.feature_type
    }

    if cfg.ignore is None:
        cfg.ignore = []

    with open(os.path.join(save_dir, f'eval_experiment_{cfg.save_dir}.txt'), 'w') as f:
        print(settings, file=f)

    print(settings)

    # Load dataset CSV
    all_data_df = pd.read_csv(cfg.csv_path)

    # Determine folds
    if cfg.k_start == -1:
        start = 0
    else:
        start = cfg.k_start
    if cfg.k_end == -1:
        end = cfg.k
    else:
        end = cfg.k_end

    if hasattr(cfg, 'fold') and cfg.fold != -1:
        folds = range(cfg.fold, cfg.fold+1)
    else:
        folds = range(start, end)

    all_auc = []
    all_acc = []

    for fold in folds:
        # Load split CSV for this fold
        split_csv = os.path.join(cfg.splits_dir, f'splits_{fold}.csv')
        split_df = pd.read_csv(split_csv)
        test_ids = split_df['test'].dropna().tolist()

        # Prepare test dataset
        class SimplePTDataset(torch.utils.data.Dataset):
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
                obj = torch.load(pt_path)
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

        test_dir_i = cfg.test_dir.format(fold=fold)
        test_dataset = SimplePTDataset(test_ids, all_data_df, test_dir_i, cfg.label_dict)

        # Load model checkpoint
        ckpt_path = os.path.join(models_dir, f's_{fold}_checkpoint.pt')
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        # Evaluate classifier
        model, results, test_error, auc, df = eval_classifier(test_dataset, cfg, ckpt_path)
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        df.to_csv(os.path.join(save_dir, f'fold_{fold}.csv'), index=False)

    final_df = pd.DataFrame({'folds': list(folds), 'test_auc': all_auc, 'test_acc': all_acc})
    if len(list(folds)) != cfg.k:
        save_name = f'summary_partial_{list(folds)[0]}_{list(folds)[-1]}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(save_dir, save_name))

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")