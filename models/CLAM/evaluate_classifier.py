
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
        str(cfg.test_dir),
        str(cfg.k),
        str(cfg.k_start),
        str(cfg.k_end),
        str(cfg.split_dir),

        str(cfg.feature_type),
        str(cfg.embed_dim),

        str(cfg.models_dir),
        str(cfg.model_type),
        str(cfg.model_size),
        str(cfg.ignore),

        str(cfg.save_dir),
        str(cfg.exp_code),

        str(cfg.micro_average)
    ])

@hydra.main(version_base="1.3.2", 
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion',
            config_name='evaluate_classifier')

def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device)

    experiment_name = build_experiment_name(cfg)
    save_dir = cfg.save_dir
    models_dir = os.path.join(cfg.models_dir)
    print(models_dir)
    os.makedirs(save_dir, exist_ok=True)

    if cfg.split_dir is None:
        cfg.split_dir = models_dir

    assert os.path.isdir(models_dir)
    assert os.path.isdir(cfg.split_dir)

    settings = {
        'seed': cfg.seed,

        'task': cfg.task,
        'n_classes': cfg.n_classes,
        'label_dict': cfg.label_dict,

        'csv_path': cfg.csv_path,
        'test_dir': cfg.test_dir,
        'k': cfg.k,
        'k_start': cfg.k_start,
        'k_end': cfg.k_end,
        'split_dir': cfg.split_dir,

        'feature_type': cfg.feature_type,
        'embed_dim': cfg.embed_dim,

        'models_dir': models_dir,
        'model_type': cfg.model_type,
        'model_size': cfg.model_size,
        'ignore': cfg.ignore,
        
        'save_dir': cfg.save_dir,
        'exp_code': cfg.exp_code,

        'micro_average': cfg.micro_average
    }

    if cfg.ignore is None:
        cfg.ignore = []

        with open(os.path.join(save_dir, 'eval_experiment.txt'), 'w') as f:
            print(settings, file=f)

    print(settings)

    # load dataset CSV
    all_data_df = pd.read_csv(cfg.csv_path)

    # determine folds
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
        # load split CSV for this fold
        split_csv = os.path.join(cfg.split_dir, f'splits_{fold}.csv')
        split_df = pd.read_csv(split_csv)
        test_ids = split_df['test'].dropna().tolist()

        # prepare test dataset
        class SimplePTDataset(torch.utils.data.Dataset):
            def __init__(self, split_ids, all_data_df, pt_dir, label_dict):
                self.df = all_data_df[all_data_df['slide_id'].isin(split_ids)].reset_index(drop=True)
                self.pt_dir = pt_dir
                self.label_dict = label_dict
                # add slide_data attribute for compatibility with classifier_eval_utils.py
                self.slide_data = self.df
            def __len__(self):
                return len(self.df)
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                slide_id = row['slide_id']
                label = self.label_dict[row['label']] if isinstance(row['label'], str) else row['label']
                pt_path = os.path.join(self.pt_dir, "pt_files", f"{slide_id}.pt")
                obj = torch.load(pt_path, weights_only=False)
                if isinstance(obj, dict):
                    fusion_keys = ['cross_attended', 'element_wise_mult', 'concatenated']
                    matched_key = next((k for k in fusion_keys if k in obj), None)
                    if matched_key is not None:
                        arr = obj[matched_key]
                        if isinstance(arr, np.ndarray):
                            features = torch.from_numpy(arr)
                        else:
                            features = arr
                    else:
                        raise KeyError(f"No recognized fusion key found in {pt_path}. Available keys: {list(obj.keys())}")
                else:
                    features = obj
                return features, label

        test_dir_i = cfg.test_dir.format(fold=fold)
        test_dataset = SimplePTDataset(test_ids, all_data_df, test_dir_i, cfg.label_dict)

        # load model checkpoint
        ckpt_path = os.path.join(models_dir, f's_{fold}_checkpoint.pt')
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        # initialise model
        model = Classifier(embed_dim=cfg.embed_dim, n_classes=cfg.n_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        # create loader
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # evaluate classifier
        results, test_error, auc, df, _ = eval_classifier(model, loader, cfg.n_classes, device)
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        df.to_csv(os.path.join(save_dir, f'fold_{fold}.csv'), index=False)

    n_results = min(len(list(folds)), len(all_auc), len(all_acc))
    final_df = pd.DataFrame({
        'folds': list(folds)[:n_results],
        'test_auc': all_auc[:n_results],
        'test_acc': all_acc[:n_results]
    })
    if n_results != cfg.k:
        save_name = f'summary_partial_{list(folds)[0]}_{list(folds)[n_results-1]}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(save_dir, save_name))

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")