# imports
from __future__ import print_function
import os
import numpy as np
import hydra
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
import pandas as pd

# internal imports
from intermediate_fusion.cross_attention_classifier import CrossAttentionClassifier
from intermediate_fusion.classifier_utils import Accuracy_Logger
from utils.utils import calculate_error, print_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  seeding                                                             #
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


#  DATASET UTILITIES                                                             #
class BiModalPTDataset(Dataset):
    """
    Loads one modality 1 .pt file and one modality 2 .pt file per subject.
    Returns (m1, m2, label) where both modality tensors are (1, d).
    """

    def __init__(self, split_ids, all_data_df, mod1_dir, mod2_dir, label_dict):
        self.df = all_data_df[all_data_df['slide_id'].isin(split_ids)].reset_index(drop=True)
        self.mod1_dir = mod1_dir
        self.mod2_dir = mod2_dir
        self.label_dict = label_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        label = self.label_dict[row['label']] if isinstance(row['label'], str) else int(row['label'])

        def load_vec(pt_dir):
            obj = torch.load(os.path.join(pt_dir, f"{slide_id}.pt"), weights_only=False)
            vec = obj['subject_attention'] if isinstance(obj, dict) else obj
            return torch.tensor(vec, dtype=torch.float32).view(1, -1)

        return load_vec(self.mod1_dir), load_vec(self.mod2_dir), label

def collate_bimodal(batch):
    m1     = torch.stack([b[0] for b in batch])
    m2     = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return m1, m2, labels


def get_bimodal_loader(dataset):
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}
    return DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                      collate_fn=collate_bimodal, **kwargs)


#  evaluation                                                     #
def summary_ca(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.

    all_probs  = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.df['slide_id']
    patient_results = {}

    with torch.inference_mode():
        for batch_idx, (m1, m2, label) in enumerate(loader):
            m1    = m1.squeeze(0).to(device)
            m2    = m2.squeeze(0).to(device)
            label = label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            logits, Y_prob, Y_hat = model(m1, m2)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx]  = probs
            all_labels[batch_idx] = label.item()

            patient_results[slide_id] = {
                'slide_id': np.array(slide_id),
                'prob': probs,
                'label': label.item(),
                'Y_hat': Y_hat.item()
            }
            test_error += calculate_error(Y_hat, label)

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=list(range(n_classes)))
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(aucs)

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': np.zeros(len(loader))}
    for c in range(n_classes):
        results_dict[f'p_{c}'] = all_probs[:, c]
    # fill Y_hat column
    for batch_idx, sid in enumerate(slide_ids):
        results_dict['Y_hat'][batch_idx] = patient_results[sid]['Y_hat']

    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc, df, acc_logger

@hydra.main(version_base="1.3.2",
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion',
            config_name='evaluate_intermediate_cross_attention_classifier')

def main(cfg: DictConfig):

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device_)

    os.makedirs(cfg.save_dir, exist_ok=True)

    assert os.path.isdir(cfg.models_dir), f"models_dir not found: {cfg.models_dir}"
    assert os.path.isdir(cfg.split_dir),  f"split_dir not found: {cfg.split_dir}"

    settings = {k: v for k, v in cfg.items()}
    with open(os.path.join(cfg.save_dir, 'eval_experiment.txt'), 'w') as f:
        print(settings, file=f)
    print(settings)

    all_data_df = pd.read_csv(cfg.csv_path)

    start = 0          if cfg.k_start == -1 else cfg.k_start
    end   = cfg.k      if cfg.k_end   == -1 else cfg.k_end
    folds = range(start, end)

    all_auc = []
    all_acc = []

    for fold in folds:
        split_csv = os.path.join(cfg.split_dir, f'splits_{fold}.csv')
        split_df  = pd.read_csv(split_csv)
        test_ids  = split_df['test'].dropna().tolist()

        mod1_test = cfg.mod1_dir.format(fold=fold)
        mod2_test = cfg.mod2_dir.format(fold=fold)

        test_dataset = BiModalPTDataset(test_ids, all_data_df, mod1_test, mod2_test, cfg.label_dict)
        test_loader  = get_bimodal_loader(test_dataset)

        # load checkpoint
        ckpt_path = os.path.join(cfg.models_dir, f's_{fold}_checkpoint.pt')
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        model = CrossAttentionClassifier(embed_dim=cfg.embed_dim, n_classes=cfg.n_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device_, weights_only=True))
        model = model.to(device_)
        print_network(model)

        patient_results, test_error, auc, df, acc_logger = summary_ca(model, test_loader, cfg.n_classes)

        print(f'Fold {fold} — test_error: {test_error:.4f}, AUC: {auc:.4f}')
        for i in range(cfg.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print(f'  class {i}: acc {acc}, correct {correct}/{count}')

        all_auc.append(auc)
        all_acc.append(1 - test_error)
        df.to_csv(os.path.join(cfg.save_dir, f'fold_{fold}.csv'), index=False)

    folds_list = list(folds)
    final_df = pd.DataFrame({
        'folds':    folds_list,
        'test_auc': all_auc,
        'test_acc': all_acc
    })
    save_name = 'summary_partial_{}_{}.csv'.format(start, end - 1) if len(folds_list) != cfg.k else 'summary.csv'
    final_df.to_csv(os.path.join(cfg.save_dir, save_name))
    print(f'\nMean AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}')
    print(f'Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}')


if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")
