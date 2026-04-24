# imports
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

# internal imports
from intermediate_fusion.cross_attention_classifier import CrossAttentionClassifier
from intermediate_fusion.classifier_utils import Accuracy_Logger, EarlyStopping
from utils.utils import get_optim, get_lr_scheduler, calculate_error, print_network
from utils.file_utils import save_pkl

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


# DATASET UTILITIES
class BiModalPTDataset(Dataset):
    """
    Loads one modality 1 .pt file and one modality 2 .pt file per subject.
    Returns (m1, m2, label) where both modality tensors are (1, d).
    """

    def __init__(self, split_ids, all_data_df, mod1_dir, mod2_dir, label_dict):
        self.df = all_data_df[all_data_df['slide_id'].isin(split_ids)].reset_index(drop=True)
        self.mod1_dir = mod1_dir  # modality 1 pt_files directory
        self.mod2_dir = mod2_dir  # modality 2 pt_files directory
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

    def get_class_weights(self):
        labels = self.df['label']
        if isinstance(labels.iloc[0], str):
            labels = labels.map(self.label_dict)
        unique_classes = np.array(sorted(set(labels)))
        return compute_class_weight('balanced', classes=unique_classes, y=labels.to_numpy())

def collate_bimodal(batch):
    m1     = torch.stack([b[0] for b in batch])    # (B, 1, d)
    m2     = torch.stack([b[1] for b in batch])    # (B, 1, d)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return m1, m2, labels


def get_bimodal_loader(dataset, training=False, weighted=False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}
    if training:
        if weighted:
            from torch.utils.data import WeightedRandomSampler
            weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float)
            label_weights = weights[[
                dataset.label_dict[r] if isinstance(r, str) else int(r)
                for r in dataset.df['label']
            ]]
            sampler = WeightedRandomSampler(label_weights, len(label_weights))
            return DataLoader(dataset, batch_size=1, sampler=sampler,
                              collate_fn=collate_bimodal, **kwargs)
        else:
            from torch.utils.data import RandomSampler
            return DataLoader(dataset, batch_size=1, sampler=RandomSampler(dataset),
                              collate_fn=collate_bimodal, **kwargs)
    else:
        from torch.utils.data import SequentialSampler
        return DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                          collate_fn=collate_bimodal, **kwargs)

#  train and validate                            #
def train_loop_ca(epoch, model, loader, optimizer, n_classes, writer=None,
                  loss_fn=None, scheduler=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    current_lr = optimizer.param_groups[0]['lr']

    print('\n')
    for batch_idx, (m1, m2, label) in enumerate(loader):
        m1    = m1.squeeze(0).to(device)    # (1, d)
        m2    = m2.squeeze(0).to(device)    # (1, d)
        label = label.to(device)

        logits, Y_prob, Y_hat = model(m1, m2)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        train_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss.item(), label.item()))

        error = calculate_error(Y_hat, label)
        train_error += error

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

    train_loss  /= len(loader)
    train_error /= len(loader)
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/learning_rate', current_lr, epoch)


def validate_ca(cur, epoch, model, loader, n_classes, early_stopping=None,
                writer=None, loss_fn=None, save_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob   = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (m1, m2, label) in enumerate(loader):
            m1    = m1.squeeze(0).to(device, non_blocking=True)
            m2    = m2.squeeze(0).to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat = model(m1, m2)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)

            prob[batch_idx]   = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss  += loss.item()
            val_error += calculate_error(Y_hat, label)

    val_error /= len(loader)
    val_loss  /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert save_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(save_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_ca(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.

    all_probs  = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.df['slide_id']
    patient_results = {}

    for batch_idx, (m1, m2, label) in enumerate(loader):
        m1    = m1.squeeze(0).to(device)
        m2    = m2.squeeze(0).to(device)
        label = label.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.inference_mode():
            logits, Y_prob, Y_hat = model(m1, m2)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx]  = probs
        all_labels[batch_idx] = label.item()

        patient_results[slide_id] = {
            'slide_id': np.array(slide_id),
            'prob': probs,
            'label': label.item()
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

    return patient_results, test_error, auc, acc_logger


#  train a single fold                                                 #
def train_ca(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.save_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    train_split, val_split, test_split = datasets

    # save split info
    split_df = pd.concat([
        pd.DataFrame({'train': train_split.df['slide_id']}),
        pd.DataFrame({'val':   val_split.df['slide_id']}),
        pd.DataFrame({'test':  test_split.df['slide_id']}),
    ], axis=1)
    split_df.to_csv(os.path.join(args.save_dir, f'splits_{cur}.csv'), index=False)

    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    # loss
    if args.use_class_weights:
        class_weights = torch.tensor(train_split.get_class_weights(), dtype=torch.float)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f'Using weighted CE. Class weights: {class_weights}')
    else:
        loss_fn = nn.CrossEntropyLoss()
    if device.type == 'cuda':
        loss_fn = loss_fn.cuda()

    # model
    model = CrossAttentionClassifier(embed_dim=args.embed_dim, n_classes=args.n_classes)
    model = model.to(device)
    print_network(model)

    # optimizer & scheduler
    optimizer = get_optim(model, args)
    steps = len(get_bimodal_loader(train_split, training=True)) * (args.max_epochs + 1)
    scheduler = get_lr_scheduler(optimizer, steps, args) if args.lr_scheduler else None

    # early stopping
    early_stopping = EarlyStopping(
        min_epochs=args.min_epochs, patience=args.patience,
        stop_epoch=args.stop_epoch, verbose=True
    ) if args.early_stopping else None

    # loaders
    train_loader = get_bimodal_loader(train_split, training=True, weighted=args.weighted_sample)
    val_loader   = get_bimodal_loader(val_split)
    test_loader  = get_bimodal_loader(test_split)

    for epoch in range(args.max_epochs):
        train_loop_ca(epoch, model, train_loader, optimizer, args.n_classes,
                      writer, loss_fn, scheduler)
        stop = validate_ca(cur, epoch, model, val_loader, args.n_classes,
                           early_stopping, writer, loss_fn, args.save_dir)
        if stop:
            break

    ckpt_path = os.path.join(args.save_dir, "s_{}_checkpoint.pt".format(cur))
    if args.early_stopping:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        torch.save(model.state_dict(), ckpt_path)

    _, val_error, val_auc, _ = summary_ca(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary_ca(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error

@hydra.main(version_base="1.3.2",
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/intermediate_fusion',
            config_name='run_intermediate_cross_attention_classifier')
def main(cfg: DictConfig):

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(cfg.seed, device_)

    if not os.path.isdir(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.n_classes = cfg.n_classes

    if cfg.ignore is None:
        cfg.ignore = []

    cfg.save_dir = os.path.join(cfg.save_dir, str(cfg.exp_code) + '_s{}'.format(cfg.seed))
    os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.split_dir is None:
        cfg.split_dir = os.path.join('splits', cfg.task + '_{}'.format(int(cfg.label_frac * 100)))
    else:
        cfg.split_dir = os.path.join('splits', cfg.split_dir)

    print('split_dir: ', cfg.split_dir)
    assert os.path.isdir(cfg.split_dir)

    start = 0 if cfg.k_start == -1 else cfg.k_start
    end   = cfg.k if cfg.k_end == -1 else cfg.k_end
    folds = np.arange(start, end)

    all_test_auc = []
    all_val_auc  = []
    all_test_acc = []
    all_val_acc  = []

    all_data_df = pd.read_csv(cfg.csv_path)

    for i in folds:
        split_csv = f"{cfg.split_dir}/splits_{i}.csv"
        split_df  = pd.read_csv(split_csv)
        train_ids = split_df['train'].dropna().tolist()
        val_ids   = split_df['val'].dropna().tolist()
        test_ids  = split_df['test'].dropna().tolist()

        # build per-fold pt_files dirs for both modalities
        # yaml mod1_dir/mod2_dir already contain pt_files before fold_{fold}
        mod1_train = cfg.mod1_dir.format(fold=i)
        mod1_val   = cfg.mod1_dir.format(fold=i).replace('/train/', '/val/')
        mod1_test  = cfg.mod1_dir.format(fold=i).replace('/train/', '/test/')
        mod2_train = cfg.mod2_dir.format(fold=i)
        mod2_val   = cfg.mod2_dir.format(fold=i).replace('/train/', '/val/')
        mod2_test  = cfg.mod2_dir.format(fold=i).replace('/train/', '/test/')

        train_dataset = BiModalPTDataset(train_ids, all_data_df, mod1_train, mod2_train, cfg.label_dict)
        val_dataset   = BiModalPTDataset(val_ids,   all_data_df, mod1_val,   mod2_val,   cfg.label_dict)
        test_dataset  = BiModalPTDataset(test_ids,  all_data_df, mod1_test,  mod2_test,  cfg.label_dict)

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train_ca(datasets, i, cfg)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        filename = os.path.join(cfg.save_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)

    final_df = pd.DataFrame({
        'folds':    folds,
        'test_auc': all_test_auc,
        'val_auc':  all_val_auc,
        'test_acc': all_test_acc,
        'val_acc':  all_val_acc
    })

    save_name = 'summary_partial_{}_{}.csv'.format(start, end) if len(folds) != cfg.k else 'summary.csv'
    final_df.to_csv(os.path.join(cfg.save_dir, save_name))


if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")
