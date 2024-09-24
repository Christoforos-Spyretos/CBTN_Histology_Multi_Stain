import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits_dual_modality
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_abmil import ABMIL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, min_epochs:int=10, patience=5, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.min_epochs = min_epochs
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        if epoch > self.min_epochs:
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
            elif score < self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split_1, val_split_1, test_split_1, train_split_2, val_split_2, test_split_2 = datasets
    save_splits_dual_modality(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')

    print("Training on {} samples".format(len(train_split_1)))
    print("Validating on {} samples".format(len(val_split_1)))
    print("Testing on {} samples".format(len(test_split_1)))

    print("Training on {} samples".format(len(train_split_2)))
    print("Validating on {} samples".format(len(val_split_2)))
    print("Testing on {} samples".format(len(test_split_2)))

    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
    else:
        if args.use_class_weights:
            class_weights = torch.Tensor(train_split_1.get_class_weights())
            print(f'Using weighted CE as bag loss. Class weights: {class_weights}')
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and any([args.model_type != 'mil', args.model_type != 'abmil']):
        model_dict.update({"size_arg": args.model_size})
    else:
        model_dict.update({"size_arg": None})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    elif args.model_type == 'abmil':
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        model = ABMIL(**model_dict)
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader_1 = get_split_loader(train_split_1, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader_1 = get_split_loader(val_split_1,  testing = args.testing)
    test_loader_1 = get_split_loader(test_split_1, testing = args.testing)
    train_loader_2 = get_split_loader(train_split_2, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader_2 = get_split_loader(val_split_2,  testing = args.testing)
    test_loader_2 = get_split_loader(test_split_2, testing = args.testing)
    steps = len(train_loader_1) * (args.max_epochs+1) # this is for the lr scheduler
    print('Done!')
    # LR scheduler
    if args.lr_scheduler:
        print('\nInit LR scheduler ...', end=' ')
        scheduler = get_lr_scheduler(optimizer,steps, args)
        print('Done!')
    else:
        scheduler = None

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 5, stop_epoch=20, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader_1, train_loader_2, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, scheduler=scheduler)
            stop = validate_clam(cur, epoch, model, val_loader_1, val_loader_2, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader_1, train_loader_2, optimizer, args.n_classes, writer, loss_fn, scheduler=scheduler)
            stop = validate(cur, epoch, model, val_loader_1, val_loader_2, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)),weights_only=True))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader_1, val_loader_2, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader_1, test_loader_2, args.n_classes)
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
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader1, loader2, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, scheduler = None):
    model.train()
    acc_logger1 = Accuracy_Logger(n_classes=n_classes)
    inst_logger1 = Accuracy_Logger(n_classes=n_classes)

    acc_logger2 = Accuracy_Logger(n_classes=n_classes)
    inst_logger2 = Accuracy_Logger(n_classes=n_classes)

    acc_logger_avg = Accuracy_Logger(n_classes=n_classes)

    
    train_loss1 = 0.
    train_error1 = 0.
    train_inst_loss1 = 0.
    inst_count1 = 0

    train_loss2 = 0.
    train_error2 = 0.
    train_inst_loss2 = 0.
    inst_count2 = 0

    train_loss_avg = 0.
    train_error_avg = 0.
    train_inst_loss_avg = 0.
    inst_count = 0

    current_lr = optimizer.param_groups[0]['lr']

    epoch_total_loss1 = 0
    epoch_total_loss2 = 0
    epoch_total_loss_avg = 0

    print('\n')
    for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):
        data1, label1 = data1.to(device), label1.to(device)
        data2, label2 = data2.to(device), label2.to(device)

        logits1, Y_prob1, Y_hat1, _, instance_dict1 = model(data1, label=label1, instance_eval=True)
        logits2, Y_prob2, Y_hat2, _, instance_dict2 = model(data2, label=label2, instance_eval=True)

        # Aggregate predictions via averaging (late fusion)
        logits_avg = (logits1 + logits2) / 2
        Y_prob_avg = (Y_prob1 + Y_prob2) / 2
        
        # Convert averaged probabilities to hard predictions (Y_hat)
        Y_hat_avg = torch.argmax(Y_prob_avg, dim=1)

        acc_logger1.log(Y_hat1, label1)
        acc_logger2.log(Y_hat2, label2)
        acc_logger_avg.log(Y_hat_avg, label1)

        loss1 = loss_fn(logits1, label1)
        loss_value1 = loss1.item()

        loss2 = loss_fn(logits2, label2)
        loss_value2 = loss2.item()

        loss_avg = (loss1 + loss2) / 2
        loss_value_avg = loss_avg.item()

        instance_loss1 = instance_dict1['instance_loss']
        inst_count1+=1
        instance_loss_value1 = instance_loss1.item()
        train_inst_loss1 += instance_loss_value1
        total_loss1 = bag_weight * loss1 + (1-bag_weight) * instance_loss1

        instance_loss2 = instance_dict2['instance_loss']
        inst_count2+=1
        instance_loss_value2 = instance_loss2.item()
        train_inst_loss2 += instance_loss_value2
        total_loss2 = bag_weight * loss2 + (1-bag_weight) * instance_loss2
       
        instance_loss_avg = (instance_dict1['instance_loss'] + instance_dict2['instance_loss']) / 2
        inst_count+=1
        instance_loss_value_avg = instance_loss_avg.item()
        train_inst_loss_avg += instance_loss_value_avg
        total_loss_avg = bag_weight * loss_avg + (1-bag_weight) * instance_loss_avg

        inst_preds1 = instance_dict1['inst_preds']
        inst_labels1 = instance_dict1['inst_labels']
        inst_logger1.log_batch(inst_preds1, inst_labels1)

        inst_preds2 = instance_dict2['inst_preds']
        inst_labels2 = instance_dict2['inst_labels']
        inst_logger2.log_batch(inst_preds2, inst_labels2)

        train_loss1 += loss_value1
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value1, instance_loss_value1, total_loss1.item(), current_lr) + 
                'label: {}, bag_size: {}'.format(label1.item(), data1.size(0)))
            
        train_loss2 += loss_value2
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value2, instance_loss_value2, total_loss2.item(), current_lr) + 
                'label: {}, bag_size: {}'.format(label2.item(), data2.size(0)))

        train_loss_avg += loss_value_avg
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value_avg, instance_loss_value_avg, total_loss_avg.item(), current_lr) + 
                'label: {}, bag_size: {}'.format(label1.item(), data1.size(0), data2.size(0)))

        error1 = calculate_error(Y_hat1, label1)
        train_error1 += error1
        epoch_total_loss1 += total_loss1

        error2 = calculate_error(Y_hat2, label2)
        train_error2 += error2
        epoch_total_loss2 += total_loss2

        error_avg = (error1 + error2) / 2
        train_error_avg += error_avg
        epoch_total_loss_avg += total_loss_avg

        # backward pass
        total_loss1.backward()
        total_loss2.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

    # calculate loss and error for epoch
    train_loss1 /= len(loader1)
    train_error1 /= len(loader1)
    epoch_total_loss1 /= len(loader1)

    train_loss2 /= len(loader2)
    train_error2 /= len(loader2)
    epoch_total_loss2 /= len(loader2)

    train_loss_avg /= len(loader1)
    train_error_avg /= len(loader1)
    epoch_total_loss_avg /= len(loader1)

    if inst_count1 > 0:
        train_inst_loss1 /= inst_count1
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger1.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if inst_count2 > 0:
        train_inst_loss2 /= inst_count2
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger2.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss1: {:.4f}, train_clustering_loss1:  {:.4f}, train_error1: {:.4f}'.format(epoch, train_loss1, train_inst_loss1, train_error1, current_lr))
    print('Epoch: {}, train_loss2: {:.4f}, train_clustering_loss2:  {:.4f}, train_error2: {:.4f}'.format(epoch, train_loss2, train_inst_loss2, train_error2, current_lr))
    print('Epoch: {}, train_loss_avg: {:.4f}, train_clustering_loss_avg:  {:.4f}, train_error_avg: {:.4f}'.format(epoch, train_loss_avg, train_inst_loss_avg, train_error_avg, current_lr))
    
    for i in range(n_classes):
        acc1, correct, count = acc_logger1.get_summary(i)
        acc2, correct, count = acc_logger2.get_summary(i)
        acc_avg, correct, count = acc_logger_avg.get_summary(i)
        print('class {}: acc1 {}, correct {}/{} '.format(i, acc1, correct, count))
        print('class {}: acc2 {}, correct {}/{} '.format(i, acc2, correct, count))
        print('class {}: acc_avg {}, correct {}/{} '.format(i, acc_avg, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc1'.format(i), acc1, epoch)
            writer.add_scalar('train/class_{}_acc2'.format(i), acc2, epoch)
            writer.add_scalar('train/class_{}_acc_avg'.format(i), acc_avg, epoch)

    if writer:
        writer.add_scalar('train/loss1', train_loss1, epoch)
        writer.add_scalar('train/error1', train_error1, epoch)
        writer.add_scalar('train/clustering_loss1', train_inst_loss1, epoch)
        writer.add_scalar('train/epoch_total_loss1', epoch_total_loss1, epoch)

        writer.add_scalar('train/loss1', train_loss2, epoch)
        writer.add_scalar('train/error1', train_error2, epoch)
        writer.add_scalar('train/clustering_loss1', train_inst_loss2, epoch)
        writer.add_scalar('train/epoch_total_loss1', epoch_total_loss2, epoch)

        writer.add_scalar('train/loss1', train_loss_avg, epoch)
        writer.add_scalar('train/error1', train_error_avg, epoch)
        writer.add_scalar('train/clustering_loss1', train_inst_loss_avg, epoch)
        writer.add_scalar('train/epoch_total_loss1', epoch_total_loss_avg, epoch)

        writer.add_scalar('train/learning_rate', current_lr, epoch)

def train_loop(epoch, model, loader1, loader2, optimizer, n_classes, writer = None, loss_fn = None, scheduler = None):   
    model.train()

    acc_logger1 = Accuracy_Logger(n_classes=n_classes)
    acc_logger2 = Accuracy_Logger(n_classes=n_classes)
    acc_logger_avg = Accuracy_Logger(n_classes=n_classes)

    train_loss1 = 0.
    train_error1 = 0.

    train_loss2 = 0.
    train_error2 = 0.

    train_loss_avg = 0.
    train_error_avg = 0.

    current_lr = optimizer.param_groups[0]['lr']

    print('\n')
    for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):
        
        data1, label1 = data1.to(device), label1.to(device)
        data2, label2 = data2.to(device), label2.to(device)

        logits1, Y_prob1, Y_hat1, _, _ = model(data1)
        logits2, Y_prob2, Y_hat2, _, _ = model(data2)
        
        acc_logger1.log(Y_hat1, label1)
        loss1= loss_fn(logits1, label1)
        loss_value1 = loss1.item()

        acc_logger2.log(Y_hat2, label2)
        loss2= loss_fn(logits2, label2)
        loss_value2 = loss2.item()

        logits_avg = (logits1 + logits2) / 2
        Y_prob_avg = (Y_prob1 + Y_prob2) / 2
        Y_hat_avg = torch.argmax(Y_prob_avg, dim=1)

        acc_logger_avg.log(Y_hat_avg, label1)
        loss_avg = (loss1 + loss2) / 2
        loss_value_avg = loss_avg.item()
        
        train_loss1 += loss_value1
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value1, label1.item(), data1.size(0), current_lr))
        
        train_loss2 += loss_value2
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value2, label2.item(), data2.size(0), current_lr))

        train_loss_avg += (loss_value1 + loss_value2) / 2
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, (loss_value1 + loss_value2) / 2, label1.item(), data1.size(0), data2.size(0), current_lr))


        error1 = calculate_error(Y_hat1, label1)
        train_error1 += error1

        error2 = calculate_error(Y_hat2, label2)
        train_error2 += error2

        error_avg = (error1 + error2) / 2
        train_error_avg += error_avg
        
        # backward pass
        loss1.backward()
        loss2.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

    # calculate loss and error for epoch
    train_loss1 /= len(loader1)
    train_error1 /= len(loader1)

    train_loss2 /= len(loader2)
    train_error2 /= len(loader2)

    train_loss_avg /= len(loader1)
    train_error_avg /= len(loader1)

    print('Epoch: {}, train_loss1: {:.4f}, train_error1: {:.4f}'.format(epoch, train_loss1, train_error1))
    print('Epoch: {}, train_loss2: {:.4f}, train_error2: {:.4f}'.format(epoch, train_loss2, train_error2))
    print('Epoch: {}, train_loss_avg: {:.4f}, train_error_avg: {:.4f}'.format(epoch, train_loss_avg, train_error_avg))

    for i in range(n_classes):
        acc1, correct, count = acc_logger1.get_summary(i)
        acc2, correct, count = acc_logger2.get_summary(i)
        acc_avg, correct, count = acc_logger_avg.get_summary(i)
        print('class {}: acc1 {}, correct {}/{}'.format(i, acc1, correct, count))
        print('class {}: acc2 {}, correct {}/{}'.format(i, acc2, correct, count))
        print('class {}: acc_avg {}, correct {}/{}'.format(i, acc_avg, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc1'.format(i), acc1, epoch)
            writer.add_scalar('train/class_{}_acc2'.format(i), acc2, epoch)
            writer.add_scalar('train/class_{}_acc_avg'.format(i), acc_avg, epoch)

    if writer:
        writer.add_scalar('train/loss1', train_loss1, epoch)
        writer.add_scalar('train/error1', train_error1, epoch)

        writer.add_scalar('train/loss2', train_loss2, epoch)
        writer.add_scalar('train/error2', train_error2, epoch)

        writer.add_scalar('train/loss_avg', train_loss_avg, epoch)
        writer.add_scalar('train/error_avg', train_error_avg, epoch)

        writer.add_scalar('train/learning_rate', current_lr, epoch)
   
def validate(cur, epoch, model, loader1, loader2, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger1 = Accuracy_Logger(n_classes=n_classes)
    acc_logger2 = Accuracy_Logger(n_classes=n_classes)
    acc_logger_avg = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss1 = 0.
    val_error1 = 0.

    val_loss2 = 0.
    val_error2 = 0.

    val_loss_avg = 0.
    val_error_avg = 0.
    
    prob1 = np.zeros((len(loader1), n_classes))
    labels1 = np.zeros(len(loader1))

    prob2 = np.zeros((len(loader2), n_classes))
    labels2 = np.zeros(len(loader2))

    prob_avg = np.zeros((len(loader1), n_classes))

    with torch.no_grad():
        for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):
            data1, label1 = data1.to(device, non_blocking=True), label1.to(device, non_blocking=True)
            data2, label2 = data2.to(device, non_blocking=True), label2.to(device, non_blocking=True)

            logits1, Y_prob1, Y_hat1, _, _ = model(data1)
            acc_logger1.log(Y_hat1, label1)
            loss1 = loss_fn(logits1, label1)

            prob1[batch_idx] = Y_prob1.cpu().numpy()
            labels1[batch_idx] = label1.item()

            val_loss1 += loss1.item()
            error1 = calculate_error(Y_hat1, label1)
            val_error1 += error1

            logits2, Y_prob2, Y_hat2, _, _ = model(data2)
            acc_logger2.log(Y_hat2, label2)
            loss2 = loss_fn(logits2, label2)

            prob2[batch_idx] = Y_prob2.cpu().numpy()
            labels2[batch_idx] = label2.item()

            val_loss2 += loss2.item()
            error2 = calculate_error(Y_hat2, label2)
            val_error2 += error2

            logits_avg = (logits1 + logits2) / 2
            Y_prob_avg = (Y_prob1 + Y_prob2) / 2
            Y_hat_avg = torch.argmax(Y_prob_avg, dim=1)

            acc_logger_avg.log(Y_hat_avg, label1)
            loss_avg = loss_fn(logits_avg, label1)

            prob_avg[batch_idx] = Y_prob_avg.cpu().numpy()

            val_loss_avg += loss_avg.item() 
            error_avg = calculate_error(Y_hat_avg, label1)
            val_error_avg += error_avg


    val_error1 /= len(loader1)
    val_loss1 /= len(loader1)

    val_error2 /= len(loader2)
    val_loss2 /= len(loader2)

    val_error_avg /= len(loader1)
    val_loss_avg /= len(loader1)

    if n_classes == 2:
        auc1 = roc_auc_score(labels1, prob1[:, 1])
        auc2 = roc_auc_score(labels2, prob2[:, 1])
        auc_avg = roc_auc_score(labels1, prob_avg[:, 1])
    
    else:
        auc1 = roc_auc_score(labels1, prob1, multi_class='ovr')
        auc2 = roc_auc_score(labels2, prob2, multi_class='ovr')
        auc_avg = roc_auc_score(labels1, prob_avg, multi_class='ovr')

    
    
    if writer:
        writer.add_scalar('val/loss1', val_loss1, epoch)
        writer.add_scalar('val/auc1', auc1, epoch)
        writer.add_scalar('val/error1', val_error1, epoch)

        writer.add_scalar('val/loss2', val_loss2, epoch)
        writer.add_scalar('val/auc2', auc2, epoch)
        writer.add_scalar('val/error2', val_error2, epoch)

        writer.add_scalar('val/loss_avg', val_loss_avg, epoch)
        writer.add_scalar('val/auc_avg', auc_avg, epoch)
        writer.add_scalar('val/error_avg', val_error_avg, epoch)

    print('\nVal Set, val_loss1: {:.4f}, val_error1: {:.4f}, auc1: {:.4f}'.format(val_loss1, val_error1, auc1))
    print('Val Set, val_loss2: {:.4f}, val_error2: {:.4f}, auc2: {:.4f}'.format(val_loss2, val_error2, auc2))
    print('Val Set, val_loss_avg: {:.4f}, val_error_avg: {:.4f}, auc_avg: {:.4f}'.format(val_loss_avg, val_error_avg, auc_avg))
    for i in range(n_classes):
        acc1, correct1, count1 = acc_logger1.get_summary(i)
        acc2, correct2, count2 = acc_logger2.get_summary(i)
        acc_avg, correct_avg, count_avg = acc_logger_avg.get_summary(i)
        print('class {}: acc1 {}, correct1 {}/{}'.format(i, acc1, correct1, count1)) 
        print('class {}: acc2 {}, correct2 {}/{}'.format(i, acc2, correct2, count2))
        print('class {}: acc_avg {}, correct_avg {}/{}'.format(i, acc_avg, correct_avg, count_avg))    

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss1, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        early_stopping(epoch, val_loss2, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        early_stopping(epoch, val_loss_avg, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader1, loader2, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger1 = Accuracy_Logger(n_classes=n_classes)
    inst_logger1 = Accuracy_Logger(n_classes=n_classes)

    acc_logger2 = Accuracy_Logger(n_classes=n_classes)
    inst_logger2 = Accuracy_Logger(n_classes=n_classes)

    acc_logger_avg = Accuracy_Logger(n_classes=n_classes)
    inst_logger_avg = Accuracy_Logger(n_classes=n_classes)

    val_loss1 = 0.
    val_error1 = 0.

    val_loss2 = 0.
    val_error2 = 0.

    val_loss_avg = 0.
    val_error_avg = 0.

    val_inst_loss1 = 0.
    val_inst_acc1 = 0.
    inst_count1=0

    val_inst_loss2 = 0.
    val_inst_acc2 = 0.
    inst_count2=0

    val_inst_loss_avg = 0.
    val_inst_acc_avg = 0.
    inst_count_avg=0
    
    prob1 = np.zeros((len(loader1), n_classes))
    labels1 = np.zeros(len(loader1))

    prob2 = np.zeros((len(loader2), n_classes))
    labels2 = np.zeros(len(loader2))

    prob_avg = np.zeros((len(loader1), n_classes))
    labels = np.zeros(len(loader1))

    sample_size = model.k_sample
    with torch.inference_mode():
        for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):
            data1, label1 = data1.to(device), label1.to(device) 
            data2, label2 = data2.to(device), label2.to(device)

            logits1, Y_prob1, Y_hat1, _, instance_dict1 = model(data1, label=label1, instance_eval=True)
            logits2, Y_prob2, Y_hat2, _, instance_dict2 = model(data2, label=label2, instance_eval=True)

            acc_logger1.log(Y_hat1, label1)
            loss1 = loss_fn(logits1, label1)
            val_loss1 += loss1.item()
            instance_loss1 = instance_dict1['instance_loss']

            acc_logger2.log(Y_hat2, label2)
            loss2 = loss_fn(logits2, label2)
            val_loss2 += loss2.item()
            instance_loss2 = instance_dict2['instance_loss']

            logits_avg = (logits1 + logits2) / 2
            Y_prob_avg = (Y_prob1 + Y_prob2) / 2
            Y_hat_avg = torch.argmax(Y_prob_avg, dim=1)

            acc_logger_avg.log(Y_hat_avg, label1)
            loss_avg = loss_fn(logits_avg, label1)
            val_loss_avg += loss_avg.item()
            instance_loss_avg = (instance_dict1['instance_loss'] + instance_dict2['instance_loss']) / 2
            
            inst_count1+=1
            instance_loss_value1 = instance_loss1.item()
            val_inst_loss1 += instance_loss_value1

            inst_count2+=1
            instance_loss_value2 = instance_loss2.item()
            val_inst_loss2 += instance_loss_value2

            inst_count_avg+=1
            instance_loss_value_avg = instance_loss_avg.item()
            val_inst_loss_avg += instance_loss_value_avg

            inst_preds1 = instance_dict1['inst_preds']
            inst_labels1 = instance_dict1['inst_labels']
            inst_logger1.log_batch(inst_preds1, inst_labels1)

            inst_preds2 = instance_dict2['inst_preds']
            inst_labels2 = instance_dict2['inst_labels']
            inst_logger2.log_batch(inst_preds2, inst_labels2)


            prob1[batch_idx] = Y_prob1.cpu().numpy()
            labels1[batch_idx] = label1.item()

            prob2[batch_idx] = Y_prob2.cpu().numpy()
            labels2[batch_idx] = label2.item()

            prob_avg[batch_idx] = Y_prob_avg.cpu().numpy()
            labels[batch_idx] = label1.item()
            
            error1 = calculate_error(Y_hat1, label1)
            val_error1 += error1

            error2 = calculate_error(Y_hat2, label2)
            val_error2 += error2

            error_avg = calculate_error(Y_hat_avg, label1)
            val_error_avg += error_avg

    val_error1 /= len(loader1)
    val_loss1 /= len(loader1)

    val_error2 /= len(loader2)
    val_loss2 /= len(loader2)

    val_error_avg /= len(loader1)
    val_loss_avg /= len(loader1)

    if n_classes == 2:
        auc1 = roc_auc_score(labels1, prob1[:, 1])
        aucs1 = []

        auc2 = roc_auc_score(labels2, prob2[:, 1])
        aucs2 = []

        auc_avg = roc_auc_score(labels, prob_avg[:, 1])
        aucs_avg = []
    else:
        aucs1 = []
        aucs2 = []
        aucs_avg = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob1[:, class_idx])
                aucs1.append(calc_auc(fpr, tpr))

                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob2[:, class_idx])
                aucs2.append(calc_auc(fpr, tpr))

                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob_avg[:, class_idx])
                aucs_avg.append(calc_auc(fpr, tpr))
            else:
                aucs1.append(float('nan'))
                aucs2.append(float('nan'))
                aucs_avg.append(float('nan'))

        auc_avg = np.nanmean(np.array(aucs_avg))

    print('\nVal Set 1, val_loss1: {:.4f}, val_error1: {:.4f}, auc1: {:.4f}'.format(val_loss1, val_error1, auc1))
    print('Val Set 2, val_loss2: {:.4f}, val_error2: {:.4f}, auc2: {:.4f}'.format(val_loss2, val_error2, auc2))
    print('Val Set avg, val_loss_avg: {:.4f}, val_error_avg: {:.4f}, auc_avg: {:.4f}'.format(val_loss_avg, val_error_avg, auc_avg))

    if inst_count1 > 0:
        val_inst_loss1 /= inst_count1
        for i in range(2):
            acc1, correct1, count1 = inst_logger1.get_summary(i)
            print('class {} clustering acc1 {}: correct1 {}/{}'.format(i, acc1, correct1, count1))

    if inst_count2 > 0:
        val_inst_loss2 /= inst_count2
        for i in range(2):
            acc2, correct2, count2 = inst_logger2.get_summary(i)
            print('class {} clustering acc2 {}: correct2 {}/{}'.format(i, acc2, correct2, count2))

    if inst_count_avg > 0:
        val_inst_loss_avg /= inst_count_avg
        for i in range(2):
            acc_avg, correct_avg, count_avg = inst_logger_avg.get_summary(i)
            print('class {} clustering acc_avg {}: correct_avg {}/{}'.format(i, acc_avg, correct_avg, count_avg))       
    
    if writer:
        writer.add_scalar('val/loss1', val_loss1, epoch)
        writer.add_scalar('val/auc1', auc1, epoch)
        writer.add_scalar('val/error1', val_error1, epoch)
        writer.add_scalar('val/inst_loss1', val_inst_loss1, epoch)

        writer.add_scalar('val/loss2', val_loss2, epoch)
        writer.add_scalar('val/auc2', auc2, epoch)
        writer.add_scalar('val/error2', val_error2, epoch)
        writer.add_scalar('val/inst_loss2', val_inst_loss2, epoch)

        writer.add_scalar('val/loss_avg', val_loss_avg, epoch)
        writer.add_scalar('val/auc_avg', auc_avg, epoch)
        writer.add_scalar('val/error_avg', val_error_avg, epoch)
        writer.add_scalar('val/inst_loss_avg', val_inst_loss_avg, epoch)

    for i in range(n_classes):
        acc1, correct1, count1 = acc_logger1.get_summary(i)
        acc2, correct2, count2 = acc_logger2.get_summary(i)
        acc_avg, correct_avg, count_avg = acc_logger_avg.get_summary(i)
        print('class {}: acc1 {}, correct1 {}/{}'.format(i, acc1, correct1, count1))
        print('class {}: acc2 {}, correct2 {}/{}'.format(i, acc2, correct2, count2))
        print('class {}: acc_avg {}, correct_avg {}/{}'.format(i, acc_avg, correct_avg, count_avg))
        
        if writer and acc1 is not None:
            writer.add_scalar('val/class_{}_acc1'.format(i), acc1, epoch)
        
        if writer and acc2 is not None:
            writer.add_scalar('val/class_{}_acc2'.format(i), acc2, epoch)

        if writer and acc_avg is not None:
            writer.add_scalar('val/class_{}_acc_avg'.format(i), acc_avg, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss1, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        early_stopping(epoch, val_loss2, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        early_stopping(epoch, val_loss_avg, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader1, loader2, n_classes):
    acc_logger1 = Accuracy_Logger(n_classes=n_classes)
    acc_logger2 = Accuracy_Logger(n_classes=n_classes)
    acc_logger_avg = Accuracy_Logger(n_classes=n_classes)

    model.eval()

    test_loss1 = 0.
    test_error1 = 0.

    test_loss2 = 0.
    test_error2 = 0.

    test_loss_avg = 0.
    test_error_avg = 0.

    all_probs1 = np.zeros((len(loader1), n_classes))
    all_probs2 = np.zeros((len(loader2), n_classes))
    all_probs_avg = np.zeros((len(loader1), n_classes))

    all_labels= np.zeros(len(loader1))

    slide_ids = loader1.dataset.slide_data['slide_id']
    patient_results = {}

    for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):
        data1, label1 = data1.to(device), label1.to(device)
        data2, label2 = data2.to(device), label2.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits1, Y_prob1, Y_hat1, _, _ = model(data1)
            logits2, Y_prob2, Y_hat2, _, _ = model(data2)

        logits_avg = (logits1 + logits2) / 2
        Y_prob_avg = (Y_prob1 + Y_prob2) / 2
        Y_hat_avg = torch.argmax(Y_prob_avg, dim=1)

        # acc_logger1.log(Y_hat1, label1)
        # probs1 = Y_prob1.cpu().numpy()
        # all_probs1[batch_idx] = probs1

        # acc_logger2.log(Y_hat2, label2)
        # probs2 = Y_prob2.cpu().numpy()
        # all_probs2[batch_idx] = probs2

        acc_logger_avg.log(Y_hat_avg, label1)
        probs_avg = Y_prob_avg.cpu().numpy()
        all_probs_avg[batch_idx] = probs_avg

        all_labels[batch_idx] = label1.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob_avg': probs_avg, 'label': label1.item()}})

        error_avg = calculate_error(Y_hat_avg, label1)
        test_error_avg += error_avg

    test_error1 /= len(loader1)

    if n_classes == 2:
        auc_avg = roc_auc_score(all_labels, all_probs_avg[:, 1])
        aucs_avg = []
    else:
        aucs_avg = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs1[:, class_idx])
                aucs_avg.append(calc_auc(fpr, tpr))
            else:
                aucs_avg.append(float('nan'))

        auc_avg = np.nanmean(np.array(aucs_avg))


    return patient_results, test_error_avg, auc_avg, acc_logger_avg
