# imports
from utils.utils import print_network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# internal import 
from utils.core_utils import Accuracy_Logger
from utils.utils import *
from intermediate_fusion.classifier import Classifier

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')
    # Build model_dict from args/config
    model_dict = {
        'embed_dim': args.embed_dim,
        'n_classes': args.n_classes,
    }
    model = Classifier(**model_dict)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    print_network(model)

    # Optionally, you can clean the checkpoint if needed, but for now just load as above
    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_logits = np.zeros((len(loader), args.n_classes))
    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    print("Loader length:", len(loader))
    
   # If the dataset is a tuple, unpack it
    if isinstance(loader.dataset, tuple):
        for idx, dataset in enumerate(loader.dataset):
            print(f"Dataset {idx}: {dataset}")
            break

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):

        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        
        with torch.no_grad():
            logits, Y_prob, Y_hat = model(data)
        
        acc_logger.log(Y_hat, label)
        
        logits = logits.cpu().numpy()
        probs = Y_prob.cpu().numpy()

        all_logits[batch_idx] = logits
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item(), 'logits': logits}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger