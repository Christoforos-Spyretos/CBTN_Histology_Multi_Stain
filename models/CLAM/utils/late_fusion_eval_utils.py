import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_abmil import ABMIL
from models.model_fusion_mlp import MLP
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'abmil':
        model = ABMIL(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path, weights_only=True)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(datasets, args, ckpt_paths):
    models = [initiate_model(args, ckpt_path) for ckpt_path in ckpt_paths]
    print('Init Loaders')
    loaders = [get_simple_loader(dataset) for dataset in datasets]
    patient_results, test_error, auc, df, _ = summary(models, loaders, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return models, patient_results, test_error, auc, df

def summary(models, loaders, args):

    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    for model in models:
        model.eval()

    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loaders[0]), args.n_classes))
    all_labels = np.zeros(len(loaders[0]))
    all_preds = np.zeros(len(loaders[0]))

    slide_ids = loaders[0].dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, *data_batches in zip(range(len(loaders[0])), *loaders):
        data = [data_batch[0].to(device) for data_batch in data_batches]
        labels = [data_batch[1].to(device) for data_batch in data_batches]

        data1, label1 = data1.to(device), label1.to(device)
        data2, label2 = data2.to(device), label2.to(device)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits = [model(data[i]) for i, model in enumerate(models)]
            Y_probs = [F.softmax(logits[i][1], dim=1) for i in range(len(models))]
            Y_hats = [torch.argmax(Y_probs[i], dim=1) for i in range(len(models))]

            # late fusion
            Y_prob, Y_hat = late_fusion(Y_probs, Y_hats, args)
            
        acc_logger.log(Y_hat, label1)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label1.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label1.item()}})
        
        error = calculate_error(Y_hat, label1)
        test_error += error

    del data
    test_error /= len(loaders[0])

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

def late_fusion(Y_probs, Y_hats, args):
            if args.late_fusion_method == 'aggregation_mean_logits':
                logits = sum(logit[0] for logit in Y_probs) / len(Y_probs)
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif args.late_fusion_method == 'aggregation_mean_probs':
                Y_prob = sum(Y_prob[0] for Y_prob in Y_probs) / len(Y_probs)
                Y_hat = torch.argmax(Y_prob, dim=1)

            elif args.late_fusion_method == 'majority_voting':
                votes = torch.stack(Y_hats, dim=0)  
                # get the mode (majority vote) across all models
                Y_hat, counts = torch.mode(votes, dim=0) 
                Y_prob = torch.zeros_like(Y_probs[0][0])  

                # for each sample, assign probabilities based on the majority vote
                for i in range(len(Y_hat)):
                    winning_model_idx = (votes[:, i] == Y_hat[i]).nonzero(as_tuple=True)[0]
                    Y_prob[i] = Y_probs[winning_model_idx[0]][i] 

                # tie-breaking using the model with the highest confidence
                if counts.size(0) > 1 and counts[0] == counts[1]:  
                    confidence = [torch.max(Y_probs[j][i], dim=1)[0] for j in range(len(Y_probs))]
                    winning_model_idx = torch.argmax(torch.stack(confidence), dim=0) 
                    for i in range(len(Y_hat)):
                        Y_prob[i] = Y_probs[winning_model_idx[i]][i]

            if args.late_fusion_method == 'mlp': 
                concatenated_features = torch.cat([logit[0] for logit in Y_probs], dim=1)
                model = MLP(input_dim=concatenated_features.shape[-1], hidden_dim=64, n_classes=args.n_classes)
                model = model.to(device)
                logits = model(concatenated_features)
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.argmax(Y_prob, dim=1)