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

def eval(dataset1, dataset2, args, ckpt_path1, ckpt_path2):
    model1 = initiate_model(args, ckpt_path1)
    model2 = initiate_model(args, ckpt_path2)
    print('Init Loaders')
    loader1 = get_simple_loader(dataset1)
    loader2 = get_simple_loader(dataset2)
    patient_results, test_error, auc, df, _ = summary(model1, model2, loader1, loader2, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model1, model2, patient_results, test_error, auc, df

def summary(model1, model2, loader1, loader2, args):

    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    model1.eval()
    model2.eval()

    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader1), args.n_classes))
    all_labels = np.zeros(len(loader1))
    all_preds = np.zeros(len(loader1))

    slide_ids = loader1.dataset.slide_data['slide_id']
    patient_results = {}

    for (batch_idx, ((data1, label1), (data2, label2))) in enumerate(zip(loader1, loader2)):

        data1, label1 = data1.to(device), label1.to(device)
        data2, label2 = data2.to(device), label2.to(device)


        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits1, Y_prob1, Y_hat1, _, results_dict = model1(data1)
            logits2, Y_prob2, Y_hat2, _, results_dict = model2(data2)

            if args.late_fusion_method == 'aggregation_mean_logits':

                logits = (logits1 + logits2) / 2
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.argmax(Y_prob, dim=1)

            if args.late_fusion_method == 'aggregation_mean_probs':

                Y_prob = (Y_prob1 + Y_prob2) / 2
                Y_hat = torch.argmax(Y_prob, dim=1)

            if args.late_fusion_method == 'majority_voting':

                votes = torch.stack([Y_hat1, Y_hat2], dim=0)  
                Y_hat, counts = torch.mode(votes, dim=0)  # perform majority voting

                Y_prob = torch.zeros_like(Y_prob1)

                # assign Y_prob based on the final prediction from Y_hat
                for i in range(len(Y_hat)):
                    if Y_hat[i] == Y_hat1[i]:
                        Y_prob[i] = Y_prob1[i]  
                    else:
                        Y_prob[i] = Y_prob2[i] 

                # if a tie happens when both classes have equal votes 
                # get the maximum probability for each models
                if counts.size(0) > 1 and counts[0] == counts[1]:  
                    confidence1, _ = torch.max(Y_prob1, dim=1)  
                    confidence2, _ = torch.max(Y_prob2, dim=1) 
                    Y_prob = torch.where(confidence1 > confidence2, Y_prob1, Y_prob2)
                    Y_hat = torch.argmax(Y_prob, dim=1)

            if args.late_fusion_method == 'mlp': 

                concatenated_features = torch.cat([logits1, logits2], dim=1)
                model = MLP(input_dim=concatenated_features.shape[-1], hidden_dim=64, n_classes=args.n_classes)
                model = model.to(device)
                logits = model(concatenated_features)
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.argmax(Y_prob, dim=1)
            
        acc_logger.log(Y_hat, label1)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label1.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label1.item()}})
        
        error = calculate_error(Y_hat, label1)
        test_error += error

    del data1, data2
    test_error /= len(loader1)

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
