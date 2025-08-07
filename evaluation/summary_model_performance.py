# %%
import pandas as pd
from tqdm import tqdm
import math
import pickle
import sklearn.metrics as metrics
import os
import glob
import numpy as np
import itertools
import pathlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import json
from datetime import datetime

# local imports
from utils import (
    plotConfusionMatrix,
    get_performance_metrics,
    plotROC,
    get_confusion_matrix,
)

# %% UTILITIES


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, "*"))


# %% PATHS

CLASSIFICATION_TASK_EVALUATION_FOLDER = (
    "/Users/iulta54/Desktop/Projects/temp_plot_CS/raw_evaluation_results/LGG_vs_HGG"
)

AGGREGATED_FILE_SAVE_PATH = Path("/Users/iulta54/Desktop/Projects/temp_plot_CS/output")
AGGREGATED_FILE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# %% LOOP THROUGH THE DIFFERENT FOLDER RESULTS, GATHER INFORMATION AND SDD TO THE PROCESS STACK

process_stack = []
# looping through the different multimodal approaches (input configuration)
for multimodal_level_experiments in listdir_nohidden(
    CLASSIFICATION_TASK_EVALUATION_FOLDER
):
    # looping through the different feature extractors
    for feature_extractor in listdir_nohidden(multimodal_level_experiments):
        # loop through the different experiments
        for experiment in listdir_nohidden(feature_extractor):
            # gather information about this experiment and save path for later processing
            # # more general info
            classification_task = os.path.basename(
                CLASSIFICATION_TASK_EVALUATION_FOLDER
            )
            multimodal_approach = os.path.basename(multimodal_level_experiments)
            feature_extractor = os.path.basename(feature_extractor)
            # # specific experiment info
            experiment_name = os.path.basename(experiment)
            model = "clam"  # here assuming that clam is used. Should be changed in the future
            aggregator = "_".join(experiment_name.split("_")[-3:-1])
            input_modalities = experiment_name.split("Merged_")[-1].split("_small")[0]

            # save information to process stack
            process_stack.append(
                {
                    "experiment_path": experiment,
                    "classification_task": classification_task,
                    "multimodal_approach": multimodal_approach,
                    "feature_extractor": feature_extractor,
                    "model": model,
                    "aggregator": aggregator,
                    "input_modalities": input_modalities,
                }
            )

print(f"Found {len(process_stack)} models to evaluate.")

# %% EVALUATE
for_aggregation = []

# % loop through the process stack and evaluate
with tqdm(total=len(process_stack), unit="experiment") as model_tqdm:
    for experiment in process_stack:
        trained_model_dir = experiment["experiment_path"]
        # create save path
        SAVE_PATH = Path(trained_model_dir, "summary_evaluation")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        # get a list of all the folds to evaluate
        per_fold_csv_path = glob.glob(os.path.join(trained_model_dir, "fold_*.csv"))

        # define metrics to aggregate
        prec_lists = []
        rec_lists = []
        f1_lists = []
        averaged_prec = []
        averaged_rec = []
        averaged_f1 = []
        averaged_AUC_list = []
        acc_list = []
        mcc_list = []
        class_acc_list = []
        balanced_acc = []
        cumulative_confusion_matrix = []

        with tqdm(total=len(per_fold_csv_path), unit="folds", leave=False) as fold_tqdm:
            for fold_csv in per_fold_csv_path:
                # open fold csv file
                res_df = pd.read_csv(fold_csv)

                # get logits, prediction and labels
                probs = np.stack(
                    np.array(res_df.drop(columns=["slide_id", "Y", "Y_hat"]))
                )
                preds = np.argmax(probs, axis=-1)
                labels = np.stack(np.array(res_df["Y"])).astype(int)

                # # make one hot label encoding
                one_hot_labels = np.zeros_like(probs)
                one_hot_labels[np.arange(one_hot_labels.shape[0]), labels] = 1

                # get performance metrics
                GT = one_hot_labels
                PRED = probs
                metric_dict = get_performance_metrics(GT, PRED, average="macro")

                # save for aggregation
                prec_lists.append(metric_dict["precision"])
                rec_lists.append(metric_dict["recall"])
                f1_lists.append(metric_dict["f1-score"])
                averaged_prec.append(metric_dict["overall_precision"])
                averaged_rec.append(metric_dict["overall_recall"])
                averaged_f1.append(metric_dict["overall_f1-score"])
                class_acc_list.append(metric_dict["accuracy"])
                acc_list.append(np.mean(metric_dict["overall_accuracy"]))
                mcc_list.append(metric_dict["matthews_correlation_coefficient"])
                averaged_AUC_list.append(metric_dict["overall_auc"])
                balanced_acc.append(metric_dict["balanced_accuracy"])

                fold_tqdm.update()

        # % SAVE SUMMARY IN A .csv FILE THAT CAN BE AGGREGATED WITH ALL THE OTHER RUNS

        summary_df = []

        # gather information
        for f in range(len(per_fold_csv_path)):
            temp_dict = {
                "multimodal_approach": experiment["multimodal_approach"],
                "model": experiment["model"],
                "aggregation": experiment["aggregator"],
                "features": experiment["feature_extractor"],
                "input_configuration": experiment["input_modalities"],
                "task": experiment["classification_task"],
                "nbr_classes": GT.shape[-1],
                "repetition": 1,
                "fold": f,
                "set": "test",
                "mcc": mcc_list[f],
                "balanced_accuracy": balanced_acc[f],
                "accuracy": acc_list[f],
                "auc": averaged_AUC_list[f],
                "f1-score": averaged_f1[f],
            }

            summary_df.append(temp_dict)

        # make dataframe
        summary_df = pd.DataFrame(summary_df)

        # save
        summary_df.to_csv(os.path.join(SAVE_PATH, "summary_evaluation.csv"))

        # save for aggregation
        if len(process_stack) > 1:
            for_aggregation.append(summary_df)

        model_tqdm.update()

# %% SAVE AGGREGATED ACCROSS EXPERIMENTS
summary_evaluation_df = pd.concat(for_aggregation, axis=0, ignore_index=True)
aggregated_file_path = os.path.join(
    AGGREGATED_FILE_SAVE_PATH,
    f'aggregated_evaluation_{datetime.now().strftime("%Y%m%d")}.csv',
)
summary_evaluation_df.to_csv(aggregated_file_path)
print(f"Aggregated file save as: {aggregated_file_path}")
