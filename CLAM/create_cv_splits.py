# %% IMPORTS
import pandas as pd
import warnings
from pathlib import Path

from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
)

# %%
def get_repetition_split(df, random_seed: int = 29122009, print_summary: bool = False, class_stratification = True, validation_fraction = 0.2, test_fraction = 0.3, number_of_folds = 5):
    """
    Utility that splits the slide_ids in the df using a per case_id split (subject wise-splitting).
    It applies label and/or site stratification is requested.

    INPUT
        df : pandas Dataframe.
            Dataframe with the case_id, slide_id, label and site (if requested) information.
        random_seed : int
            Seeds the random split

    OUTPUT
        df : pandas Dataframe
            Dataframe with each of the slide_id as training, val or test for each of the specified folds.
    """

    if print_summary:
        # print summary before start splitting
        print_df_summary(df)

    # ################## work on splitting
    if class_stratification:
        # ################## TEST SET
        # perform stratified split
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / test_fraction),
            shuffle=True,
            random_state=random_seed,
        )

        train_val_ix, test_ix = next(sgkf.split(df, y=df.label, groups=df.case_id))

        # get testing set
        df_test_split = df.loc[test_ix].reset_index()
        if print_summary:
            print(
                f'{"Test set":9s}: {len(test_ix):5d} {"test":10} files ({len(pd.unique(df_test_split.case_id)):4d} unique subjects ({pd.unique(df_test_split.label)} {[len(pd.unique(df_test_split.loc[df_test_split.label == c].case_id)) for c in list(pd.unique(df_test_split.label))]}))'
            )
        # get train_val set
        df_train_val_split = df.loc[train_val_ix].reset_index()
        # make a copy of the df_train_val_split to use as back bone for the dataframe to be returned (add the test at the end)
        dataset_split_df = df_train_val_split.copy()

        # ################# TRAINING and VALIDATION SETs
        # build nbr_splits. This is needed in case the cfg.number_of_folds and cfg.validation_fraction is provided
        if number_of_folds == 1:
            if validation_fraction is not None:
                n_splits = int(1 / validation_fraction)
            else:
                n_splits = 2
        else:
            n_splits = number_of_folds

        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

        # if only one internal fold is requested, do as in the testing. Else,
        # get all the folds (just use next as many times as the one requested by nbr of folds)
        for cv_f, (train_ix, val_ix) in enumerate(
            sgkf.split(
                df_train_val_split,
                groups=df_train_val_split.case_id,
                y=df_train_val_split.label,
            )
        ):
            # add a column in the dataset_split_df and flag all the files based on the split
            dataset_split_df[f"fold_{cv_f+1}"] = "train"
            # flag the training files
            dataset_split_df.loc[val_ix, f"fold_{cv_f+1}"] = "validation"

            # add to the df_test_split the flag for this fold
            df_test_split[f"fold_{cv_f+1}"] = "test"

            # check that there are elements in the training, val and test for each of the classes
            for s_ix, s in zip((train_ix, val_ix), ("train", "validation")):
                aus_df = df_train_val_split.loc[s_ix]
                # get nbr. unique subjects per class
                classes = list(pd.unique(df.label))
                classes.sort()
                per_class_nbr_subjs = [
                    len(pd.unique(aus_df.loc[aus_df.label == c].case_id))
                    for c in list(pd.unique(df.label))
                ]
                # if any of the classes has nbr_subjs == 0, raise warning
                if any([i == 0 for i in per_class_nbr_subjs]):
                    warnings.warn(
                        f"Some of the classes in {s} set have nbr_subjs == 0 (fold=={cv_f}).\n   Unique classes: {list(pd.unique(df.label))}\n   Unique subjects: {per_class_nbr_subjs}"
                    )

            # print summary
            if print_summary:
                aus_df = df_train_val_split.loc[train_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(train_ix):5d} {"training":10} files ({len(pd.unique(df_train_val_split.loc[train_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )
                aus_df = df_train_val_split.loc[val_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(val_ix):5d} {"validation":10} files ({len(pd.unique(df_train_val_split.loc[val_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )

            if cv_f + 1 == number_of_folds:
                break

    # finish up the dataset_split_df by merging the df_test_split
    dataset_split_df = pd.concat(
        [dataset_split_df, df_test_split], ignore_index=True
    ).reset_index(drop=True)

    return dataset_split_df


def print_df_summary(df):
    # print totals first
    print(f"Number of slides: {len(df)}")
    print(f"Number of unique case_ids (subjects): {len(pd.unique(df.case_id))}")
    if "site_id" in df.columns:
        print(f"Number of sites: {len(pd.unique(df.site_id))}")
    print(f"Number of unique classes/labels: {len(pd.unique(df.label))}")

    # break down on a class level
    if "site_id" in df.columns:
        aus = df.groupby(["label"]).agg(
            {
                "case_id": lambda x: len(pd.unique(x)),
                "slide_id": lambda x: len(x),
                "site_id": lambda x: len(pd.unique(x)),
            }
        )
    else:
        aus = df.groupby(["label"]).agg(
            {"case_id": lambda x: len(pd.unique(x)), "slide_id": lambda x: len(x)}
        )
    print(aus)

def save_for_clam(df, split_strategy, output_dir, experiment_name, number_of_folds = 5):
    """
    See README.md file for a detailed description of how the CLAM framework needs the .csv files saved for training and evaluation (tune).
    """

    # make save_path
    if split_strategy == "cv":
        save_path = Path(
            output_dir,
            experiment_name,
        )
    elif split_strategy == "npb":
        save_path = Path(
            output_dir,
            experiment_name,
        )
    else:
        raise NotImplemented

    save_path.mkdir(parents=True, exist_ok=True)

    for fold in range(number_of_folds):
        # # make and save split_nbr.csv file
        df_for_save = df[[f"fold_{fold+1}", "slide_id"]]
        df_for_save = df_for_save.rename(columns={f"fold_{fold+1}": "set"})

        # combine and take out each set
        gb = df_for_save.groupby(["set"])
        train = (
            gb.get_group("train")
            .drop(columns=["set"])
            .rename(columns={"slide_id": "train"})
            .reset_index()
        )
        validation = (
            gb.get_group("validation")
            .drop(columns=["set"])
            .rename(columns={"slide_id": "val"})
            .reset_index()
        )
        test = (
            gb.get_group("test")
            .drop(columns=["set"])
            .rename(columns={"slide_id": "test"})
            .reset_index()
        )

        # concatenate and save
        split_to_save = pd.concat([train, validation, test], axis=1).drop(
            columns=["index"]
        )
        if split_strategy == "npb":
            split_to_save.to_csv(
                Path(save_path, f"splits_{fold}.csv"),
                index_label=False,
                index=False,
            )
        else:
            split_to_save.to_csv(
                Path(save_path, f"splits_{fold}.csv"), index_label=False, index=False
            )

        # # make and save split_nbr_bool.csv file
        df_for_save["train"] = df_for_save.apply(lambda x: x.set == "train", axis=1)
        df_for_save["val"] = df_for_save.apply(lambda x: x.set == "validation", axis=1)
        df_for_save["test"] = df_for_save.apply(lambda x: x.set == "test", axis=1)

        # refine and save
        split_bool_to_save = df_for_save.drop(columns=["set"])
        if split_strategy == "npb":
            split_bool_to_save.to_csv(
                Path(save_path, f"splits_{fold}_bool.csv"),
                index_label=False,
                index=False,
            )
        else:
            split_bool_to_save.to_csv(
                Path(save_path, f"splits_{fold}_bool.csv"),
                index_label=False,
                index=False,
            )

        # # make and save split_nbr_descriptor.csv file
        df_for_save = df[[f"fold_{fold+1}", "slide_id", "label"]]
        df_for_save = df_for_save.rename(columns={f"fold_{fold+1}": "set"})

        # create dummy columns needed for later
        df_for_save["train"] = df_for_save.apply(lambda x: x.set == "train", axis=1)
        df_for_save["val"] = df_for_save.apply(lambda x: x.set == "validation", axis=1)
        df_for_save["test"] = df_for_save.apply(lambda x: x.set == "test", axis=1)

        gb = df_for_save.groupby(["label"]).agg(
            {"train": "sum", "val": "sum", "test": "sum"}
        )

        if split_strategy == "npb":
            gb.to_csv(
                Path(save_path, f"splits_{fold}_descriptor.csv"),
                index_label="class",
                index=True,
            )
        else:
            gb.to_csv(
                Path(save_path, f"splits_{fold}_descriptor.csv"),
                index_label="class",
                index=True,
            )  

# %%
# %%
csv_file = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset.csv')
save_folds = '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits'

# %%
folds = get_repetition_split(df=csv_file)

save_for_clam(folds, split_strategy='cv', output_dir=save_folds,experiment_name='HE_7_class_tumor_subtyping_100')

# %%