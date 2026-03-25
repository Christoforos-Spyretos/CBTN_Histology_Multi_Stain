# %% IMPORTS
import os

import pandas as pd


# %% LOAD PATHS
CLASS_5_CSV = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv'
LGG_HGG_CSV = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv'

LGG_vs_HGG_70_10_20 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_LGG_vs_HGG_0.7_0.1_0.2_100'
class_5_70_10_20 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_5_class_0.7_0.1_0.2_100'
LGG_vs_HGG_50_20_30 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_LGG_vs_HGG_0.5_0.2_0.3_100'
class_5_50_20_30 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/splits/Merged_HE_KI67_5_class_0.5_0.2_0.3_100'

default_keep_train_frac = 0.5
default_n_folds = 50

jobs = [
	('LGG_vs_HGG_70_10_20', LGG_vs_HGG_70_10_20, LGG_HGG_CSV),
	('class_5_70_10_20', class_5_70_10_20, CLASS_5_CSV),
	('LGG_vs_HGG_50_20_30', LGG_vs_HGG_50_20_30, LGG_HGG_CSV),
	('class_5_50_20_30', class_5_50_20_30, CLASS_5_CSV),
]


def _build_label_map(label_df):
	return dict(zip(label_df['case_id'].astype(str), label_df['label'].astype(str)))


def _stratified_keep(train_ids, label_map, keep_train_frac, seed):
	train_series = pd.Series(train_ids, name='case_id', dtype='object')
	train_labels = train_series.map(label_map)

	missing_labels = train_series[train_labels.isna()].tolist()
	if missing_labels:
		raise ValueError(f'Missing labels for {len(missing_labels)} training case(s): {missing_labels[:5]}')

	keep_ids = []
	train_df = pd.DataFrame({'case_id': train_series, 'label': train_labels})

	for _, group in train_df.groupby('label'):
		n_total = len(group)
		n_keep = max(1, int(n_total * keep_train_frac))
		kept_group = group.sample(n=n_keep, random_state=seed, replace=False)
		keep_ids.extend(kept_group['case_id'].tolist())

	return keep_ids


def _build_split_dataframe(train_ids, val_ids, test_ids):
	max_len = max(len(train_ids), len(val_ids), len(test_ids))

	def _pad(values):
		return values + [None] * (max_len - len(values))

	return pd.DataFrame({'train': _pad(train_ids), 'val': _pad(val_ids), 'test': _pad(test_ids)})


def _build_descriptor_dataframe(train_ids, val_ids, test_ids, label_map, descriptor_index_order):
	def _counts(ids):
		labels = pd.Series(ids, dtype='object').map(label_map)
		return labels.value_counts()

	train_counts = _counts(train_ids)
	val_counts = _counts(val_ids)
	test_counts = _counts(test_ids)

	labels_in_counts = set(train_counts.index) | set(val_counts.index) | set(test_counts.index)
	ordered_labels = [label for label in descriptor_index_order if label in labels_in_counts]
	ordered_labels += sorted(labels_in_counts - set(ordered_labels))

	descriptor_df = pd.DataFrame(index=ordered_labels, columns=['train', 'val', 'test'])
	descriptor_df['train'] = train_counts.reindex(ordered_labels, fill_value=0).astype(int)
	descriptor_df['val'] = val_counts.reindex(ordered_labels, fill_value=0).astype(int)
	descriptor_df['test'] = test_counts.reindex(ordered_labels, fill_value=0).astype(int)
	return descriptor_df


def drop_training_data_for_splits(source_dir, label_df, keep_train_frac=default_keep_train_frac, n_folds=default_n_folds):
	label_map = _build_label_map(label_df)
	target_dir = f'{source_dir}_{keep_train_frac}_training_drop'
	os.makedirs(target_dir, exist_ok=True)

	for fold_idx in range(n_folds):
		bool_path = os.path.join(source_dir, f'splits_{fold_idx}_bool.csv')
		descriptor_path = os.path.join(source_dir, f'splits_{fold_idx}_descriptor.csv')

		bool_df = pd.read_csv(bool_path, index_col=0)
		descriptor_df = pd.read_csv(descriptor_path, index_col=0)

		train_ids = bool_df.index[bool_df['train'] == True].astype(str).tolist()
		val_ids = bool_df.index[bool_df['val'] == True].astype(str).tolist()
		test_ids = bool_df.index[bool_df['test'] == True].astype(str).tolist()

		kept_train_ids = _stratified_keep(train_ids, label_map, keep_train_frac=keep_train_frac, seed=fold_idx)

		out_bool_df = bool_df.copy()
		out_bool_df['train'] = out_bool_df.index.astype(str).isin(kept_train_ids)
		out_split_df = _build_split_dataframe(kept_train_ids, val_ids, test_ids)
		out_descriptor_df = _build_descriptor_dataframe(
			kept_train_ids,
			val_ids,
			test_ids,
			label_map,
			descriptor_index_order=descriptor_df.index.astype(str).tolist(),
		)

		out_bool_df.to_csv(os.path.join(target_dir, f'splits_{fold_idx}_bool.csv'))
		out_split_df.to_csv(os.path.join(target_dir, f'splits_{fold_idx}.csv'))
		out_descriptor_df.to_csv(os.path.join(target_dir, f'splits_{fold_idx}_descriptor.csv'))

	return target_dir


def _run_selected_jobs(selected_jobs, keep_train_frac, n_folds):
	created_dirs = []

	for _, source_dir, label_csv in selected_jobs:
		label_df = pd.read_csv(label_csv)
		created_dir = drop_training_data_for_splits(
			source_dir=source_dir,
			label_df=label_df,
			keep_train_frac=keep_train_frac,
			n_folds=n_folds,
		)
		created_dirs.append(created_dir)

	return created_dirs


# %% INTERACTIVE CONFIG (edit and run this cell)
keep_train_frac = 0.5
n_folds = 50

# Options: LGG_vs_HGG_70_10_20, class_5_70_10_20, LGG_vs_HGG_50_20_30, class_5_50_20_30
selected_job_names = [
	'LGG_vs_HGG_70_10_20',
	'class_5_70_10_20',
	'LGG_vs_HGG_50_20_30',
	'class_5_50_20_30',
]


# %% RUN IN VS CODE INTERACTIVE WINDOW
selected_jobs = [job for job in jobs if job[0] in selected_job_names]
if not selected_jobs:
	raise ValueError('No valid jobs selected. Check selected_job_names.')

created_dirs = _run_selected_jobs(selected_jobs, keep_train_frac, n_folds)
print('Created split directories:')
for path in created_dirs:
	print(path)

# %%