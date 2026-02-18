# %% IMPORTS
import pandas as pd
import os

# %% CREATE CSV FOR HE LGG vs HGG HEATMAPS
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_LGG_vs_HGG_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1_5/fold_15.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)
output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/HE_LGG_vs_HGG_heatmaps.csv'

correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]
heatmap_df.to_csv(output_path, index=False)

# %% CREATE CSV FOR KI67 LGG vs HGG HEATMAPS SVS
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5/fold_15.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)
path_to_svs_images = '/run/media/chrsp39/Expansion/CBTN_v2/KI67/WSI_svs'
output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_heatmaps_svs.csv'

correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]

# Filter for slides that exist as .svs files
available_svs_files = set([f.replace('.svs', '') for f in os.listdir(path_to_svs_images) if f.endswith('.svs')])
heatmap_df = heatmap_df[heatmap_df['slide_id'].isin(available_svs_files)]

heatmap_df.to_csv(output_path, index=False)

# %% CREATE CSV FOR KI67 LGG vs HGG HEATMAPS TIF
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5/fold_15.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)
path_to_tif_images = '/run/media/chrsp39/Expansion/CBTN_v2/KI67/WSI_tif'
output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_heatmaps_tif.csv'

correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]

# Filter for slides that exist as .tif files
available_tif_files = set([f.replace('.tif', '') for f in os.listdir(path_to_tif_images) if f.endswith('.tif')])
heatmap_df = heatmap_df[heatmap_df['slide_id'].isin(available_tif_files)]

heatmap_df.to_csv(output_path, index=False)

# %% CREATE CSV FOR HE 5 CLASS HEATMAPS
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1_5/fold_12.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)

output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_heatmaps.csv'
correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]
heatmap_df.to_csv(output_path, index=False)

# %% CREATE CSV FOR KI67 5 CLASS HEATMAPS SVS
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1_5/fold_12.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)
path_to_svs_images = '/run/media/chrsp39/Expansion/CBTN_v2/KI67/WSI_svs'
output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_heatmaps_svs.csv'

correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]

# Filter for slides that exist as .svs files
available_svs_files = set([f.replace('.svs', '') for f in os.listdir(path_to_svs_images) if f.endswith('.svs')])
heatmap_df = heatmap_df[heatmap_df['slide_id'].isin(available_svs_files)]

heatmap_df.to_csv(output_path, index=False)

# %% CREATE CSV FOR KI67 5 CLASS HEATMAPS TIF
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv')
results_df = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1_5/fold_12.csv')
results_df.rename(columns={'slide_id': 'case_id'}, inplace=True)
path_to_tif_images = '/run/media/chrsp39/Expansion/CBTN_v2/KI67/WSI_tif'
output_path = '/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_heatmaps_tif.csv'

correct_predictions = results_df[results_df['Y'] == results_df['Y_hat']].copy()
heatmap_df = correct_predictions.merge(df[['case_id', 'slide_id', 'label']], on='case_id', how='left')
heatmap_df = heatmap_df[['case_id', 'slide_id', 'label']]

# Filter for slides that exist as .tif files
available_tif_files = set([f.replace('.tif', '') for f in os.listdir(path_to_tif_images) if f.endswith('.tif')])
heatmap_df = heatmap_df[heatmap_df['slide_id'].isin(available_tif_files)]

heatmap_df.to_csv(output_path, index=False)

# %%