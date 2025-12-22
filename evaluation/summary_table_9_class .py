# %% IMPORT
import pandas as pd
import numpy as np

# %% LOAD CSV
file_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/9_class/9_class_summary.csv'
df = pd.read_csv(file_path)

# %% CREATE SUMMARY TABLE
# Group by the specified columns and calculate mean and std
summary = df.groupby(['Feature_Encoder', 'Aggregation', 'Fusion', 'Modality']).agg(
    BA=('BA', 'mean'),
    BA_std=('BA', 'std'),
    MCC=('MCC', 'mean'),
    MCC_std=('MCC', 'std'),
    AUC=('AUC', 'mean'),
    AUC_std=('AUC', 'std'),
    F1_Score=('F1-Score', 'mean'),
    F1_Score_std=('F1-Score', 'std')
).reset_index()

# Format the summary table
summary['BA'] = summary['BA'].round(2).astype(str) + ' ± ' + summary['BA_std'].round(2).astype(str)
summary['MCC'] = summary['MCC'].round(2).astype(str) + ' ± ' + summary['MCC_std'].round(2).astype(str)
summary['AUC'] = summary['AUC'].round(2).astype(str) + ' ± ' + summary['AUC_std'].round(2).astype(str)
summary['F1_Score'] = summary['F1_Score'].round(2).astype(str) + ' ± ' + summary['F1_Score_std'].round(2).astype(str)

# Select the final columns
summary = summary[['Feature_Encoder', 'Aggregation', 'Fusion', 'Modality', 'BA', 'MCC', 'AUC', 'F1_Score']]

# %%
# Save the summary table to a new CSV file
summary.to_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/9_class/9_class_summary_table.csv', index=False)
# %%
