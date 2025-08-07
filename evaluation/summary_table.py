# %% IMPORT
import pandas as pd
import numpy as np

# %% LOAD CSV
file_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/5_class_summary.csv'
df = pd.read_csv(file_path)

# %% CREATE SUMMARY TABLE
# Group by the specified columns and calculate mean and std
summary = df.groupby(['Feature_Extractor', 'Aggregation_Method', 'Fusion', 'Modality']).agg(
    Balanced_Accuracy=('Balanced_Accuracy', 'mean'),
    Balanced_Accuracy_std=('Balanced_Accuracy', 'std'),
    MCC=('MCC', 'mean'),
    MCC_std=('MCC', 'std'),
    AUC=('AUC', 'mean'),
    AUC_std=('AUC', 'std'),
    F1_Score=('F1-Score', 'mean'),
    F1_Score_std=('F1-Score', 'std')
).reset_index()

# Format the summary table
summary['Balanced_Accuracy'] = summary['Balanced_Accuracy'].round(2).astype(str) + ' ± ' + summary['Balanced_Accuracy_std'].round(2).astype(str)
summary['MCC'] = summary['MCC'].round(2).astype(str) + ' ± ' + summary['MCC_std'].round(2).astype(str)
summary['AUC'] = summary['AUC'].round(2).astype(str) + ' ± ' + summary['AUC_std'].round(2).astype(str)
summary['F1_Score'] = summary['F1_Score'].round(2).astype(str) + ' ± ' + summary['F1_Score_std'].round(2).astype(str)

# Select the final columns
summary = summary[['Feature_Extractor', 'Aggregation_Method', 'Fusion', 'Modality', 'Balanced_Accuracy', 'MCC', 'AUC', 'F1_Score']]

# %%
# Save the summary table to a new CSV file
summary.to_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/5_class_summary_table.csv', index=False)
# %%
