# %% IMPORT
import pandas as pd
import numpy as np

# %% LOAD CSV
file_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_summary.csv'
df = pd.read_csv(file_path)

# %% CREATE SUMMARY TABLE
# group by the specified columns and calculate mean, std, and count
summary = df.groupby(['Feature_Encoder', 'Aggregation', 'Fusion', 'Modality']).agg(
    BA=('BA', 'mean'),
    BA_std=('BA', 'std'),
    BA_count=('BA', 'count'),
    MCC=('MCC', 'mean'),
    MCC_std=('MCC', 'std'),
    MCC_count=('MCC', 'count'),
    AUC=('AUC', 'mean'),
    AUC_std=('AUC', 'std'),
    AUC_count=('AUC', 'count'),
    F1_Score=('F1-Score', 'mean'),
    F1_Score_std=('F1-Score', 'std'),
    F1_Score_count=('F1-Score', 'count')
).reset_index()

# calculate 95% CI 
def format_metric_with_ci(mean, std, count):
    ci = 1.96 * (std / np.sqrt(count))
    lower = mean - ci
    upper = mean + ci
    return f"{mean:.2f} ± {std:.2f} ({lower:.2f}-{upper:.2f})"

summary['BA'] = summary.apply(lambda row: format_metric_with_ci(row['BA'], row['BA_std'], row['BA_count']), axis=1)
summary['MCC'] = summary.apply(lambda row: format_metric_with_ci(row['MCC'], row['MCC_std'], row['MCC_count']), axis=1)
summary['AUC'] = summary.apply(lambda row: format_metric_with_ci(row['AUC'], row['AUC_std'], row['AUC_count']), axis=1)
summary['F1_Score'] = summary.apply(lambda row: format_metric_with_ci(row['F1_Score'], row['F1_Score_std'], row['F1_Score_count']), axis=1)

# columns to include in the summary table
summary = summary[['Feature_Encoder', 'Aggregation', 'Fusion', 'Modality', 'BA', 'MCC', 'AUC', 'F1_Score']]

# save the summary table
summary.to_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_summary_table.csv', index=False)

# %%
