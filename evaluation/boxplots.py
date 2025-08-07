# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% LOAD CSV
summary_results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/LGG_vs_HGG_summary.csv'
summary_results = pd.read_csv(summary_results_path)

# %%
# Feature_Extractor: uni, conch 
# Aggregation_Method: small_clam_sb
# Fusion = Single_Stain, Early_Fusion, Late_Fusion_agg_probs, Late_Fusion_agg_logits, Late_Fusion_majority_voting, Late_Fusion_simple_model_mlp
# Modality = HE, KI67, GFAP, HE_KI67, HE_GFAP, HE_KI67_GFAP
# Balanced_Accuracy,
# MCC,
# AUC,
# F1_Score

# %% FILTERING
# pick the Feature_Extractor interested in
summary_results = summary_results[summary_results['Feature_Extractor'] == 'conch']

# %% SUMMARY
# combine the Fusion + Modality + Feature_Extraction columns into one column
summary_results['Configuration'] = summary_results['Fusion'] + '_' + summary_results['Modality'] + '_' + summary_results['Aggregation_Method']

# %% BOX PLOT
# set the colours
palette = sns.color_palette("deep", 10)  
custom_colors = [palette[0], palette[1], palette[2], palette[4], palette[6], palette[8]]
sns.set_style("darkgrid", {"grid.color": ".6"})

# boxplot
plt.figure(figsize=(15, 10))
plt.gca().set_facecolor('white')  # set the background of the plot to pure white

# set the outlier properties
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')

boxplot = sns.boxplot(x='Configuration', y='Balanced_Accuracy', data=summary_results, hue="Fusion", palette=custom_colors, flierprops=flierprops)
# legend = plt.legend(loc='upper left')  # remove title from legend box
# legend.get_frame().set_facecolor('white')  # set legend box background color to white

# rename legend labels
# new_legend_labels = ['Single Stain', 'Early Fusion', 'Aggregation of Probabilities', 'Aggregation of Logits', 'Majority Voting', 'Single-Layer Network']
# for t, l in zip(legend.texts, new_legend_labels):
#     t.set_text(l)
plt.legend().remove()  # correct method to remove the legend

plt.xlabel('')
plt.ylabel('')
plt.ylim(0.45, 1)
plt.xticks(rotation=90)
plt.xticks([])  # remove x-axis labels
plt.show()

# %%



