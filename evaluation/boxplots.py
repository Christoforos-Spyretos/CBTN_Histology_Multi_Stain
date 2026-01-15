# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% LOAD CSV
summary_results = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_summary.csv')  

# %%
# Feature_Encoder: conch_v1
# Aggregation_Method: small_clam_sb
# Fusion = Single_Stain, 
#          Early_Fusion, 
#          Intermediate_Fusion_CA, 
#          Late_Fusion_PA, 
#          Late_Fusion_LA, 
#          Late_Fusion_LM_SM, 
#          Late_Fusion_LM_OHL, 
#          Late_Fusion_LM_AM
# Modality = HE, KI67, HE_KI67
# BA,
# MCC,
# AUC,
# F1_Score

# %% FILTERING
# pick the Feature_Encoder interested in
summary_results = summary_results[summary_results['Feature_Encoder'] == 'conch_v1']

# %% SUMMARY
# combine the Fusion + Modality + Feature_Extraction columns into one column
summary_results['Configuration'] = summary_results['Fusion'] + '_' + summary_results['Modality'] + '_' + summary_results['Aggregation']

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

boxplot = sns.boxplot(x='Configuration', y='BA', data=summary_results, hue="Fusion", palette=custom_colors, flierprops=flierprops)
legend = plt.legend(loc='lower left', bbox_to_anchor=(0, 0))  # place legend at bottom left
legend.get_frame().set_facecolor('white')  # set legend box background color to white

# rename legend labels
new_legend_labels = [
    'Single Stain', 
    'Early Fusion', 
    'Aggregation of Probabilities', 
    'Aggregation of Logits', 
    'Single-Layer Network']
for t, l in zip(legend.texts, new_legend_labels):
    t.set_text(l)

plt.xlabel('')
plt.ylabel('')
plt.ylim(0.45, 1)
plt.xticks(rotation=45)
plt.xticks()  # remove x-axis labels
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_boxplot.png', bbox_inches='tight')
plt.show()

# %%


