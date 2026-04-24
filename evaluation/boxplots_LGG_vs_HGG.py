# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

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
summary_results = summary_results[summary_results['Feature_Encoder'] == 'conch_v1_5']

# %% SUMMARY
# combine the Fusion + Modality + Feature_Extraction columns into one column
summary_results['Configuration'] = summary_results['Fusion'] + '_' + summary_results['Modality'] + '_' + summary_results['Aggregation']

# Create a new column for legend that distinguishes Single Stain types
summary_results['Fusion_Legend'] = summary_results.apply(
    lambda row: 'H&E' if (row['Fusion'] == 'Single_Stain' and row['Modality'] == 'HE') 
    else ('Ki-67' if (row['Fusion'] == 'Single_Stain' and row['Modality'] == 'KI67') 
    else row['Fusion']), 
    axis=1
)

# %% BOX PLOT SETTINGS
# set the colours
palette = sns.color_palette("deep", 10)  
custom_colors = [palette[0], palette[1], palette[2], palette[3], palette[4], palette[5], palette[6], palette[7], palette[8], palette[9], 
                 (0.7, 0.3, 0.7),   # Two-Hidden Layer (magenta)
                 (0.2, 0.6, 0.6),   # Attention Layer (teal)
                 (0.7, 0.55, 0.0)]  # New 13th method (golden/amber)
sns.set_style("white")
n_hues = len(summary_results['Fusion_Legend'].unique())

# %% BOX PLOT BALANCED ACCURACY
plt.figure(figsize=(15.8, 7.3))
plt.gca().set_facecolor('white')  # set the background of the plot to pure white

# set the outlier properties
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')

boxplot = sns.boxplot(x='Configuration', y='BA', data=summary_results, hue="Fusion_Legend", palette=custom_colors, flierprops=flierprops, width=0.5, medianprops={'linewidth': 2.5, 'color': 'black'})

# Apply hatch patterns to boxes (index by hue so each hue always gets the same hatch)
hatches = ['/', '\\', '|', '..', 'x', 'o', 'O', '.', '\\\\', '//', 'xx', '||', '////']
for i, patch in enumerate(boxplot.patches):
    hatch = hatches[(i % n_hues) % len(hatches)]
    patch.set_hatch(hatch)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Get handles in seaborn's default order, then apply matching hatches
default_legend = plt.legend()
handles = list(default_legend.legend_handles)
for j, h in enumerate(handles):
    h.set_hatch(hatches[j % len(hatches)])
    h.set_edgecolor('black')
padded_labels = [
    'Single-Stain H&E',
    'Single-Stain Ki-67',
    'Early Fusion',
    'Intermediate H&E-Guided\nCross Attention Fusion',
    'Intermediate Ki-67-Guided\nCross Attention Fusion',
    'Intermediate Element-Wise\nMultiplication Fusion',
    'Intermediate Concatenation\nFusion',
    'Aggregation of Softmax\nScores Late Fusion',
    'Aggregation of Logits\nLate Fusion',
    'Linear Layer Late\nFusion Learning Model',
    'One-Hidden Layer Late\nFusion Learning Model',
    'Two-Hidden Layer Late\nFusion Learning Model',
    'Attention Layer Late\nFusion Learning Model'
]
legend = plt.legend(handles, padded_labels, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=4, fontsize=12.44, columnspacing=1.0, handletextpad=0.5, labelspacing=0.7)  # 4+4+4+1 layout
legend.get_frame().set_facecolor('white')  # set legend box background color to white
legend.get_frame().set_linewidth(1)  # make border more bold
legend.get_frame().set_edgecolor('black')  # set border color

plt.xlabel('')
plt.ylabel('Balanced Accuracy [0,1] ', fontsize=12)
plt.ylim(0, 1.00)
# set BA y-ticks every 0.1 and format labels with two decimals
ba_ticks = np.arange(0.0, 1.01, 0.1)
plt.yticks(ba_ticks, [f"{t:.2f}" for t in ba_ticks], fontsize=12)
# draw horizontal grid lines at the y-ticks and keep them behind the boxes
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', color='grey', alpha=0.5, linewidth=0.8)
plt.xticks([])  # remove x-axis labels
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_BA_boxplot.png', bbox_inches='tight', dpi=300)
plt.show()

# %% BOX PLOT MCC
plt.figure(figsize=(15.8, 7.3))
plt.gca().set_facecolor('white')  # set the background of the plot to pure white

# set the outlier properties
flierprops = dict(marker='D', markerfacecolor='darkgrey', markersize=5, linestyle='none')

boxplot = sns.boxplot(x='Configuration', y='MCC', data=summary_results, hue="Fusion_Legend", palette=custom_colors, flierprops=flierprops, width=0.5, medianprops={'linewidth': 2.5, 'color': 'black'})

# Apply hatch patterns to boxes (index by hue so each hue always gets the same hatch)
hatches = ['/', '\\', '|', '..', 'x', 'o', 'O', '.', '\\\\', '//', 'xx', '||', '////']
for i, patch in enumerate(boxplot.patches):
    hatch = hatches[(i % n_hues) % len(hatches)]
    patch.set_hatch(hatch)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Get handles in seaborn's default order, then apply matching hatches
default_legend = plt.legend()
handles = list(default_legend.legend_handles)
for j, h in enumerate(handles):
    h.set_hatch(hatches[j % len(hatches)])
    h.set_edgecolor('black')
padded_labels = [
    'Single-Stain H&E',
    'Single-Stain Ki-67',
    'Early Fusion',
    'Intermediate H&E-Guided\nCross Attention Fusion',
    'Intermediate Ki-67-Guided\nCross Attention Fusion',
    'Intermediate Element-Wise\nMultiplication Fusion',
    'Intermediate Concatenation\nFusion',
    'Aggregation of Softmax\nScores Late Fusion',
    'Aggregation of Logits\nLate Fusion',
    'Linear Layer Late\nFusion Learning Model',
    'One-Hidden Layer Late\nFusion Learning Model',
    'Two-Hidden Layer Late\nFusion Learning Model',
    'Attention Layer Late\nFusion Learning Model'
]
legend = plt.legend(handles, padded_labels, loc='upper center', bbox_to_anchor=(0.5, 1.31), ncol=4, fontsize=12.44, columnspacing=1.0, handletextpad=0.5, labelspacing=0.7)  # 4+4+4+1 layout
legend.get_frame().set_facecolor('white')  # set legend box background color to white
legend.get_frame().set_linewidth(1)  # make border more bold
legend.get_frame().set_edgecolor('black')  # set border color

plt.xlabel('')
plt.ylabel('Matthews Correlation Coefficient [-1,1] ', fontsize=12)
plt.ylim(-1, 1.00)
plt.yticks([-1, -0.80, -0.60, -0.40, -0.20, 0, 0.20, 0.40, 0.60, 0.80, 1.00], ['-1.00', '-0.80', '-0.60', '-0.40', '-0.20', '0.00', '0.20', '0.40', '0.60', '0.80', '1.00'], fontsize=12)
# draw horizontal grid lines at the y-ticks and keep them behind the boxes
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', color='grey', alpha=0.5, linewidth=0.8)
plt.xticks([])  # remove x-axis labels
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/LGG_vs_HGG_MCC_boxplot.png', bbox_inches='tight', dpi=300)
plt.show()

# %%


