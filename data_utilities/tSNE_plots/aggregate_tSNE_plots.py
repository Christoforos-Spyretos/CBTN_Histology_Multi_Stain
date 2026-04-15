# %% IMPORTS
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %% PATHs TO LGG vs HGG tSNE PLOTS
HE_raw_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_HE_features_LGG_vs_HGG_train.png'
HE_raw_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_HE_features_LGG_vs_HGG_test.png'
KI67_raw_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_KI67_features_LGG_vs_HGG_train.png'
KI67_raw_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_KI67_features_LGG_vs_HGG_test.png'

HE_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_HE_features_LGG_vs_HGG_train.png'
HE_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_HE_features_LGG_vs_HGG_test.png'
KI67_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_KI67_features_LGG_vs_HGG_train.png'
KI67_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_KI67_features_LGG_vs_HGG_test.png'

EF_raw_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_Early_Fusion_HE_KI67_features_LGG_vs_HGG_train.png'
EF_raw_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_raw_Early_Fusion_HE_KI67_features_LGG_vs_HGG_test.png'
EF_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_Early_Fusion_HE_KI67_features_LGG_vs_HGG_train.png'
EF_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_Early_Fusion_HE_KI67_features_LGG_vs_HGG_test.png'

IF_CA_HE_inform_KI67_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_CrossAttn_HE_inform_KI67_features_LGG_vs_HGG_train.png'
IF_CA_HE_inform_KI67_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_CrossAttn_HE_inform_KI67_features_LGG_vs_HGG_test.png'

IF_CA_KI67_inform_HE_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_CrossAttn_KI67_inform_HE_features_LGG_vs_HGG_train.png'
IF_CA_KI67_inform_HE_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_CrossAttn_KI67_inform_HE_features_LGG_vs_HGG_test.png'

IF_CONCAT_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_Concatenation_HE_KI67_features_LGG_vs_HGG_train.png'
IF_CONCAT_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_Concatenation_HE_KI67_features_LGG_vs_HGG_test.png'

IF_EWM_learned_train_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_ElemWiseMult_HE_KI67_features_LGG_vs_HGG_train.png'
IF_EWM_learned_test_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/tSNE_learned_ElemWiseMult_HE_KI67_features_LGG_vs_HGG_test.png'

legend_box_LGG_vs_HGG = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/legend_box_LGG_vs_HGG.png'

# %% AGGREGATE LGG vs HGG tSNE PLOTS
images_LGG_vs_HGG = [
    HE_raw_train_LGG_vs_HGG, KI67_raw_train_LGG_vs_HGG, HE_learned_train_LGG_vs_HGG, KI67_learned_train_LGG_vs_HGG, EF_raw_train_LGG_vs_HGG,
    HE_raw_test_LGG_vs_HGG,  KI67_raw_test_LGG_vs_HGG,  HE_learned_test_LGG_vs_HGG,  KI67_learned_test_LGG_vs_HGG,  EF_raw_test_LGG_vs_HGG,
    EF_learned_train_LGG_vs_HGG, IF_CA_HE_inform_KI67_learned_train_LGG_vs_HGG, IF_CA_KI67_inform_HE_learned_train_LGG_vs_HGG, IF_CONCAT_learned_train_LGG_vs_HGG, IF_EWM_learned_train_LGG_vs_HGG,
    EF_learned_test_LGG_vs_HGG,  IF_CA_HE_inform_KI67_learned_test_LGG_vs_HGG,  IF_CA_KI67_inform_HE_learned_test_LGG_vs_HGG,  IF_CONCAT_learned_test_LGG_vs_HGG,  IF_EWM_learned_test_LGG_vs_HGG,
]

labels = [
    'H&E Raw\nFeatures',   'Ki-67 Raw\nFeatures',   'H&E Learned\nFeatures',   'Ki-67 Learned\nFeatures',   'Early Fusion\nRaw Features',
    'H&E Raw\nFeatures',    'Ki-67 Raw\nFeatures',    'H&E Learned\nFeatures',    'Ki-67 Learned\nFeatures',    'Early Fusion\nRaw Features',
    'Early Fusion\nLearned\nFeatures',  'Intermediate\nH&E-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate\nKi-67-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate\nConcatenation\nLearned\nFeatures',  'Intermediate\nElement-Wise\nMultiplication\nLearned Features',
    'Early Fusion\nLearned\nFeatures',   'Intermediate\nH&E-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate\nKi-67-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate\nConcatenation\nLearned\nFeatures',   'Intermediate\nElement-Wise\nMultiplication\nLearned Features',
]

row_labels = ['Train Set', 'Test Set', 'Train Set', 'Test Set']

fig = plt.figure(figsize=(25, 16))

# Rows 1 & 2 — slight gap between them
gs_top  = GridSpec(2, 5, figure=fig, top=0.97, bottom=0.52, hspace=0.18, wspace=0.02)
# Row 3 — gap of ~0.035 below row 2
gs_row3 = GridSpec(1, 5, figure=fig, top=0.495, bottom=0.26, wspace=0.02)
# Row 4 — gap of ~0.07 below row 3
gs_row4 = GridSpec(1, 5, figure=fig, top=0.2, bottom=-0.040, wspace=0.02)

axes = [fig.add_subplot(gs_top[r, c])  for r in range(2) for c in range(5)] + \
       [fig.add_subplot(gs_row3[0, c]) for c in range(5)] + \
       [fig.add_subplot(gs_row4[0, c]) for c in range(5)]

for ax, path, title in zip(axes, images_LGG_vs_HGG, labels):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.57, -0.02, title, ha='center', va='top', transform=ax.transAxes, fontsize=24)

row_first_axes = [axes[0], axes[5], axes[10], axes[15]]
for ax, row_label in zip(row_first_axes, row_labels):
    ax.text(-0.05, 0.60, row_label, ha='right', va='center',
            transform=ax.transAxes, fontsize=24, rotation=90)

# Legend box — centred at the bottom
# Adjust these to reposition/resize the legend box:
legend_left_LGG_vs_HGG   = 0.43   # horizontal position (0=left edge, 1=right edge)
legend_bottom_LGG_vs_HGG = -0.17  # vertical position (negative = below figure)
legend_width_LGG_vs_HGG  = 0.18   # width in figure fraction
legend_height_LGG_vs_HGG = 0.055   # height in figure fraction
ax_legend = fig.add_axes([legend_left_LGG_vs_HGG, legend_bottom_LGG_vs_HGG,
                          legend_width_LGG_vs_HGG, legend_height_LGG_vs_HGG])
ax_legend.imshow(plt.imread(legend_box_LGG_vs_HGG), aspect='auto')
ax_legend.axis('off')
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/LGG_vs_HGG/tSNE/aggregate_tSNE_LGG_vs_HGG.png', dpi=150, bbox_inches='tight')
plt.show()

# %% PATHs TO 5-CLASS tSNE PLOTS
HE_raw_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_HE_features_5_class_train.png'
HE_raw_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_HE_features_5_class_test.png'
KI67_raw_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_KI67_features_5_class_train.png'
KI67_raw_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_KI67_features_5_class_test.png'

HE_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_HE_features_5_class_train.png'
HE_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_HE_features_5_class_test.png'
KI67_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_KI67_features_5_class_train.png'
KI67_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_KI67_features_5_class_test.png'

EF_raw_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_Early_Fusion_HE_KI67_features_5_class_train.png'
EF_raw_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_raw_Early_Fusion_HE_KI67_features_5_class_test.png'
EF_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_Early_Fusion_HE_KI67_features_5_class_train.png'
EF_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_Early_Fusion_HE_KI67_features_5_class_test.png'

IF_CA_HE_inform_KI67_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_CrossAttn_HE_inform_KI67_features_5_class_train.png'
IF_CA_HE_inform_KI67_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_CrossAttn_HE_inform_KI67_features_5_class_test.png'

IF_CA_KI67_inform_HE_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_CrossAttn_KI67_inform_HE_features_5_class_train.png'
IF_CA_KI67_inform_HE_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_CrossAttn_KI67_inform_HE_features_5_class_test.png'

IF_CONCAT_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_Concatenation_HE_KI67_features_5_class_train.png'
IF_CONCAT_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_Concatenation_HE_KI67_features_5_class_test.png'

IF_EWM_learned_train_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_ElemWiseMult_HE_KI67_features_5_class_train.png'
IF_EWM_learned_test_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/tSNE_learned_ElemWiseMult_HE_KI67_features_5_class_test.png'

legend_box_5_class = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/legend_box_5_class.png'

# %% AGGREGATE 5 CLASS tSNE PLOTS
images_5_class = [
    HE_raw_train_5_class, KI67_raw_train_5_class, HE_learned_train_5_class, KI67_learned_train_5_class, EF_raw_train_5_class,
    HE_raw_test_5_class,  KI67_raw_test_5_class,  HE_learned_test_5_class,  KI67_learned_test_5_class,  EF_raw_test_5_class,
    EF_learned_train_5_class, IF_CA_HE_inform_KI67_learned_train_5_class, IF_CA_KI67_inform_HE_learned_train_5_class, IF_CONCAT_learned_train_5_class, IF_EWM_learned_train_5_class,
    EF_learned_test_5_class,  IF_CA_HE_inform_KI67_learned_test_5_class,  IF_CA_KI67_inform_HE_learned_test_5_class,  IF_CONCAT_learned_test_5_class,  IF_EWM_learned_test_5_class,
]

labels = [
    'H&E Raw\nFeatures',   'Ki-67 Raw\nFeatures',   'H&E Learned\nFeatures',   'Ki-67 Learned\nFeatures',   'Early Fusion\nRaw Features',
    'H&E Raw\nFeatures',    'Ki-67 Raw\nFeatures',    'H&E Learned\nFeatures',    'Ki-67 Learned\nFeatures',    'Early Fusion\nRaw Features',
    'Early Fusion\nLearned\nFeatures',  'Intermediate\nH&E-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate\nKi-67-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate\nConcatenation\nLearned\nFeatures',  'Intermediate\nElement-Wise\nMultiplication\nLearned Features',
    'Early Fusion\nLearned\nFeatures',   'Intermediate\nH&E-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate\nKi-67-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate\nConcatenation\nLearned\nFeatures',   'Intermediate\nElement-Wise\nMultiplication\nLearned Features',
]

row_labels = ['Train Set', 'Test Set', 'Train Set', 'Test Set']

fig = plt.figure(figsize=(25, 16))

# Rows 1 & 2 — slight gap between them
gs_top  = GridSpec(2, 5, figure=fig, top=0.97, bottom=0.52, hspace=0.18, wspace=0.02)
# Row 3 — gap of ~0.035 below row 2
gs_row3 = GridSpec(1, 5, figure=fig, top=0.495, bottom=0.26, wspace=0.02)
# Row 4 — gap of ~0.07 below row 3
gs_row4 = GridSpec(1, 5, figure=fig, top=0.2, bottom=-0.040, wspace=0.02)

axes = [fig.add_subplot(gs_top[r, c])  for r in range(2) for c in range(5)] + \
       [fig.add_subplot(gs_row3[0, c]) for c in range(5)] + \
       [fig.add_subplot(gs_row4[0, c]) for c in range(5)]

for ax, path, title in zip(axes, images_5_class, labels):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.57, -0.02, title, ha='center', va='top', transform=ax.transAxes, fontsize=24)

row_first_axes = [axes[0], axes[5], axes[10], axes[15]]
for ax, row_label in zip(row_first_axes, row_labels):
    ax.text(-0.05, 0.60, row_label, ha='right', va='center',
            transform=ax.transAxes, fontsize=24, rotation=90)

# Legend box — centred at the bottom
# Adjust these to reposition/resize the legend b60ox:
legend_left_5_class   = 0.20   # horizontal position (0=left edge, 1=right edge)
legend_bottom_5_class = -0.20  # vertical position (negative = below figure)
legend_width_5_class  = 0.63   # width in figure fraction
legend_height_5_class = 0.09  # height in figure fraction
ax_legend = fig.add_axes([legend_left_5_class, legend_bottom_5_class,
                          legend_width_5_class, legend_height_5_class])
ax_legend.imshow(plt.imread(legend_box_5_class), aspect='auto')
ax_legend.axis('off')
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/aggregate_tSNE_5_class.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
