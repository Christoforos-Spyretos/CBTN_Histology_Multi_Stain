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
    'H&E Raw Features',   'Ki-67 Raw Features',   'H&E Learned Features',   'Ki-67 Learned Features',   'Early Fusion Raw Features',
    'H&E Raw Features',    'Ki-67 Raw Features',    'H&E Learned Features',    'Ki-67 Learned Features',    'Early Fusion Raw Features',
    'Early Fusion\nLearned Features',  'Intermediate H&E-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate Ki-67-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate Concatenation\nLearned Features',  'Intermediate Element-Wise\nMultiplication Learned\nFeatures',
    'Early Fusion\nLearned Features',   'Intermediate H&E-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate Ki-67-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate Concatenation\nLearned Features',   'Intermediate Element-Wise\nMultiplication Learned\nFeatures',
]

row_labels = ['Train Set', 'Test Set', 'Train Set', 'Test Set']

fig = plt.figure(figsize=(25, 16))

# Rows 1 & 2 — tight spacing
gs_top  = GridSpec(2, 5, figure=fig, top=0.97, bottom=0.53, hspace=0, wspace=0.02)
# Row 3 — same height as rows 1 & 2 (0.22 in figure coords)
gs_row3 = GridSpec(1, 5, figure=fig, top=0.54, bottom=0.295, wspace=0.02)
# Row 4 — adjust top to control gap between rows 3 and 4
gs_row4 = GridSpec(1, 5, figure=fig, top=0.295, bottom=0.05, wspace=0.02)

axes = [fig.add_subplot(gs_top[r, c])  for r in range(2) for c in range(5)] + \
       [fig.add_subplot(gs_row3[0, c]) for c in range(5)] + \
       [fig.add_subplot(gs_row4[0, c]) for c in range(5)]

for ax, path, title in zip(axes, images_LGG_vs_HGG, labels):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.55, -0.02, title, ha='center', va='top', transform=ax.transAxes, fontsize=16)

row_first_axes = [axes[0], axes[5], axes[10], axes[15]]
for ax, row_label in zip(row_first_axes, row_labels):
    ax.text(-0.05, 0.5, row_label, ha='right', va='center',
            transform=ax.transAxes, fontsize=16, rotation=90)

# Legend box — centred at the bottom
ax_legend = fig.add_axes([0.43, -0.03, 0.18, 0.06])
ax_legend.imshow(plt.imread(legend_box_LGG_vs_HGG))
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
    'H&E Raw Features',   'Ki-67 Raw Features',   'H&E Learned Features',   'Ki-67 Learned Features',   'Early Fusion Raw Features',
    'H&E Raw Features',    'Ki-67 Raw Features',    'H&E Learned Features',    'Ki-67 Learned Features',    'Early Fusion Raw Features',
    'Early Fusion\nLearned Features',  'Intermediate H&E-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate Ki-67-Guided\nCross-Attention Fusion\nLearned Features',  'Intermediate Concatenation\nLearned Features',  'Intermediate Element-Wise\nMultiplication Learned\nFeatures',
    'Early Fusion\nLearned Features',   'Intermediate H&E-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate Ki-67-Guided\nCross-Attention Fusion\nLearned Features',   'Intermediate Concatenation\nLearned Features',   'Intermediate Element-Wise\nMultiplication Learned\nFeatures',
]

row_labels = ['Train Set', 'Test Set', 'Train Set', 'Test Set']

fig = plt.figure(figsize=(25, 16))

# Rows 1 & 2 — tight spacing
gs_top  = GridSpec(2, 5, figure=fig, top=0.97, bottom=0.53, hspace=0, wspace=0.02)
# Row 3 — same height as rows 1 & 2 (0.22 in figure coords)
gs_row3 = GridSpec(1, 5, figure=fig, top=0.54, bottom=0.295, wspace=0.02)
# Row 4 — adjust top to control gap between rows 3 and 4
gs_row4 = GridSpec(1, 5, figure=fig, top=0.295, bottom=0.05, wspace=0.02)

axes = [fig.add_subplot(gs_top[r, c])  for r in range(2) for c in range(5)] + \
       [fig.add_subplot(gs_row3[0, c]) for c in range(5)] + \
       [fig.add_subplot(gs_row4[0, c]) for c in range(5)]

for ax, path, title in zip(axes, images_5_class, labels):
    img = plt.imread(path)
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.55, -0.02, title, ha='center', va='top', transform=ax.transAxes, fontsize=16)

row_first_axes = [axes[0], axes[5], axes[10], axes[15]]
for ax, row_label in zip(row_first_axes, row_labels):
    ax.text(-0.05, 0.5, row_label, ha='right', va='center',
            transform=ax.transAxes, fontsize=16, rotation=90)

# Legend box — centred at the bottom
ax_legend = fig.add_axes([0.27, -0.04, 0.50, 0.08])
ax_legend.imshow(plt.imread(legend_box_5_class))
ax_legend.axis('off')
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/tSNE_plots/5_class/tSNE/aggregate_tSNE_5_class.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
