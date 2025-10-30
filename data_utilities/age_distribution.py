# %% IMPORTS
import pandas as pd
import os
import matplotlib.pyplot as plt 

# %% LOAD CSVs
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv')
histological_diagnosis_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Histological Diagnoses', engine='openpyxl')
participants_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Participants', engine='openpyxl')

# %% MATCH CASE_ID AND EXTRACT AGE AT DIAGNOSIS
# Map the age at diagnosis to df based on case_id
df['age_at_diagnosis_(days)'] = df['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Age at Diagnosis (Days)'].to_dict())

df['gender'] = df['case_id'].map(participants_df_from_portal.set_index('External Id')['Gender'].to_dict())

# Convert age from days to years
df['age_at_diagnosis_(years)'] = df['age_at_diagnosis_(days)'] / 365.25

# %% PLOT AGE DISTRIBUTION - DAYS
plt.figure(figsize=(10, 6))
df['age_at_diagnosis_(days)'].hist(bins=30, edgecolor='black')
plt.title('Age at Diagnosis Distribution (Days)')
plt.xlabel('Age at Diagnosis (Days)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_days.png', dpi=300, bbox_inches='tight')
plt.show()

# %% PLOT AGE DISTRIBUTION - YEARS
plt.figure(figsize=(10, 6))
df['age_at_diagnosis_(years)'].hist(bins=30, edgecolor='black')
plt.title('Age at Diagnosis Distribution (Years)')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_years.png', dpi=300, bbox_inches='tight')
plt.show()

# %% PLOT AGE DISTRIBUTION BY GENDER
plt.figure(figsize=(10, 6))
df_male = df[df['gender'] == 'Male']
df_female = df[df['gender'] == 'Female']

plt.hist(df_male['age_at_diagnosis_(years)'].dropna(), bins=30, alpha=0.6, label='Male', color='blue', edgecolor='black')
plt.hist(df_female['age_at_diagnosis_(years)'].dropna(), bins=30, alpha=0.6, label='Female', color='pink', edgecolor='black')

plt.title('Age at Diagnosis Distribution by Gender')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_by_gender.png', dpi=300, bbox_inches='tight')
plt.show()

# %% PLOT AGE DISTRIBUTION BY LABEL 
# LGG 
plt.figure(figsize=(10, 6))
df_lgg = df[df['label'] == 'LGG']
plt.hist(df_lgg['age_at_diagnosis_(years)'].dropna(), bins=30, edgecolor='black', color='C0')
plt.title('Age at Diagnosis Distribution - LGG')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_LGG.png', dpi=300, bbox_inches='tight')
plt.show()

# HGG
plt.figure(figsize=(10, 6))
df_hgg = df[df['label'] == 'HGG']
plt.hist(df_hgg['age_at_diagnosis_(years)'].dropna(), bins=30, edgecolor='black', color='C1')
plt.title('Age at Diagnosis Distribution - HGG')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_HGG.png', dpi=300, bbox_inches='tight')
plt.show()

# MB
plt.figure(figsize=(10, 6))
df_mb = df[df['label'] == 'MB']
plt.hist(df_mb['age_at_diagnosis_(years)'].dropna(), bins=30, edgecolor='black', color='C2')
plt.title('Age at Diagnosis Distribution - MB')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_MB.png', dpi=300, bbox_inches='tight')
plt.show()

# EP
plt.figure(figsize=(10, 6))
df_ep = df[df['label'] == 'EP']
plt.hist(df_ep['age_at_diagnosis_(years)'].dropna(), bins=30, edgecolor='black', color='C3')
plt.title('Age at Diagnosis Distribution - EP')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_EP.png', dpi=300, bbox_inches='tight')
plt.show()

# GG
plt.figure(figsize=(10, 6))
df_gg = df[df['label'] == 'GG']
plt.hist(df_gg['age_at_diagnosis_(years)'].dropna(), bins=30, edgecolor='black', color='C4')
plt.title('Age at Diagnosis Distribution - GG')
plt.xlabel('Age at Diagnosis (Years)')
plt.ylabel('Frequency')
plt.xlim(left=0)
plt.grid(False)
plt.savefig('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/data_utilities/age_at_diagnosis_distribution_GG.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
