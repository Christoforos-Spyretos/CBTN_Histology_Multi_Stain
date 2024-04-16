# %% IMPORTS
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %% LOAD DATA
df = pd.read_csv(r'/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/CBTN_histology_summary.csv')

# %% CLEAN DATA
# remove 'Not available' subjectID
df = df.loc[df['subjectID'] != 'Not_available']

df = df[[
    'subjectID',
    'gender',
    'age_at_diagnosis',
    'diagnosis', 
    'image_type'
    ]]

# select tumour types interested in
# tumours to be selected
tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Medulloblastoma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)'
    ]
 
df = df[df['diagnosis'].isin(tumour_types)]

df['diagnosis'] = df['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)': 'HGG', 
    'Brainstem glioma- Diffuse intrinsic pontine glioma': 'DIPG',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT'
})

# %% TYPES OF DIAGNOSIS
# diagnosis_counts_sorted = df.iloc[:, 7].value_counts().sort_values(ascending=True)

# df_diagnosis_sums = pd.DataFrame(diagnosis_counts_sorted).reset_index()
# df_diagnosis_sums.columns = ['Diagnosis', 'Number of subjects']

# # bar plot
# plt.figure(figsize=(10, 10))
# plt.barh(df_diagnosis_sums['Diagnosis'], df_diagnosis_sums['Number of subjects'], color = 'royalblue')
# plt.ylabel('Diagnosis')
# plt.xlabel('Number of Subjects')
# plt.title('Number of Subjects per Diagnosis')
# for i, v in enumerate(diagnosis_counts_sorted):
#     plt.text(v + 0.5, i, f'{v}', color='black', va='center')
# plt.show()

# %% STAIN METHODS 
sorted_stains = df.iloc [:,4].value_counts().sort_values(ascending=False)
df_stains = pd.DataFrame(sorted_stains).reset_index()
df_stains.columns = ['stain_methods', 'number_of_wsis']

output_path = '/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY'
csv_file_path = os.path.join(output_path, 'Stain Types.csv')
df_stains.to_csv(csv_file_path, index=False)

# top 10 stain methods
top_stains = df_stains.head(10)

# bar plot
plt.figure(figsize=(12, 8))  
plt.barh(top_stains['stain_methods'], top_stains['number_of_wsis'], color='royalblue')
plt.xlabel('Number of WSIs')  
plt.ylabel('Stain Methods')  
plt.title('Top 10 Stain Methods by Number of WSIs')
plt.gca().invert_yaxis() 
for i, v in enumerate(top_stains['number_of_wsis']):
    plt.text(v + 2, i, str(v), color='black', va='center')  
plt.title('Top 10 Stain Methods Bar Plot')
plt.savefig('Top 10 Stain Methods Bar Plot.png')
plt.show() 

# %% UNIQUE COMBINATION OF WSIS AND STAIN METHODS Per Subject
df_stains2 = df[['subjectID', 'image_type']]
df_stains2 = df_stains2.groupby(['subjectID', 'image_type']).first().reset_index()

sorted_stains2 = df_stains2.iloc [:,1].value_counts().sort_values(ascending=False)
df_stains3 = pd.DataFrame(sorted_stains2).reset_index()
df_stains3.columns = ['stain_methods', 'number_of_wsis']

df_stains3_top = df_stains3.head(10)

# bar plot
plt.figure(figsize=(10, 10))
plt.barh(df_stains3_top['stain_methods'], df_stains3_top['number_of_wsis'], color = 'royalblue')
plt.ylabel('Stain Methods')  
plt.xlabel('Number of WSIs')  
plt.title('Unique Combinations of WSIs and Top 10 Stain Methods Per Subject')
plt.gca().invert_yaxis() 
plt.yticks(rotation=0)  
for i, v in enumerate(df_stains3_top["number_of_wsis"]):
    plt.text(v + 0.5, i, f'{v}', color='black', va='center')  
plt.savefig('Unique Combinations of WSIs and Top 10 Stain Methods Per Subject.png')
plt.show()

# %% GENDER PIE
gender_counts = df['gender'].value_counts()

colors = ['royalblue', 'hotpink', 'grey']

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.axis('equal')  
plt.savefig('Gender Pie.png')
plt.show()

# %% VIOLIN PLOT
df_violin = df[['subjectID', 'gender', 'age_at_diagnosis', 'diagnosis']]

# smallest age for each unique combination of 'subjectID' and 'diagnosis'
df_violin = df_violin.groupby(['subjectID', 'diagnosis', 'gender'])['age_at_diagnosis'].min().reset_index()

df_violin['age_at_diagnosis'] = pd.to_numeric(df_violin['age_at_diagnosis'])

colors = {'Male': 'royalblue', 'Female': 'hotpink' , 'Not Available': 'grey'}

# violin plot
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x='diagnosis', y='age_at_diagnosis', hue='gender', data=df_violin, palette=colors, cut=0)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.set_facecolor("whitesmoke")
for idx, (diagnosis, gender) in enumerate(df_violin.groupby(['diagnosis', 'gender']).groups.keys()):
    group_data = df_violin[(df_violin['diagnosis'] == diagnosis) & (df_violin['gender'] == gender)]
    count = len(group_data)
    ypos = group_data['age_at_diagnosis'].max() + 250 
    xpos = df_violin['diagnosis'].unique().tolist().index(diagnosis) + 0.2 * (-1.3 if gender == 'Male' else 0 if gender == 'Female' else 1.4)
    ax.text(xpos, ypos, f'N={count}', ha='center', weight = 'bold')
plt.xlabel('Diagnosis')
plt.ylabel('Age at Diagnosis')
plt.xticks(ha='center')
plt.ylim(0, 14000)
plt.legend(title = '')
plt.savefig('Violin Plot.png')
plt.show()

# %% TUMOUR TYPES
# unique combination of 'subjectID' and 'diagnosis'
df_tumor_types = df.groupby(['subjectID', 'diagnosis']).first().reset_index()

sorted_tumor_types = df_tumor_types['diagnosis'].value_counts().sort_values(ascending=False)

tumor_type_colors = {
    'LGG': 'maroon',
    'HGG': 'darkgoldenrod',
    'DIPG': 'cadetblue',
    'ATRT': 'darkmagenta',
    'Medulloblastoma': 'royalblue',
    'Ependymoma': 'olive'
}

plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor("whitesmoke")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='dimgrey')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
bars = plt.bar(sorted_tumor_types.index, sorted_tumor_types.values, 
               color=[tumor_type_colors[tumor_type] for tumor_type in sorted_tumor_types.index])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(bar.get_height()), 
             ha='center', va='bottom', fontsize=10, color='black', weight = 'bold')
plt.xlabel('Tumor Types')
plt.ylabel('Number of Subjects')
plt.title('Number of Subjects per Tumor Type')
plt.savefig('Number of Subjects per Tumor Type.png')
plt.show()

# %% NUMBER OF WSIS PER TUMOUR TYPE 
df_slices = df[["diagnosis", "image_type"]]
df_slices = df_slices.loc[df_slices['image_type'] != 'UNKNOWN']

# %%
diagnosis_counts = df_slices['diagnosis'].value_counts()

# %%
sorted_diagnosis_counts = diagnosis_counts.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = plt.axes()
ax.set_facecolor("whitesmoke")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='dimgrey')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
bars = plt.bar(sorted_diagnosis_counts.index, sorted_diagnosis_counts.values, 
               color=[tumor_type_colors.get(diagnosis, 'blue') for diagnosis in sorted_diagnosis_counts.index])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(bar.get_height()), 
             ha='center', va='bottom', fontsize=10, color='black', weight='bold')
plt.xlabel('Tumor Types')
plt.ylabel('Number of WSIs')
plt.title('Number of WSIs per Tumor Type')
plt.savefig('Number of WSIs per Tumor Type.png')
plt.show()

# %% DATAFRAME FOR DENSITY PLOTS
df_age_at_diagnosis = df[['subjectID', 'gender', 'diagnosis', 'age_at_diagnosis']]

# unique combination of 'subjectID' and 'diagnosis'
df_age_at_diagnosis = df_age_at_diagnosis.groupby(['subjectID','diagnosis']).min().reset_index()

df_age_at_diagnosis = df_age_at_diagnosis.loc[df_age_at_diagnosis['gender'] != 'Not Available']

df_age_at_diagnosis['age_at_diagnosis'] = pd.to_numeric(df_age_at_diagnosis['age_at_diagnosis'])

# standardise 'age_at_diagnosis' between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_age_at_diagnosis['age_at_diagnosis_scaled'] = scaler.fit_transform(df_age_at_diagnosis[['age_at_diagnosis']])

colors = {'Male': 'royalblue', 'Female': 'hotpink'}

# %% DENSITY PLOTS OF AGE AT DIAGBOSIS BY GENDER FOR EACH TUMOUR TYPE
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df_age_at_diagnosis, x='age_at_diagnosis', hue='gender', palette=colors, fill=True, common_norm=False)
ax.set_facecolor("whitesmoke")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
plt.xlabel('Age at Diagnosis')
plt.xlim(0,14000)
plt.title('Density Plot of Age at Diagnosis by Gender')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Male'], markersize=10, label='Male'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Female'], markersize=10, label='Female')]
plt.legend(handles=legend_handles)
plt.savefig('Density Plot of Age at Diagnosis by Gender.png')
plt.show()

# scaled age
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df_age_at_diagnosis, x='age_at_diagnosis_scaled', hue='gender', palette=colors, fill=True, common_norm=False)
ax.set_facecolor("whitesmoke")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
plt.xlabel('Age at Diagnosis (Scaled)')
plt.title('Density Plot of Scaled Age at Diagnosis by Gender')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Male'], markersize=10, label='Male'),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Female'], markersize=10, label='Female')]
plt.legend(handles=legend_handles)
plt.savefig('Density Plot of Scaled Age at Diagnosis by Gender.png')
plt.show()

# %% DENSITY PLOTS OF AGE AT DIAGNOSIS BY GENDER FOR EACH TUMOUR TYPE
tumour_types = [
    'LGG',
    'HGG', 
    'Ependymoma',
    'DIPG',
    'Medulloblastoma',
    'ATRT'
    ]

for tumor_type in tumour_types:
    df_tumor_type = df_age_at_diagnosis[df_age_at_diagnosis['diagnosis'] == tumor_type]
    plt.figure(figsize=(10, 6))
    for gender, color in colors.items():
        df_gender = df_tumor_type[df_tumor_type['gender'] == gender]
        ax = sns.kdeplot(data=df_gender, x='age_at_diagnosis', color=color, fill=True, label=gender)
    ax.set_facecolor("whitesmoke")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.xlabel('Age at Diagnosis')
    plt.xlim(0,14000)
    plt.title(f'Density Plot of Age at Diagnosis by Gender for {tumor_type}')
    plt.legend(title='Gender')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Male'], markersize=10, label='Male'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Female'], markersize=10, label='Female')
                      ]
    plt.legend(handles=legend_handles)
    plt.savefig(f'Density Plot of Age at Diagnosis by Gender For {tumor_type}.png')
    plt.show()

# scaled age
for tumor_type in tumour_types:
    df_tumor_type = df_age_at_diagnosis[df_age_at_diagnosis['diagnosis'] == tumor_type]
    plt.figure(figsize=(10, 6))
    for gender, color in colors.items():
        df_gender = df_tumor_type[df_tumor_type['gender'] == gender]
        ax = sns.kdeplot(data=df_gender, x='age_at_diagnosis_scaled', color=color, fill=True, label=gender)
    ax.set_facecolor("whitesmoke")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    plt.xlabel('Age at Diagnosis (Scaled)')
    plt.title(f'Density Plot of Scaled Age at Diagnosis by Gender for {tumor_type}')
    plt.legend(title='Gender')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Male'], markersize=10, label='Male'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['Female'], markersize=10, label='Female')
                      ]
    plt.legend(handles=legend_handles)
    plt.savefig(f'Density Plot of Scaled Age at Diagnosis by Gender For {tumor_type}.png')
    plt.show()   
    
# %% DENSITY PLOTS OF AGE AT DIAGNOSIS FOR EACH TUMOUR TYPE
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df_age_at_diagnosis, x='age_at_diagnosis', hue='diagnosis', palette=tumor_type_colors, fill=True, common_norm=False)
ax.set_facecolor("whitesmoke")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
plt.xlabel('Age at Diagnosis')
plt.xlim(0,14000)
plt.title('Density Plot of Age at Diagnosis for Each Tumour Type')
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['LGG'], markersize=10, label='LGG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['HGG'], markersize=10, label='HGG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['Ependymoma'], markersize=10, label='Ependymoma'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['Medulloblastoma'], markersize=10, label='Medulloblastoma'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['DIPG'], markersize=10, label='DIPG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['ATRT'], markersize=10, label='ATRT'),
    ]
plt.legend(handles=legend_handles)
plt.show()

# scaled age
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df_age_at_diagnosis, x='age_at_diagnosis_scaled', hue='diagnosis', palette=tumor_type_colors, fill=True, common_norm=False)
ax.set_facecolor("whitesmoke")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
plt.xlabel('Age at Diagnosis (Scaled)')
plt.title('Density Plot of Scaled Age at Diagnosis for Each Tumour Type')
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['LGG'], markersize=10, label='LGG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['HGG'], markersize=10, label='HGG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['Ependymoma'], markersize=10, label='Ependymoma'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['Medulloblastoma'], markersize=10, label='Medulloblastoma'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['DIPG'], markersize=10, label='DIPG'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tumor_type_colors['ATRT'], markersize=10, label='ATRT'),
    ]
plt.legend(handles=legend_handles)
plt.show()

# %% RIDGE DENSITY PLOTS OF AGE AT DIAGNOSIS FOR EACH TUMOUR TYPE
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("Set2", 12)
g = sns.FacetGrid(df_age_at_diagnosis, palette=tumor_type_colors, row="diagnosis", hue="diagnosis", aspect=5, height=2)
g.map_dataframe(sns.kdeplot, x="age_at_diagnosis", fill=True, alpha=1)
g.map_dataframe(sns.kdeplot, x="age_at_diagnosis", color='black')
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=13,
            ha="left", va="center", transform=ax.transAxes)
g.map(label, "diagnosis")
g.fig.subplots_adjust(hspace=-.5)
g.set_titles("")
g.set(yticks=[], xlabel="Age at Diagnosis", ylabel= "")
g.despine( left=True)
plt.suptitle('Ridge Density Plots of Age at Diagnosis for Each Tumour Type', y=1)
plt.savefig('Ridge Density Plots of Age at Diagnosis for Each Tumour Type.png')


# scaled age
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("Set2", 12)
g = sns.FacetGrid(df_age_at_diagnosis, palette=tumor_type_colors, row="diagnosis", hue="diagnosis", aspect=5, height=2)
g.map_dataframe(sns.kdeplot, x="age_at_diagnosis_scaled", fill=True, alpha=1)
g.map_dataframe(sns.kdeplot, x="age_at_diagnosis_scaled", color='black')
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=13,
            ha="left", va="center", transform=ax.transAxes)
g.map(label, "diagnosis")
g.fig.subplots_adjust(hspace=-.5)
g.set_titles("")
g.set(yticks=[], xlabel="Age at Diagnosis (Scaled)", ylabel= "")
g.despine( left=True)
plt.suptitle('Ridge Density Plots of Scaled Age at Diagnosis for Each Tumour Type', y=1)
plt.savefig('Ridge Density Plots of Scaled Age at Diagnosis for Each Tumour Type.png')

# %%