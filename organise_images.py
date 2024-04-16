# %% IMPORTS
import os
import pandas as pd
import shutil

# %% COPY DATA 
df = pd.read_csv('/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/CBTN_histology_summary.csv')
histology_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS"
new_histology_path = "/local/data2/chrsp39/CBTN_v2/CLAM/HE/WSI"

# %%
df = df.loc[df['subjectID'] != 'Not_available']

df = df[[
    'subjectID',
    'session_name',
    'image_type',
    'diagnosis'
    ]]

# select tumour types interested in
tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Medulloblastoma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)'
    ]
 
df = df[df['diagnosis'].isin(tumour_types)]

subject_IDs = df['subjectID'].unique().tolist()

# %%
HE_stains = {
    "H&E": ["HandE", "HandE_B1", "HandE_A1", "H_and_E_A1", "H_and_E_B1", "HandE_C1", 
            "HandE_B2", "HandE_A", "HandE_B", "HandE_A2", "H_and_E_A2", "HandE_B3",
            "HandE_C2", "HandE_C", "HandE_D1", "H_and_E_C1", "HandE_A3", "H_and_E_B2",
            "HandE_", "HandE_FSA", "HandE_FS", "HandE_(B)", "HandE_B4", "HandE_C3", "HandE_A4",
            "HandE_E1", "1_B_HE", "HandE_1", "HandE_FS_(DGD)", "1_A_HE", "HandE_2", "HandE_B5",
            "H_and_E_FSA", "H_and_E_D1", "HandE_(A)", "1_C_HE", "HandE_(B2)", "H_and_E_A3", "HandE_(A1)",
            "HandE_D2", "1_D_HE", "H_and_E_FS", "HandE_D", "H_and_E_B3", "H_and_E_2", "HandE_(A2)", 
            "HandE_C4", "HandE_D3", "HandE_4", "H_and_E_DGD", "H_and_E_C2", "HandE_A6", "HandE_E2",
            "HandE_(2)", "HandE_F1", "HandE_G1", "HandE_3", "H_and_E_E1", "HandE_B8", "HandE_B6",
            "HandE_B7", "HandE_(B1)", "HandE_(1)", "H_and_E_B", "HandE_AFS", "HandE_(B3)", "HandE_(C2)",
            "1_E_HE", "H_and_E_3", "HandE_(C1)", "H_and_E_A", "HandE_FSB", "HandE_FS_DGD", "HandE_E",
            "HandE_Td", "HandE_(FS)", "H_and_E_D2", "FS_H_and_E", "2_A_HE", "HandE_A5", "HandE_(3)", 
            "H_and_E_1", "H_and_E_C3", "HandE_A7", "HandE_BLOCK_B", "1_F_HE", "H_and_E_C4", "HandE_BLOCK_B1",
            "HandE_6", "HandE_F2", "HandE_BLOCK_C", "H_and_E_B4", "HandE_(C3)", "HandE_D4",
            "HandE_5", "Hande_FS1", "H_and_E_Tf", "HandE_(C4)", "HandE_G", "HandE_PBSa",
            "HandE_(C)", "HandE_BLOCK_A1", "HandE_C5", "HandE_(A3)", "HandE_E3", "H_and_E_7",
            "HandE_3B", "H_and_E_DGD", "HandE_B9", "H_and_E_A4", "HandE_Te", "H_and_E_C",
            "HandE_Tc", "HandE_(D)", "HandEB1", "HandE_(A5)", "HandE_C8", "H_and_E_Th", 
            "H_and_E_F3", "HandE_BLOCK_A3", "HandE_B11", "HandE_(8)", "H_and_E_FS1", "1_H_HE", 
            "1_G_HE", "HandE_DGD", "HandE_12", "HandE_7", "HandE_C6", "HandE_(B4)", "HandE_(FSB)",
            "HandE_(D2)", "H_and_E_B6", "HandE_H", "HandE_2D", "FS1_H_and_E", "H_and_E_5", "HandE_(B5)",
            "H_and_E_C8", "H_and_E_G1", "H_and_E_6", "HandE_FSC", "HandE_F", "HandE_(FSA)", "HandE_2A",
            "HandE_2C", "HandE_BFS", "HandE_3A", "H_and_E_14", "HandE_BLOCK_B3", "HandE_B14", "HandE_B22",
            "2_B_HE", "H_and_E_4", "HandE_Oa", "HandE_FS_A", "HandE_B13", "HandE_B10", "H_and_E_A8", "HandEB2",
            "H_and_E_BSa", "H_and_E_Syn", "HandE_(D5)", "HandE_CSF", "HandE_506915", "HandE_(2515_3)", 
            "HandE_(2431-B)", "_HandE_D1", "H_and_E_RSH1", "HandE_B-R", "HandE_504953", "HandE_504954", 
            "HandE_504952", "HandE_A11", "H_and_E_A1_7716", "H_and_E___C2", "H_and_E__C2", "HandE_504950",
            "HandE_504951", "HandE_504949", "DGD_H_and_E", "HandEB10", "H_and_E_2A", "4992-he-004", "956_2A_HE",
            "956_2A_HE-003", "666-he-003", "3423_2C_HE", "666-he", "3477-he-003", "10412_1D_HE", "9980_1B_HE",
            "956_HE", "3477-he-001", "1184-2A_19-HHE", "5432-HE", "956_2A_HE-001", "666-he-009", "666-he-002", 
            "3423_2D_HE", "3477-he-005", "4992-he", "3477-he-004", "1_I_HE", "4492-2B_HE", "3423_1A_HE",
            "10412_1A_HE", "956_1A_HE-003", "10412_1B_HE", "4992-he-001", "4492-HE", "3_A_HE", "10412_1E_HE", 
            "666-he-005", "4992-he-002", "3423_1A_HE-001", "666-he-007", "666-he-006", "666-imp_he", "1184-2A_HHE", 
            "666-he-004", "10412_1F_HE", "956_1A_HE-002", "3423_2E_HE", "3423-1A_HE", "4992-he-003", "1184-2A_20-HHE",
            "3477-he_(2)", "956_2A_HE-002", "666-imp_he-001", "HEE_C1", "4992-he-005", "3477-he", "956_1A_HE-001", 
            "3477-he-002", "666-he-001", "956_1A_HE", "Hand_E_A2", "Hand_E_B", "HandD_17", "Hand_D2", "hande", "HANDE", 
            "H_and_E", "4492-2A_H&E_Lvl_1", "HandE_506913", "HandE_2X", "HandE_(S14-904)", "HandE_PERM_of_FS_", "HandE_F_R",
            "HandE_2H", "HandE_fma", "HandE_tg", "HandE_30", "HandE_BLOCK_1", "HandE_2Z", "HandE_B2_", 
            "HandD_17", "HandE_FSA_DGD", "HandE_BLOCK_D1", "HandE_BR1", "H_and_E_22", "HandE_B1", "H_and_E_", "1_E_HE",
            "H_and_E_3", "H_and_E_A", "HandE_(B3)", "H_and_E_D2", "FS_H_and_E", "2_A_HE", "H_and_E_1", "H_and_E_C3",
            "1_F_HE", "H_and_E_C4", "H_and_E_B4", "H_and_E_DGD_", "HandE_9", "H_and_E_____C2", "HandE_504961", "HandE_504962",
            "HandE_504963", "H_and_E_frozen", "H_and_E_A-F", "H_and_E_Fa", "H_and_E_E3", "H_and_E__C3", "HandE_29", "HandE_27",
            "HandE_28", "HandE_506912", "HandE_504958", "H_and_E_B5", "H_and_E_B7", "HandE_B21", "HandE_B19", "HandE_H6", "HandE_I",
            "HandE4", "HandE_3D", "TP1_H_and_E", "H_and_E_D-3", "H_and_E_D-1", "HandE_BLOCK_C2", "HandE_F3", "HandE_AFS_biop",
            "HandE_initial_tumor", "HandE_C1_S-12-56", "HandE_E4", "HandE_H4", "HandE_BLOCK_A9", "HandE_BLOCK_C5", "HandEC",
            "HandE_FS4", "HandE_FS3", "4492-HE", "HandE_FSA_2", "H_and_E_G2", "HandE_Bsa", "H_and_E_9", "HandE_th", "HandE_A10",
            "HandE_S-15-3720", "HandE_S-05-6044", "HandE_B2_S-15-3720", "H_and_E__B-02", "H_and_E_B-02", "H_and_E_B-01", "HandE_D6",
            "HandE_2T", "HandE_2BB_2", "HandE_2J", "HandE_2BB_1", "HandE_2BB", "HandE_2P", "HandE_2I", "HandE_2Q", "HandE_(C5)",
            "HandE_(C7)", "HandE_A14", "HandE_506887", "HandE_CR1", "HandE_BLOCK_A", "HandE_BLOCK_B2", "HandE_BLOCK_C3",
            "HandE_(5)", "HandE_(10)", "HandE_(14)", "HandE_(17)", "HandE_506909", "HandE_FSB_1", "HandE_H_and_EA1", "HandE_Tf",
            "HandE_D9", "HandE_506921", "HandE_I_2", "HandE_G2", "HandE_J1", "H_and_E_Tb", "S-09-1537_HandE", "S-08-5417_HandE",
            "HandE_G3", "HandE_1FS", "HandE_2F", "HandE_(A8)", "H_and_E_A1-2", "H_and_E_A2-3", "HandE_504956", "HandE_504957",
            "HandE_504955", "HandE_(S-08-2219)", "HandE_(S-14-1113)", "HandE_A1_2937", "H_and_E_A5", "HandE_fb", "H_and_E_D",
            "H_and_E_D13", "H_and_E_FSB", "HandE_FSA1", "HandE_DR1", "3_A_HE", "H_and_E_AFS", "HandE_506917", "HandE_S-12-681",
            "HandE_S-06-2617", "H_and_E_A16", "H_and_E_A15", "H_and_E_A17", "H_and_E_A7", "H_and_E_C6", "H_and_E_B2-2", "H_and_E_B1-2",
            "HandE_B2_deeper_section", "H_and_E_A11", "H_and_E_A13", "H_and_E_C-1", "H_and_E_C-3", "HandE_504948", "HandE_504959",
            "HandE_504960", "HandE_PBS_b", "HandE_TF", "HandE_F_1", "HandE_506908", "HandE_PBS_6", "HandE(B1)", "HandE_506914",
            "HandE_AR2", "HandE_AR1", "HandE_(13)", "HandE_(S14-765)", "HandE_FSA3", "HandE_FSA2", "H_and_E_F-1", "H_and_E_J-1",
            "HandEB_1", "HandE_BRS1", "H_and_E_SFA", "HandE_CS-14-399", "H_and_E__B2", "HandE_BLOCK_A4", "HandE_BLOCK_A2",
            "HandE_504966", "HandE_504965", "HandE_504964", "HandE_BLOCK_2", "_HandE_FSA_B1", "HandE_A_Perm", "H_and_E_DFS01",
            "HandE_AR", "H_and_E_C9", "HandE_H_and_E_A_FS", "FS_HandE_", "H_and_E_F", "H_and_E_F1", "H_and_E_FS01", "HandE_B_10",
            "HandE_(E)", "HandE_(4)", "H_and_E_G1-2", "H_and_E__A1", "HandE_F4", "H_and_E_B9", "HandE_A_4", "HandE_506920",
            "H_and_E_C5", "H_and_E5", "10412_1E_HE", "10412_1B_HE", "10412_1A_HE", "HandE_FSB_Permanent", "H_and_EA1",
            "H_and_E_A_FS", "HandE_FSA_DGD_"]
}

KI67_stains = {
    "KI-67": ["KI-67", "ki-67", "Ki-67", "KI67", "Ki067_A2", "Ki-62", "Ki-57", "Ki67", "Ki-67_B1", "H_and_E_Ki-67", "KI-67_A1", "KI-67_A2",
            "KI-67_C1", "KI-67_B", "KI-67_B2", "1_A_Ki67", "1_B_Ki67", "KI-67_B3", "ki-67_C2", "ki-67_D1", "KI-67_A3", "KI-67_C", "KI-67_",
            "ki-67_A4", "KI-67_A", "ki-67_(B)", "Ki-67_C3", "1_D_Ki67", "Ki-67_D", "Ki-67_2", "Ki-67_A5", "Ki-67_3", "Ki67_A1", "Ki067_A2",
            "Ki67_C2", "Ki-67_D2", "1_C_Ki67", "Ki-67_FSA", "Ki-67_1", "Ki-67_FS", "Ki-67_C4", "Ki-57", "2_A_Ki67", "Ki-67_(B2)", "Ki67_(D5)",
            "Ki-67_E1", "Ki-67__C2", "Ki-67_(2)", "Ki67_B1", "Ki-62", "1_F_Ki67", "Ki67_B3", "__Ki-67_D1", "9980_1B_KI67", "KI-67_BLOCK_C",
            "Ki-67_BLOCK_D", "4492-Ki67", "Ki-67_(C2)", "KI-67_B4", "Ki-67_(A2)", "Ki-67_(C3)", "Ki-67_A10", "Ki-67_S-05-6044", "Ki-67_B2",
            "666-ki67-001", "666-ki67", "KI67_BLOCK_A3", "KI67-_BLOCK_B1", "Ki-67_E", "2_B_Ki67", "4992-ki67", "4992-ki67-001", "4992-ki67-002",
            "KI67_C1", "Ki-67,_Ki-67", "Ki-67_B9", "5432-Ki67", "Ki-67_(S-08-2219)", "_Ki-67_B1", "Ki-67_(3)", "Ki67_B2", "Ki-67_(FS)", "Ki-67_B4",
            "Ki-67_B6", "3477-ki67", "3477-ki67-001", "Ki-67-A", "1_E_Ki67", "956_1A_KI67", "1184_-_2A_Ki67_MIB-1", "Ki-67_(S-14-904)", "Ki-67_FSB",
            "Ki-67_(D)", "Ki-67_(A)", "KI-67_BLOCK_2", "KI-67_BLOCK_1", "Ki-67_C9", "Ki-67_(A1)", "10412_1B_KI67"]
}

# %% COPY SVS FILES
i = 0
for subject_ID in subject_IDs:
    i += 1
    subject_path = os.path.join(histology_path, subject_ID, "SESSIONS")
    new_subject_path = os.path.join(new_histology_path, subject_ID)

    if os.path.exists(subject_path):
        sessions = os.listdir(subject_path)

        for session in sessions:
            session_path = os.path.join(subject_path, session, "ACQUISITIONS", "Files", "FILES")
            new_session_path = os.path.join(new_subject_path, session)

            if os.path.exists(session_path):

                for file in os.listdir(session_path):
                    if file.endswith(".svs"):
                        file_name_without_extension = os.path.splitext(file)[0]

                        if file_name_without_extension in HE_stains["H&E"]:
                            os.makedirs(new_session_path, exist_ok=True)

                            shutil.copy(os.path.join(session_path, file), os.path.join(new_session_path,  file))

    print('Subjects copied: {} / {}'.format(i, len(subject_IDs)))

# %% RENAME SVS FILES
subject_IDs = os.listdir(new_histology_path)
 
for subject_ID in subject_IDs:
    subject_path = os.path.join(new_histology_path, subject_ID)

    if os.path.exists(subject_path):
        sessions = os.listdir(subject_path)

        for session in sessions:
            session_path = os.path.join(subject_path, session)

            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".svs"):
                        os.rename(os.path.join(session_path, file), os.path.join(session_path, subject_ID + "___" + session + "___" + file))             
                            
# %% 
subject_IDs = os.listdir(new_histology_path)

subject_IDs_to_keep = [subject_ID for subject_ID in subject_IDs if not subject_ID.endswith(".svs")]
subject_IDs = subject_IDs_to_keep

for subject_ID in subject_IDs:
    subject_path = os.path.join(new_histology_path, subject_ID)

    if os.path.exists(subject_path):
        sessions = os.listdir(subject_path)

        for session in sessions:
            session_path = os.path.join(subject_path, session)

            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if file.endswith(".svs"):
                        shutil.copy(session_path + "/" + file, new_histology_path)     
    shutil.rmtree(subject_path)
  
# %%
