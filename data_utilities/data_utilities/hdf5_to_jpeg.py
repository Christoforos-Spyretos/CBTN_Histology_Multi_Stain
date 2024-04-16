# %% IMPORTS
import os
import h5py
from PIL import Image
from datetime import datetime

# %% 
data_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBSET/PATCH/SUBJECTS"
save_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBSET/PATCHES/SUBJECTS"

# save pathches to disk as RGB
subjects = os.listdir(data_path)
total_subjects = len(subjects)
count_subjects = 0

start_time = datetime.now()

for subject_ID in subjects:
    print(f"Working on subject: {subject_ID}.")
    count_subjects += 1
    subject_path = os.path.join(data_path, subject_ID)

    count_sessions = 0
    count_hdf5 = 0
    
    for session in os.listdir(subject_path):
        count_sessions += 1 
        session_path = os.path.join(subject_path,session)
        for file in os.listdir(session_path):
            count_hdf5 += 1
    
    print(f"Total sessions to work on: {count_sessions}.")
    print(f"Total hdf5 files to work on: {count_hdf5}.")

    for session in os.listdir(subject_path):
        print(f"Workning on session: {session}.")
        session_path = os.path.join(subject_path,session)

        for file in os.listdir(session_path):
            print(f"Workning on image: {file.split('.')[0]}.")
            hdf5_file_path = os.path.join(session_path,file)
            
            # create the necessary directories for this session
            subject_session_file_save_path = os.path.join(save_path, subject_ID, session, file.split(".")[0])
            os.makedirs(subject_session_file_save_path, exist_ok=True)

            hdf5_file = h5py.File(hdf5_file_path, "r")
            print(f"Numpy arrays to be converted to jpeg: {hdf5_file['imgs'].shape[0]}")
            
            for idx in range(hdf5_file['imgs'].shape[0]):
                print(f'Saving {idx+1:0{len(str(hdf5_file["imgs"].shape[0]))}}/{ hdf5_file["imgs"].shape[0]} \r', end='')
                img = hdf5_file['imgs'][idx]
                img = Image.fromarray(img)
                # save
                img_name = os.path.join(subject_session_file_save_path, f'{os.path.basename(hdf5_file_path).split(".")[0]}_{hdf5_file["coords"][idx][0]}_{hdf5_file["coords"][idx][1]}.svs')
                img.save(img_name)
                
    print(f"Subjects processed: {count_subjects}/{total_subjects}.")
    print("------------------------------------------")  

end_time = datetime.now()

print("The process converting hdf5 files to jpegs is done!!!")         
print(f"Total time to convert: {end_time-start_time}")
# %%