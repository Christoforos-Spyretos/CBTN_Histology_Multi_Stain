"""
Script that downscales the dimensions of the WSIs.
"""

# %% IMPORTS
import os
import glob
import fnmatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import openslide
from datetime import datetime

# %% RESIZE IMAGES 
def resize_img(basewidth,path, save_path):
    """
    INPUT
    basewidth: int
        Specify the desired width to which the images should be resized
    path : str
        Path to the folder that contains the HE images to be downscaled
    save_path : str
        Path to where the downscaled images should be saved
    """
    subjects = os.listdir(path)
    subjects_count = 0

    # count the subjects that their images already resized
    folders = 0
    for folder in os.listdir(save_path):
        folders += 1
    
    for subject_ID in subjects:

        # check if resized images already exist for this subject
        resized_subject_path = os.path.join(save_path, subject_ID)
        if os.path.exists(resized_subject_path):
            print(f"Resized images already exist for subject: {subject_ID}. Skipping...")
            continue

        start_time = datetime.now()
        subjects_count += 1
        print(f"Working on subject: {subject_ID}.")
        sessions = os.listdir(os.path.join(path, subject_ID, "SESSIONS"))
        print(f"Sessions to work on: {len(sessions)}.")

        for session in sessions:
            print(f"Working on session: {session}.")
            session_path = os.path.join(path, subject_ID, "SESSIONS", session)
            image = os.path.join(session_path, "ACQUISITIONS", "Files", "FILES", "*.svs")
            matching_images = glob.glob(image)
            print(f"Total images to be resized: {len(matching_images)}")

            for dirpath, dirnames, filenames in os.walk(session_path):
                for filename in fnmatch.filter(filenames, '*.svs'):
                    image_path = os.path.join(dirpath, filename)
                    slide = openslide.OpenSlide(image_path)
                    image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
                    # resize the image
                    wpercent = (basewidth/float(image.size[0]))
                    hsize = int((float(image.size[1])*float(wpercent)))
                    image = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
                    target_dir = os.path.join(save_path, subject_ID, session)
                    os.makedirs(target_dir, exist_ok=True)
                    save_filename = os.path.join(target_dir, filename.replace('.svs', '.png'))
                    image.save(save_filename, format='PNG', optimize=True, quality=90)

        end_time = datetime.now()

        print(f"Total time to resize all the images: {end_time-start_time}")
        print(f"Subjects that their images are resized: {folders + subjects_count}/{len(subjects)}.")
        print("---------------------------------------------------------------")

    print("Resizing images finished!")

# %%
basewidth = 1000
# path to imgs
histology_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS"
# path to save resized imgs
save_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/NEW_RESIZED_IMAGES"

resize_img(basewidth,histology_path,save_path)

