# %% IMPORTS
import os
import glob
import fnmatch
import argparse
import yaml
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import openslide
from datetime import datetime

# %% LOAD CONFIG
parser = argparse.ArgumentParser(description='Resize WSI images')
parser.add_argument('--config', type=str, 
                    default='configs/data_utilities/resize_images.yaml',
                    help='Path to config file')
args = parser.parse_args()

# load configuration from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# %% LOAD PATHS
img_path = config['img_path']
save_path = config['save_path']
basewidth = config['basewidth']

print(f"Configuration loaded from: {args.config}")
print(f"Input path: {img_path}")
print(f"Output path: {save_path}")
print(f"Base width: {basewidth}")
print("------------------------------------------------------------")

if not os.path.exists(save_path):
    os.makedirs(save_path)

# %% RESIZE IMAGES
slide_ids = os.listdir(img_path)
print(f"Total number of slides to be resized: {len(slide_ids)}.")

resized_slide_ids = os.listdir(save_path)
print(f"Total number of slides already resized: {len(resized_slide_ids)}.")

slide_ids_count = 0

start_time = datetime.now()

for slide_id in slide_ids:
    slide_name = os.path.splitext(slide_id)[0]
    resized_slide_id_path = os.path.join(save_path, slide_name + ".png")
    if os.path.exists(resized_slide_id_path):
        print(f"Resized slide already exist: {slide_id}. Skipping...")
        slide_ids_count += 1
        print("------------------------------------------------------------")
        continue

    slide_ids_count += 1
    print(f"Working on slide: {slide_id}.")

    slide_id_path = os.path.join(img_path, slide_id)

    try:
        # try opening with OpenSlide first
        slide = openslide.OpenSlide(slide_id_path)
        image = slide.read_region((0, 0), 0, slide.level_dimensions[0])
        print(f"Successfully opened with OpenSlide: {slide_id}")
    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
        # if OpenSlide fails, try with PIL Image
        print(f"OpenSlide failed for {slide_id}, trying with PIL Image...")
        try:
            image = Image.open(slide_id_path)
            print(f"Successfully opened with PIL Image: {slide_id}")
        except Exception as e:
            print(f"Failed to open {slide_id} with both OpenSlide and PIL Image: {e}")
            print("------------------------------------------------------------")
            continue

    # resize the image
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)

    # Use the same method as the check above to ensure consistency
    save_filename = os.path.join(save_path, slide_name + ".png")    
    image.save(save_filename, format='PNG', optimize=True, quality=90)

    print(f"Slides that are resized: {slide_ids_count}/{len(slide_ids)}.")
    print("------------------------------------------------------------")

end_time = datetime.now()

print(f"Total time to resize all the slides: {end_time-start_time}")
print("Resizing slides finished!")

# %%