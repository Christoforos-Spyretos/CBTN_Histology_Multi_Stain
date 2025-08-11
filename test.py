# %% IMPORTS 
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = 933120000

# %% PATH TO IMAGES
img_svs = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_svs/C15375___7316-46___Ki-67.svs'


# %% OPEN IMAGES 
# svs image
slide_svs = openslide.OpenSlide(img_svs)

level_downsamples = slide_svs.level_downsamples
level_dim = slide_svs.level_dimensions

print(f'Level downsample svs: {level_downsamples}')
print(f'Level dimensions svs: {level_dim}')
print(f'Last level dimension svs: {level_dim[-1]}')

w, h = level_dim[-1]
print(f'Last level dimension svs: {w}, {h}')
print(f'Size of svs image: {w*h}')

scale = slide_svs.level_downsamples[-1]
print(f'Scale of svs image: {scale}')

best_level = slide_svs.get_best_level_for_downsample(64)
print(f'Best level for downsample 64: {best_level}')

# %%
img_tif = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI/C2442411___7316-7082___Ki-67_E1.tif'

# tif image
slide_tif = Image.open(img_tif)
print(f'Size of tif image: {slide_tif.size}')

w = slide_tif.size[0]
h = slide_tif.size[1]

print(f'Size of tif image: {w*h}')

downsample_factors = [1.0, 4.0, 16, 32]
tif_pyramid_levels = []

for downsample in downsample_factors:
    width = int(w / downsample)  
    height = int(h / downsample)
    tif_pyramid_levels.append((width, height))

print(f'TIF pyramid levels: {tif_pyramid_levels}')

# downsample tif image to match svs levels
tif_downsampled = []
for level in range(len(downsample_factors)):
    downsampled_image = slide_tif.resize(tif_pyramid_levels[level], Image.LANCZOS)
    tif_downsampled.append(downsampled_image)

w, h = tif_downsampled[-1].size
print(f'Last level dimension tif: {w}, {h}')
print(f'Size of downsampled tif image: {w*h}')

# Display the downsampled images
for i, img in enumerate(tif_downsampled):
    print(f'TIF Downsampled Level {i+1}: {img.size}')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(f'TIF Downsampled Level {i+1}')
    plt.axis('off')
    plt.show()

# %% OPEN IMAGE
import openslide
import matplotlib.pyplot as plt
from PIL import Image

path = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI/C2442411___7316-7082___Ki-67_E1.tif'

slide_tif = openslide.OpenSlide(path)
print(f'Opened TIF image: {path}')

level_downsamples = slide_tif.level_downsamples
level_dim = slide_tif.level_dimensions

print(f'Level downsample tif: {level_downsamples}')
print(f'Level dimensions tif: {level_dim}')
print(f'Last level dimension tif: {level_dim[-1]}')

w, h = level_dim[-1]
print(f'Last level dimension tif: {w}, {h}')
print(f'Size of tif image: {w*h}')

scale = slide_tif.level_downsamples[-1]
print(f'Scale of tif image: {scale}')

best_level = slide_tif.get_best_level_for_downsample(64)
print(f'Best level for downsample 64: {best_level}')

# downsample tif image by 4, 16, 32
downsample_factors = [1.0, 4.0, 16, 32]
tif_pyramid_levels = []     

for downsample in downsample_factors:
    width = int(w / downsample)  
    height = int(h / downsample)
    tif_pyramid_levels.append((width, height))

print(f'TIF pyramid levels: {tif_pyramid_levels}')

# print the downsampled images
for i, (target_width, target_height) in enumerate(tif_pyramid_levels):
    print(f'TIF Downsampled Level {i+1}: {(target_width, target_height)}')
    
    # Read the full resolution image and resize it
    if i == 0:
        # For the first level, use level 0 (full resolution)
        downsampled_image = slide_tif.read_region((0, 0), 0, slide_tif.level_dimensions[0])
        downsampled_image = downsampled_image.resize((target_width, target_height), Image.LANCZOS)
    else:
        # For other levels, read from level 0 and resize
        full_image = slide_tif.read_region((0, 0), 0, slide_tif.level_dimensions[0])
        downsampled_image = full_image.resize((target_width, target_height), Image.LANCZOS)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(downsampled_image)
    plt.title(f'TIF Downsampled Level {i+1}: {target_width}x{target_height}')
    plt.axis('off')
    plt.show()

# %%
