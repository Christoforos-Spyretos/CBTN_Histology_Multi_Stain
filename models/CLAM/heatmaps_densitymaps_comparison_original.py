from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    bg = 0
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
                
    return np.array(hist) / (h * w) 

def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img*255
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)

def earth_movers_distance(img_a, img_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
        {str} img_a: image array a
        {str} img_b: image array b
    @returns:
        wasserstein_distance between the two images
    '''
    img_a = normalize_exposure(img_a)
    img_b = normalize_exposure(img_b)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)

## Increase image reading size limit
Image.MAX_IMAGE_PIXELS = 1000000000

## Define working directories and files:
work_dir = os.getcwd()
QP_exec_file_loc = '/local/data1/juapa351/QuPath/QuPath/bin' # Set to correct location of QuPath executable file folder
QP_script_loc = work_dir + '/cellD_QuPath.groovy'
WSI_dir = '/local/data1/common/TCGA/HISTOLOGY/DIAGNOSTIC_SLIDES/' # Set to slide location
CLAM_hm_dir = 'heatmaps/heatmap_production_results/HEATMAP_OUTPUT/'
heatmap_save_folder = CLAM_hm_dir + 'cellDensity_hm_results' # If it does not exist it will be created
diff_maps_dir = CLAM_hm_dir + 'differential_maps/'

## Load wsi list and loop over it:
WSI_df = pd.read_csv('/local/data2/juapa351/CLAM_2024/heatmaps/process_lists/heatmap_demo_dataset.csv', index_col = False) # This can be any list with slide_id and label columns

## Create result dataframe:
res_df = pd.DataFrame(columns = ['slide', 'label', 'MSE', 'SSIM', 'EMD'])

for i in range(len(WSI_df)):

    image_name = WSI_df['slide_id'].iloc[i]
    check_save_dir = diff_maps_dir + image_name+ '/' + image_name 
    
    label = WSI_df['label'].iloc[i] # For CLAM attention map retrieval.
    
    image_loc = WSI_dir + image_name

    HM_radius = 100 # Set radius for cell density computation (higher radius = coarser heat map)
    pixel_size = 1.1 # Set pixel size in microns (higher size = faster computation, lower precision)

    HM_file = heatmap_save_folder + '/' + image_name + '_' + 'dmap.tif'

    ## Create current WSI directory:
    if not os.path.isdir(diff_maps_dir + image_name):
        os.makedirs(diff_maps_dir + image_name)

    if not os.path.isfile(HM_file):
    ## Build and run QuPath command if the cell density map is not already saved:
        os.chdir(QP_exec_file_loc)
        QP_command = './QuPath'  + ' script --image=' + image_loc + '.svs --args=[' + str(HM_radius) + ',' + str(pixel_size) + ',' + work_dir + ',' + heatmap_save_folder + '] ' + QP_script_loc
        os.system(QP_command)
        os.chdir(work_dir)

    ## Retrieve generated cell density raw value image:
    im = Image.open(HM_file)

    width, height = im.size
    ## Set color bar orientation depending on image dimensions
    if width < height:
        orient = 'vertical'
    else:
        orient = 'horizontal'

    ## Adjust values to optimize the visualization of the heat map (avoid highly localized maximum values):
    imarray = np.array(im)

    relevant_values = imarray[imarray > 0] # Only take into account foreground pixels
    n_relval = len(relevant_values)
    adjusted_n_relval = round(n_relval*0.95)

    sorted_relval = sorted(relevant_values)
    cm_lim = max(sorted_relval[:adjusted_n_relval])

    new_imarray = imarray/cm_lim # Normalize with obtained 95% highest value
    new_imarray[new_imarray>1] = 1 # Keep values inside normalized range for a correct comparison between heat maps

    ## Prepare cell density map for saving:
    cmapQ = plt.get_cmap('jet')
    A = cmapQ(new_imarray)

    fig, ax = plt.subplots()
    ## Assign colorbar:
    norm = mpl.colors.Normalize(vmin=0, vmax=cm_lim) 
    plt.imshow(A)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap='jet'), ax = ax, orientation = orient)

    ## Save heat map:
    plt.title('Cell density map')
    plt.savefig(diff_maps_dir + image_name+ '/'  + image_name + '_CellDM.png')
    cb.remove()
    plt.close("all")

    ## CLAM att. map retrieval:
    clam_hm_loc = CLAM_hm_dir + label + '/' + image_name + '_0.5_roi_0_blur_1_rs_1_bc_1_a_1.0_l_1_bi_0_-1.0.tif' # Change accordingly depending on CLAM attention map parameters.
    CLAM_im = ImageOps.grayscale(Image.open(clam_hm_loc))
    CLAM_imarray = np.array(CLAM_im)
    C_im_resized = CLAM_im.resize((width, height))
    C_imarray = np.array(C_im_resized)
    CL_normd_array = C_imarray/np.max(C_imarray)

    ## Save CLAM resized attention map:
    cmapC = plt.get_cmap('jet')
    C = cmapC(CL_normd_array) 
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(C_imarray))
    plt.imshow(C)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap='jet'), ax = ax, orientation = orient)

    plt.title('Attention map')
    plt.savefig(diff_maps_dir + image_name+ '/' + image_name +'_AttM.png')
    cb.remove()
    plt.close("all")

    ### Heat map comparison:
    ## Flatten arrays and exclude background to compute the MSE:
    flat_new_imarray = new_imarray.flatten()
    ind = np.argwhere(flat_new_imarray > 0)
    Cflat_n_imarray = flat_new_imarray[ind]
    flat_CD_array = CL_normd_array.flatten()
    Cflat_CD_array = flat_CD_array[ind]
    mse = mean_squared_error(Cflat_n_imarray, Cflat_CD_array)

    ## Compute SSIM, emd and difference map:
    str_sim = ssim(new_imarray, CL_normd_array, data_range=1)
    idiff = np.absolute(new_imarray - CL_normd_array)
    diff = idiff/np.max(idiff)
    emd = earth_movers_distance(CL_normd_array, new_imarray)

    print('Comparing heat maps for ' + image_name + ':')
    print('Mean squared error: ' + str(mse))
    print('Structural similarity index: ' + str(str_sim))
    print('Earth movers distance (Wasserstein): ' + str(emd) + '\n')
    res_df = res_df.append({'slide': image_name, 'label': label, 'MSE': mse, 'SSIM': str_sim, 'EMD': emd}, ignore_index=True)

    ## Save difference map:
    cmapC = plt.get_cmap('jet')
    B = cmapC(diff)
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(idiff))
    plt.imshow(B)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap='jet'), ax = ax, orientation = orient)

    plt.title('Difference map')
    plt.savefig(diff_maps_dir + image_name+ '/' + image_name + '_DiffM.png')
    cb.remove()
    plt.close("all")

    ### Code to save the cell density map with original colormap range:
    # max_val = np.max(imarray)

    # full_new_imarray = imarray/max_val # Use this command instead of the next one to use the original raw value range

    # cmapQ = plt.get_cmap('jet')
    # D = cmapQ(full_new_imarray)

    # # Set figure show timer:
    # fig, ax = plt.subplots()#layout='constrained')
    # timer = fig.canvas.new_timer(interval = 3000) # Adjust desired showing time in ms
    # timer.add_callback(close_event)

    # # Assign colorbar:
    # norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
    # plt.imshow(D)
    # cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap='jet'), ax = ax, orientation = "horizontal")

    # # Save and show heat map:
    # plt.title('Cell density map')
    # plt.savefig(diff_maps_dir + image_name+ '/'  + image_name + label + '_CellDM_full.png')
    # cb.remove()
    # plt.close("all")

    del fig,ax, cb
## Save result csv.
res_df.to_csv(diff_maps_dir + 'HM_comp_results.csv', index=False)