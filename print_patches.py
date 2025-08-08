# %% IMPORTS
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove PIL's safety limit for large images

# %% Path 
path_to_patches = '/local/data3/chrsp39/CBTN_v2/new_KI67/test/224x224/patches/C1026189___7316-7923___Ki-67_A1.h5'
path_to_img = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_tif/C1026189___7316-7923___Ki-67_A1.tif'

# %% Load and print patches
def print_patches_info(h5_path):
    """Load and print information about patches in an H5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"H5 file: {h5_path}")
            print(f"Keys in H5 file: {list(f.keys())}")
            
            # Print information about each dataset
            for key in f.keys():
                dataset = f[key]
                print(f"\nDataset '{key}':")
                print(f"  Shape: {dataset.shape}")
                print(f"  Data type: {dataset.dtype}")
                
                # If it's likely image data, show some sample patches
                if len(dataset.shape) >= 3 and key in ['imgs', 'images', 'patches']:
                    print(f"  Number of patches: {dataset.shape[0]}")
                    print(f"  Patch dimensions: {dataset.shape[1:]}")
                    
                    # Display first few patches
                    num_to_show = min(9, dataset.shape[0])
                    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                    axes = axes.flatten()
                    
                    for i in range(num_to_show):
                        patch = dataset[i]
                        
                        # Handle different data types and ranges
                        if patch.dtype == np.uint8:
                            patch_display = patch
                        else:
                            # Normalize to 0-255 range for display
                            patch_min = patch.min()
                            patch_max = patch.max()
                            if patch_max > patch_min:
                                patch_display = ((patch - patch_min) / (patch_max - patch_min) * 255).astype(np.uint8)
                            else:
                                patch_display = patch.astype(np.uint8)
                        
                        axes[i].imshow(patch_display)
                        axes[i].set_title(f'Patch {i}')
                        axes[i].axis('off')
                    
                    # Hide unused subplots
                    for i in range(num_to_show, 9):
                        axes[i].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                # Print some sample data for coordinate datasets
                elif key in ['coords', 'coordinates']:
                    print(f"  Sample coordinates (first 5): {dataset[:5]}")
                    
    except FileNotFoundError:
        print(f"Error: File not found at {h5_path}")
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def extract_and_show_patches(img_path, coords_path, patch_size=224, num_patches=9):
    """Extract patches from TIFF image using coordinates from H5 file and display them."""
    try:
        # Load coordinates
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]
            print(f"Loaded {len(coords)} coordinates from {coords_path}")
        
        # Open the TIFF image
        print(f"Opening TIFF image: {img_path}")
        img = Image.open(img_path)
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        # Extract patches more efficiently without loading entire image
        num_to_show = min(num_patches, len(coords))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        print(f"Extracting and displaying {num_to_show} patches...")
        
        for i in range(num_to_show):
            x, y = coords[i]
            print(f"Patch {i}: extracting from coordinates ({x}, {y})")
            
            # Check if coordinates are within image bounds
            if x + patch_size > img.size[0] or y + patch_size > img.size[1]:
                print(f"  Warning: Patch {i} extends beyond image bounds")
                continue
            
            # Extract patch using crop (more memory efficient)
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            
            # Convert to RGB if necessary
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            
            # Convert to numpy array for display
            patch_array = np.array(patch)
            
            axes[i].imshow(patch_array)
            axes[i].set_title(f'Patch {i}\n({x}, {y})')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_to_show, 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Close the image to free memory
        img.close()
        
        return coords
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error extracting patches: {e}")
        return None

if __name__ == "__main__":
    # First show info about the H5 file
    print("=== H5 File Information ===")
    print_patches_info(path_to_patches)
    
    print("\n=== Extracting Actual Patches ===")
    # Then extract and show actual patches from the TIFF image
    coords = extract_and_show_patches(path_to_img, path_to_patches)

# %%
mask1 = '/local/data3/chrsp39/CBTN_v2/new_KI67/test/224x224/masks/C1026189___7316-7923___Ki-67_A1.jpg'
stitches1 = '/local/data3/chrsp39/CBTN_v2/new_KI67/test/224x224/stitches/C1026189___7316-7923___Ki-67_A1.png'


# Show masks and stitches
def show_mask_and_stitch(mask_path, stitch_path):
    """Display a mask and its corresponding stitched image."""
    try:
        mask = Image.open(mask_path)
        stitch = Image.open(stitch_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(mask)
        axes[0].set_title('Mask')
        axes[0].axis('off')
        
        axes[1].imshow(stitch)
        axes[1].set_title('Stitched Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error displaying images: {e}")

if __name__ == "__main__":
    print("=== Showing Masks and Stitches ===")
    show_mask_and_stitch(mask1, stitches1)

# %%