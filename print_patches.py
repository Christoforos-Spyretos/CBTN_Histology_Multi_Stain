# %% IMPORTS
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
Image.MAX_IMAGE_PIXELS = None  # Remove PIL's safety limit for large images

# %% Path 
path_to_patches = '/local/data3/chrsp39/CBTN_v2/new_KI67/224x224_tif/patches/C2786688___7316-6785___Ki-67.h5'
path_to_img = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_tif/C2786688___7316-6785___Ki-67.tif'

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
                    
                    # Display 9 random patches
                    num_patches = dataset.shape[0]
                    if num_patches > 0:
                        indices = random.sample(range(num_patches), min(9, num_patches))
                        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                        axes = axes.flatten()
                        
                        for i, patch_idx in enumerate(indices):
                            patch = dataset[patch_idx]
                            
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
                            axes[i].set_title(f'Patch {patch_idx}')
                            axes[i].axis('off')
                        
                        # Hide unused subplots
                        for i in range(len(indices), 9):
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
        num_patches = len(coords)
        indices = random.sample(range(num_patches), min(num_patches, 9))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        print(f"Extracting and displaying 9 random patches...")
        
        for i, patch_idx in enumerate(indices):
            x, y = coords[patch_idx]
            print(f"Patch {i} (index {patch_idx}): extracting from coordinates ({x}, {y})")
            
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
            axes[i].set_title(f'Patch {patch_idx}\n({x}, {y})')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), 9):
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

# %% NUMPY APPROACH EXAMPLE
def demonstrate_numpy_approach(img_path, coords_path, patch_size=224, num_patches=4):
    """
    Demonstrate the numpy approach for patch extraction from TIF files.
    This approach avoids PIL temporary file issues by working with numpy arrays.
    """
    print("NUMPY APPROACH DEMONSTRATION")
    
    try:
        # Step 1: Load coordinates
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]
            print(f"Loaded {len(coords)} coordinates")
        
        # Step 2: Open TIF image with PIL (this loads metadata but not full image data yet)
        print(f"Opening TIF image: {img_path}")
        with Image.open(img_path) as pil_img:
            print(f"  - Image size: {pil_img.size}")
            print(f"  - Image mode: {pil_img.mode}")
            
            # Step 3: Convert PIL image to numpy array (this is the key step!)
            print("Converting PIL image to numpy array (loading full image into memory)...")
            img_array = np.array(pil_img)
            print(f"  - Numpy array shape: {img_array.shape}")
            print(f"  - Array data type: {img_array.dtype}")
            print(f"  - Memory usage: ~{img_array.nbytes / (1024*1024):.1f} MB")
            
            # Step 4: Extract patches using numpy array slicing (no file I/O!)
            print("Extracting patches using numpy array slicing...")
            
            # Select random patches to demonstrate
            selected_indices = random.sample(range(len(coords)), min(num_patches, len(coords)))
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            
            for i, patch_idx in enumerate(selected_indices):
                x, y = coords[patch_idx]
                print(f"  - Patch {i+1}: extracting from coordinates ({x}, {y})")
                
                # Check bounds
                if y + patch_size > img_array.shape[0] or x + patch_size > img_array.shape[1]:
                    print(f"Warning: Patch extends beyond image bounds, skipping")
                    continue
                
                # THIS IS THE NUMPY MAGIC: Direct array slicing instead of PIL crop
                # No temporary files, no file I/O, just pure memory operations
                patch_array = img_array[y:y+patch_size, x:x+patch_size]
                
                print(f"Extracted patch shape: {patch_array.shape}")
                
                # Display the patch
                axes[i].imshow(patch_array)
                axes[i].set_title(f'Patch {patch_idx}\nNumpy Slice [{y}:{y+patch_size}, {x}:{x+patch_size}]')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(selected_indices), 4):
                axes[i].axis('off')
            
            plt.suptitle('Patches Extracted Using Numpy Array Slicing\n(No PIL crop() calls, no temporary files)', 
                        fontsize=14)
            plt.tight_layout()
            plt.show()
            
            print("Numpy approach completed successfully!")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error in numpy approach: {e}")
        import traceback
        traceback.print_exc()

def compare_pil_vs_numpy_approach(img_path, coords_path, patch_size=224, num_comparisons=4):
    """
    Compare PIL crop approach vs Numpy slicing approach side by side for multiple random patches.
    """
    print("PIL vs NUMPY APPROACH COMPARISON")
    
    try:
        # Load coordinates
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]
        
        # Select random coordinates for comparison
        selected_indices = random.sample(range(len(coords)), min(num_comparisons, len(coords)))
        print(f"Testing with {len(selected_indices)} random patches:")
        for i, idx in enumerate(selected_indices):
            x, y = coords[idx]
            print(f"  Patch {i+1}: coordinates ({x}, {y})")
        
        with Image.open(img_path) as pil_img:
            print(f"\nImage size: {pil_img.size}, mode: {pil_img.mode}")
            
            # Convert to numpy once for all numpy operations
            print("Converting PIL image to numpy array for numpy method...")
            img_array = np.array(pil_img)
            print(f"Numpy array shape: {img_array.shape}")
            
            # Create figure for side-by-side comparisons
            fig, axes = plt.subplots(num_comparisons, 2, figsize=(12, num_comparisons * 3))
            if num_comparisons == 1:
                axes = axes.reshape(1, -1)
            
            pil_successes = 0
            numpy_successes = 0
            identical_results = 0
            
            for i, patch_idx in enumerate(selected_indices):
                x, y = coords[patch_idx]
                print(f"\n--- PATCH {i+1} at ({x}, {y}) ---")
                
                # Check bounds
                if y + patch_size > img_array.shape[0] or x + patch_size > img_array.shape[1]:
                    print(f"⚠ Warning: Patch extends beyond image bounds, skipping")
                    axes[i, 0].text(0.5, 0.5, 'Out of bounds', ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'Out of bounds', ha='center', va='center', transform=axes[i, 1].transAxes)
                    axes[i, 0].set_title(f'PIL crop() - Patch {patch_idx}')
                    axes[i, 1].set_title(f'Numpy slice - Patch {patch_idx}')
                    continue
                
                # METHOD 1: PIL crop() approach
                pil_success = False
                pil_array = None
                try:
                    box = (x, y, x + patch_size, y + patch_size)
                    pil_patch = pil_img.crop(box)  # This might create temporary files!
                    pil_array = np.array(pil_patch)
                    print(f"PIL crop successful: {pil_array.shape}")
                    pil_success = True
                    pil_successes += 1
                except Exception as e:
                    print(f"PIL crop failed: {e}")
                
                # METHOD 2: Numpy slicing approach
                numpy_success = False
                numpy_patch = None
                try:
                    numpy_patch = img_array[y:y+patch_size, x:x+patch_size]  # Pure memory operation!
                    print(f"Numpy slicing successful: {numpy_patch.shape}")
                    numpy_success = True
                    numpy_successes += 1
                except Exception as e:
                    print(f"Numpy slicing failed: {e}")
                
                # Display results
                if pil_success and pil_array is not None:
                    axes[i, 0].imshow(pil_array)
                    axes[i, 0].set_title(f'PIL crop() - Patch {patch_idx}\nShape: {pil_array.shape}')
                else:
                    axes[i, 0].text(0.5, 0.5, 'PIL Failed', ha='center', va='center', transform=axes[i, 0].transAxes, color='red')
                    axes[i, 0].set_title(f'PIL crop() - Patch {patch_idx}\n FAILED')
                
                if numpy_success and numpy_patch is not None:
                    axes[i, 1].imshow(numpy_patch)
                    axes[i, 1].set_title(f'Numpy slice - Patch {patch_idx}\nShape: {numpy_patch.shape}')
                else:
                    axes[i, 1].text(0.5, 0.5, 'Numpy Failed', ha='center', va='center', transform=axes[i, 1].transAxes, color='red')
                    axes[i, 1].set_title(f'Numpy slice - Patch {patch_idx}\n FAILED')
                
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                
                # Compare if both succeeded
                if pil_success and numpy_success and pil_array is not None and numpy_patch is not None:
                    if np.array_equal(pil_array, numpy_patch):
                        print(f"Both methods produce IDENTICAL results!")
                        identical_results += 1
                    else:
                        print(f"Results differ (this would be unexpected)")

            plt.suptitle(f'PIL crop() vs Numpy slicing - {num_comparisons} Random Patches\n'
                        f'PIL successes: {pil_successes}/{num_comparisons} | '
                        f'Numpy successes: {numpy_successes}/{num_comparisons} | '
                        f'Identical results: {identical_results}/{min(pil_successes, numpy_successes)}', 
                        fontsize=14)
            plt.tight_layout()
            plt.show()
                        
            if pil_successes < num_comparisons:
                print(f"PIL failed on {num_comparisons - pil_successes} patches - likely TIF temporary file issues")
            if numpy_successes == num_comparisons:
                print(f"Numpy approach succeeded on all patches - no file I/O issues!")
                
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # First show info about the H5 file
    print("=== H5 File Information ===")
    print_patches_info(path_to_patches)
    
    print("\n=== Extracting Actual Patches ===")
    # Then extract and show actual patches from the TIFF image
    coords = extract_and_show_patches(path_to_img, path_to_patches)
    
    # NEW: Demonstrate the numpy approach
    print("\n" + "="*60)
    demonstrate_numpy_approach(path_to_img, path_to_patches)
    
    # NEW: Compare PIL vs Numpy approaches
    print("\n" + "="*60)
    compare_pil_vs_numpy_approach(path_to_img, path_to_patches)

# %%
mask1 = '/local/data3/chrsp39/CBTN_v2/new_KI67/224x224_tif/masks/C2786688___7316-6785___Ki-67.jpg'
stitches1 = '/local/data3/chrsp39/CBTN_v2/new_KI67/224x224_tif/stitches/C2786688___7316-6785___Ki-67.png'


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