import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import openslide

class WSIWrapper:
	"""Wrapper class to handle both OpenSlide and PIL-based WSI reading"""
	def __init__(self, wsi):
		self.wsi = wsi
		self.use_pil_fallback = False
		self.pil_image = None
		self.numpy_image = None  # Store as numpy array to avoid PIL temp file issues
		self.openslide_failed = False  # Track if OpenSlide read operations fail
		
		# Check if this is a PIL image or OpenSlide object
		if isinstance(wsi, Image.Image):
			self.use_pil_fallback = True
			self.pil_image = wsi
			self.wsi = None
			
			# Convert PIL image to numpy array to avoid temp file issues during crop
			try:
				print(f"PIL image initialized: {wsi.size}, converting to numpy array...")
				# Ensure image is loaded before converting
				wsi.load()
				self.numpy_image = np.array(wsi.convert('RGB'))
				print(f"Successfully converted to numpy array: {self.numpy_image.shape}")
				
				# Create synthetic pyramid levels for PIL images
				h, w = self.numpy_image.shape[:2]  # numpy shape is (height, width, channels)
				downsample_factors = [1.0, 4.0, 16.0, 32.0]
				self.level_dim = []
				for downsample in downsample_factors:
					width = int(w / downsample)
					height = int(h / downsample)
					self.level_dim.append((width, height))
				self.level_downsamples = [(factor, factor) for factor in downsample_factors]
				
			except Exception as e:
				print(f"Failed to convert PIL image to numpy: {e}")
				self.numpy_image = None
				# Fallback to original PIL approach
				w, h = self.pil_image.size
				downsample_factors = [1.0, 4.0, 16.0, 32.0]
				self.level_dim = []
				for downsample in downsample_factors:
					width = int(w / downsample)
					height = int(h / downsample)
					self.level_dim.append((width, height))
				self.level_downsamples = [(factor, factor) for factor in downsample_factors]
		else:
			# OpenSlide object - set up pyramid structure
			try:
				print(f"OpenSlide object initialized: {wsi.dimensions}")
				if hasattr(wsi, 'level_dimensions') and len(wsi.level_dimensions) == 1:
					# Only one level exists, create synthetic pyramid levels
					w, h = wsi.level_dimensions[0]
					downsample_factors = [1.0, 4.0, 16.0, 32.0]
					self.level_dim = []
					for downsample in downsample_factors:
						width = int(w / downsample)
						height = int(h / downsample)
						self.level_dim.append((width, height))
					self.level_downsamples = [(factor, factor) for factor in downsample_factors]
				else:
					# Multiple levels exist, use OpenSlide's pyramid
					self.level_dim = wsi.level_dimensions if hasattr(wsi, 'level_dimensions') else [(1000, 1000)]
					self.level_downsamples = self._assertLevelDownsamples()
			except Exception as e:
				print(f"Error during OpenSlide initialization: {e}")
				# Mark as failed but continue
				self.openslide_failed = True
				self.level_dim = [(1000, 1000)]
				self.level_downsamples = [(1.0, 1.0)]
	
	def _assertLevelDownsamples(self):
		"""Calculate level downsamples if not available"""
		if self.use_pil_fallback:
			return [(factor, factor) for factor in [1.0, 4.0, 16.0, 32.0]]
		
		try:
			if not hasattr(self.wsi, 'level_downsamples') or not hasattr(self.wsi, 'level_dimensions'):
				# Fallback to default downsamples
				return [(1.0, 1.0), (4.0, 4.0), (16.0, 16.0), (32.0, 32.0)]
			
			level_downsamples = []
			dim_0 = self.wsi.level_dimensions[0]
			
			for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
				estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
				level_downsamples.append(estimated_downsample)
			
			return level_downsamples
		except Exception as e:
			print(f"Error calculating level downsamples: {e}, using defaults")
			return [(1.0, 1.0), (4.0, 4.0), (16.0, 16.0), (32.0, 32.0)]
	
	def read_region(self, location, level, size):
		"""Read a region from the WSI at the given location, level, and size."""
		
		# If OpenSlide has been marked as failed, return blank image immediately
		if self.openslide_failed:
			return Image.new('RGB', size, (255, 255, 255))
		
		# Handle PIL fallback case
		if self.use_pil_fallback:
			# Use numpy array cropping to avoid PIL temporary file issues
			if self.numpy_image is not None:
				return self._read_region_numpy(location, level, size)
			else:
				return self._read_region_pil(location, level, size)
		
		# For OpenSlide files, handle both native pyramid and synthetic levels
		else:
			try:
				# Try OpenSlide read_region - handle both native and synthetic levels
				if hasattr(self.wsi, 'level_dimensions') and len(self.wsi.level_dimensions) > 1 and level < len(self.wsi.level_dimensions):
					# Use OpenSlide's native pyramid levels
					region = self.wsi.read_region(location, level, size)
				elif level == 0:
					# Level 0 - direct read
					region = self.wsi.read_region(location, 0, size)
				else:
					# Synthetic level - read from level 0 and resize
					downsample_factor = self.level_downsamples[level][0]
					
					# Adjust location and size for level 0
					level_0_location = (int(location[0] * downsample_factor), 
										int(location[1] * downsample_factor))
					level_0_size = (int(size[0] * downsample_factor), 
									int(size[1] * downsample_factor))
					
					# Read from level 0
					full_region = self.wsi.read_region(level_0_location, 0, level_0_size)
					
					# Resize to target level dimensions
					if downsample_factor > 1:
						region = full_region.resize(size, Image.LANCZOS)
					else:
						region = full_region
				
				# Convert to RGB and return
				return region.convert('RGB')
				
			except Exception as e:
				# OpenSlide failed, mark as failed and return blank image
				if not self.openslide_failed:
					print(f"OpenSlide read_region failed: {e}, switching to blank image mode")
					self.openslide_failed = True
				return Image.new('RGB', size, (255, 255, 255))
	
	def _read_region_numpy(self, location, level, size):
		"""Read region using numpy array cropping to avoid PIL temp file issues"""
		try:
			x, y = location
			w, h = size
			
			if level == 0:
				# Level 0 - crop from numpy array
				img_h, img_w = self.numpy_image.shape[:2]
				
				# Ensure coordinates are within bounds
				x = max(0, min(x, img_w - w))
				y = max(0, min(y, img_h - h))
				
				# Crop using numpy slicing
				cropped = self.numpy_image[y:y+h, x:x+w]
				
				# Convert back to PIL Image
				return Image.fromarray(cropped, 'RGB')
			else:
				# For higher levels, crop larger region and resize
				downsample_factor = self.level_downsamples[level][0]
				
				# Adjust location and size for level 0
				level_0_x = int(location[0] * downsample_factor)
				level_0_y = int(location[1] * downsample_factor)
				level_0_w = int(size[0] * downsample_factor)
				level_0_h = int(size[1] * downsample_factor)
				
				img_h, img_w = self.numpy_image.shape[:2]
				
				# Ensure coordinates are within bounds
				level_0_x = max(0, min(level_0_x, img_w - level_0_w))
				level_0_y = max(0, min(level_0_y, img_h - level_0_h))
				
				# Crop using numpy slicing
				cropped = self.numpy_image[level_0_y:level_0_y+level_0_h, level_0_x:level_0_x+level_0_w]
				
				# Convert to PIL and resize
				pil_region = Image.fromarray(cropped, 'RGB')
				if downsample_factor > 1:
					resized_region = pil_region.resize(size, Image.LANCZOS)
					return resized_region
				else:
					return pil_region
					
		except Exception as e:
			print(f"Numpy crop failed: {e}, returning blank image")
			return Image.new('RGB', size, (255, 255, 255))
	
	def _read_region_pil(self, location, level, size):
		"""Fallback PIL cropping method"""
		try:
			if level == 0:
				x, y = location
				w, h = size
				
				# Ensure coordinates are within bounds
				img_w, img_h = self.pil_image.size
				x = max(0, min(x, img_w - w))
				y = max(0, min(y, img_h - h))
				
				region = self.pil_image.crop((x, y, x + w, y + h))
				return region.convert('RGB')
			else:
				# For higher levels, read from level 0 and resize
				downsample_factor = self.level_downsamples[level][0]
				
				level_0_location = (int(location[0] * downsample_factor), 
									int(location[1] * downsample_factor))
				level_0_size = (int(size[0] * downsample_factor), 
								int(size[1] * downsample_factor))
				
				x, y = level_0_location
				w, h = level_0_size
				
				img_w, img_h = self.pil_image.size
				x = max(0, min(x, img_w - w))
				y = max(0, min(y, img_h - h))
				
				full_region = self.pil_image.crop((x, y, x + w, y + h))
				if downsample_factor > 1:
					resized_region = full_region.resize(size, Image.LANCZOS)
					return resized_region.convert('RGB')
				else:
					return full_region.convert('RGB')
					
		except Exception as e:
			print(f"PIL crop failed: {e}, returning blank image")
			return Image.new('RGB', size, (255, 255, 255))

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		# Wrap the WSI object to handle both OpenSlide and PIL cases
		self.wsi = WSIWrapper(wsi)
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size))

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




