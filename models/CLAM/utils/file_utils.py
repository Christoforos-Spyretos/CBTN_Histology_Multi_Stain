import pickle
import h5py
import time

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    max_retries = 5
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            with h5py.File(output_path, mode) as file:
                for key, val in asset_dict.items():
                    data_shape = val.shape
                    if key not in file:
                        data_type = val.dtype
                        chunk_shape = (chunk_size, ) + data_shape[1:]
                        maxshape = (None, ) + data_shape[1:]
                        dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                        dset[:] = val
                        if attr_dict is not None:
                            if key in attr_dict.keys():
                                for attr_key, attr_val in attr_dict[key].items():
                                    dset.attrs[attr_key] = attr_val
                    else:
                        dset = file[key]
                        dset.resize(len(dset) + data_shape[0], axis=0)
                        dset[-data_shape[0]:] = val
            return output_path
        except BlockingIOError as e:
            if attempt < max_retries - 1:
                print(f"HDF5 file locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to open HDF5 file after {max_retries} attempts")
                raise

def open_hdf5_with_retry(file_path, mode='r', max_retries=5):
    """Open HDF5 file with retry logic to handle file locking issues."""
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            return h5py.File(file_path, mode)
        except BlockingIOError as e:
            if attempt < max_retries - 1:
                print(f"HDF5 file locked, retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to open HDF5 file {file_path} after {max_retries} attempts")
                raise