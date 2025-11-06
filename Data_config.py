import numpy as np
import json
import os
from pathlib import Path

def load_npz_file_info(cache_folder="cache"):
    """
    Load and analyze NPZ files from the cache folder.
    Shows the structure and content of cached data files.
    """
    cache_path = Path(cache_folder)
    
    if not cache_path.exists():
        print(f"Cache folder '{cache_folder}' not found!")
        return
    
    # Get all NPZ files
    npz_files = list(cache_path.glob("*.npz"))
    
    if not npz_files:
        print("No NPZ files found in cache folder!")
        return
    
    print(f"Found {len(npz_files)} NPZ files in cache folder:")
    print("-" * 60)
    
    for i, npz_file in enumerate(npz_files):
        print(f"\n{i+1}. File: {npz_file.name}")
        
        # Load the NPZ file
        try:
            data = np.load(npz_file)
            
            print(f"   Keys in NPZ file: {list(data.keys())}")
            
            # # Analyze each array in the NPZ file
            # for key in data.keys():
            #     array = data[key]
            #     print(f"   '{key}':")
            #     print(f"     - Shape: {array.shape}")
            #     print(f"     - Data type: {array.dtype}")
            #     print(f"     - Size: {array.size:,} elements")
            #     print(f"     - Memory usage: {array.nbytes / (1024**2):.2f} MB")
                
            #     # Show sample values for small arrays or first few values for large arrays
            #     if array.size <= 10:
            #         print(f"     - Values: {array}")
            #     else:
            #         if array.ndim == 1:
            #             print(f"     - First 5 values: {array[:5]}")
            #             print(f"     - Last 5 values: {array[-5:]}")
            #         elif array.ndim == 2:
            #             print(f"     - First row, first 5 values: {array[0, :5]}")
            #             print(f"     - Shape details: {array.shape[0]} rows × {array.shape[1]} columns")
            #         else:
            #             print(f"     - Multi-dimensional array with {array.ndim} dimensions")
                
            #     # Statistical info for numerical data
            #     if np.issubdtype(array.dtype, np.number) and array.size > 0:
            #         print(f"     - Min value: {np.min(array)}")
            #         print(f"     - Max value: {np.max(array)}")
            #         print(f"     - Mean value: {np.mean(array):.6f}")
            #         print(f"     - Standard deviation: {np.std(array):.6f}")
            
            # data.close()
            
        except Exception as e:
            print(f"   Error loading file: {e}")
        
        # Also load corresponding metadata if available
        metadata_file = npz_file.with_suffix('.json').name.replace('_data.json', '_metadata.json')
        metadata_path = cache_path / metadata_file
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"   \nMetadata available:")
                print(f"     - Directory: {metadata.get('directory_path', 'N/A')}")
                print(f"     - Cache timestamp: {metadata.get('cache_timestamp', 'N/A')}")
                print(f"     - Number of files: {metadata.get('num_files', 'N/A')}")
                
                if 'files' in metadata and len(metadata['files']) > 0:
                    first_file = metadata['files'][0]
                    print(f"     - First file: {first_file.get('filename', 'N/A')}")
                    if 'amplifier_shape' in first_file:
                        print(f"     - Amplifier data shape: {first_file['amplifier_shape']}")
                    if 'frequency_parameters' in first_file:
                        freq_params = first_file['frequency_parameters']
                        print(f"     - Sample rate: {freq_params.get('amplifier_sample_rate', 'N/A')} Hz")
                        print(f"     - Bandwidth: {freq_params.get('actual_lower_bandwidth', 'N/A')}-{freq_params.get('actual_upper_bandwidth', 'N/A')} Hz")
                
            except Exception as e:
                print(f"   Error loading metadata: {e}")
        
        print("-" * 60)

def load_specific_npz(filename, cache_folder="cache"):
    """
    Load a specific NPZ file and return its contents.
    
    Args:
        filename (str): Name of the NPZ file to load
        cache_folder (str): Path to cache folder
    
    Returns:
        dict: Dictionary containing the loaded arrays
    """
    cache_path = Path(cache_folder)
    file_path = cache_path / filename
    
    if not file_path.exists():
        print(f"File '{filename}' not found in cache folder!")
        return None
    
    try:
        data = np.load(file_path)
        result = {key: data[key] for key in data.keys()}
        data.close()
        return result
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        return None

def analyze_data_structure():
    """
    Main function to analyze the data structure of cached NPZ files.
    """
    print("MEEG Data Structure Analysis")
    print("=" * 60)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    # Analyze all NPZ files
    load_npz_file_info()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("\nTo load a specific file, use:")
    print("data = load_specific_npz('filename.npz')")

if __name__ == "__main__":
    # Run the analysis
    analyze_data_structure()
