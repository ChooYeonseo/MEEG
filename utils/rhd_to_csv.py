"""
Convert RHD files to CSV format.
Merges all RHD files from a directory into a single CSV file with channel names as columns.
Uses streaming to handle large files without loading all data into RAM.
"""

import os
import sys
import glob
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm

# PyInstaller-aware path resolution
def get_base_path():
    """Get base path, works for both dev and PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent.parent

# Add the load-rhd-notebook-python directory to the Python path
rhd_utils_dir = get_base_path() / "load-rhd-notebook-python"
sys.path.insert(0, str(rhd_utils_dir))

# Import the RHD utilities
try:
    import importrhdutilities as rhd_utils
    print("Successfully imported RHD utilities")
except ImportError as e:
    print(f"Error importing RHD utilities: {e}")
    raise


def rhd_to_csv(file_dir, output_filename="merged_data.csv", chunk_size=10000):
    """
    Read all RHD files from a directory and merge them into a single CSV file.
    Streams data row-by-row to avoid memory issues with large files.
    
    Parameters:
    -----------
    file_dir : str
        Path to directory containing RHD files
    output_filename : str
        Name of the output CSV file (default: "merged_data.csv")
    chunk_size : int
        Number of samples to write at once (default: 10000)
    
    The CSV will have columns named after the channels, containing the amplifier data.
    Time data (t_amplifier) is excluded as requested.
    All RHD files are merged sequentially into a single CSV file.
    """
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"Directory not found: {file_dir}")
    
    # Find all RHD files in the directory
    rhd_files = glob.glob(os.path.join(file_dir, "*.rhd"))
    
    if not rhd_files:
        print(f"No RHD files found in {file_dir}")
        return
    
    # Sort files to ensure consistent ordering
    rhd_files = sorted(rhd_files)
    print(f"Found {len(rhd_files)} RHD file(s) in {file_dir}")
    
    # Output CSV path
    csv_path = os.path.join(file_dir, output_filename)
    
    # Check if output file already exists and ask for confirmation
    if os.path.exists(csv_path):
        print(f"Warning: {output_filename} already exists and will be overwritten.")
    
    channel_names = None
    total_samples = 0
    first_file = True
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = None
        
        # Process each RHD file
        for file_idx, file_path in enumerate(tqdm(rhd_files, desc="Merging RHD files", unit="file")):
            try:
                # Load the RHD file
                result, data_present = rhd_utils.load_file(file_path)
                
                if not data_present:
                    print(f"\n⚠ Skipping {os.path.basename(file_path)} - no data present")
                    continue
                
                # Check if amplifier data exists
                if 'amplifier_data' not in result or 'amplifier_channels' not in result:
                    print(f"\n⚠ Skipping {os.path.basename(file_path)} - no amplifier data")
                    continue
                
                # Get channel names from first file
                if first_file:
                    channel_names = [ch['native_channel_name'] for ch in result['amplifier_channels']]
                    csv_writer = csv.writer(csvfile)
                    # Write header
                    csv_writer.writerow(channel_names)
                    first_file = False
                    print(f"\nChannels: {channel_names}")
                    print(f"Number of channels: {len(channel_names)}")
                else:
                    # Verify channel consistency across files
                    current_channels = [ch['native_channel_name'] for ch in result['amplifier_channels']]
                    if current_channels != channel_names:
                        print(f"\n⚠ Warning: Channel mismatch in {os.path.basename(file_path)}")
                        print(f"  Expected: {channel_names}")
                        print(f"  Got: {current_channels}")
                        continue
                
                # Get amplifier data (shape: [n_channels, n_samples])
                amplifier_data = result['amplifier_data']
                n_channels, n_samples = amplifier_data.shape
                
                # Write data in chunks to avoid memory issues
                for start_idx in range(0, n_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_samples)
                    
                    # Get chunk of data and transpose (samples x channels)
                    chunk = amplifier_data[:, start_idx:end_idx].T
                    
                    # Write rows
                    csv_writer.writerows(chunk.tolist())
                
                total_samples += n_samples
                print(f"\n✓ Processed {os.path.basename(file_path)}: {n_samples:,} samples")
                
            except Exception as e:
                print(f"\n✗ Error processing {os.path.basename(file_path)}: {e}")
                continue
    
    if total_samples > 0:
        print(f"\n{'='*60}")
        print(f"✓ Conversion complete!")
        print(f"  Output file: {output_filename}")
        print(f"  Location: {file_dir}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Number of channels: {len(channel_names)}")
        print(f"  File size: {os.path.getsize(csv_path) / (1024**3):.2f} GB")
        print(f"{'='*60}")
    else:
        print("\n✗ No data was written to the CSV file.")
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"  Removed empty file: {output_filename}")


# Example usage
if __name__ == "__main__":
    # Set the directory containing RHD files
    file_dir = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/250903 overnight EEG A_WT+pilo B_KO+pilo 2k 1-300Hz (중간 끊김)/250903 overnight EEG A_WT+pilo B_KO+pilo 2k 1-300Hz_250903_170123"
    
    # Specify output filename (will be saved in the same directory as the RHD files)
    output_filename = "merged_data.csv"
    
    # Convert and merge all RHD files to a single CSV
    # chunk_size controls how many samples are written at once (adjust for memory constraints)
    rhd_to_csv(file_dir, output_filename=output_filename, chunk_size=10000)
