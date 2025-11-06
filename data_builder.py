"""
Data Builder for Intan RHD Files

This script reads Intan RHD files, applies resampling to 256Hz for specific channels
(A-017 or B-017), chunks the data into 30-second epochs, and saves them as NPZ files.

Usage:
    python data_builder.py --source_dir <path_to_rhd_files> --target_dir <path_to_save_npz>
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Import the read_intan utilities
from utils.read_intan import read_rhd_directory, get_sample_rate, resample_dataframe


def get_next_file_number(target_dir, prefix):
    """
    Get the next available file number for a given prefix.
    
    Parameters:
    -----------
    target_dir : str
        Directory where files will be saved
    prefix : str
        Prefix of the file (e.g., 'A_17_256' or 'B_17_256')
        
    Returns:
    --------
    next_number : int
        Next available file number
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return 0
    
    # Find all files matching the pattern
    pattern = os.path.join(target_dir, f"{prefix}_*.npz")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 0
    
    # Extract numbers from filenames
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove prefix and extension
        try:
            number_part = filename.replace(f"{prefix}_", "").replace(".npz", "")
            numbers.append(int(number_part))
        except ValueError:
            continue
    
    if not numbers:
        return 0
    
    return max(numbers) + 1


def chunk_dataframe(df, epoch_duration=30.0, sample_rate=256):
    """
    Chunk a DataFrame into fixed-duration epochs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'time' column and channel data
    epoch_duration : float
        Duration of each epoch in seconds (default: 30.0)
    sample_rate : float
        Sampling rate in Hz
        
    Returns:
    --------
    epochs : list of pandas.DataFrame
        List of DataFrames, each containing one epoch
    """
    samples_per_epoch = int(epoch_duration * sample_rate)
    total_samples = len(df)
    num_epochs = total_samples // samples_per_epoch
    
    epochs = []
    for i in range(num_epochs):
        start_idx = i * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        
        epoch_df = df.iloc[start_idx:end_idx].copy()
        # Reset time to start from 0 for each epoch
        epoch_df['time'] = epoch_df['time'] - epoch_df['time'].iloc[0]
        epochs.append(epoch_df)
    
    # Handle remaining samples if any
    remaining_samples = total_samples % samples_per_epoch
    if remaining_samples > 0:
        print(f"Warning: {remaining_samples} samples ({remaining_samples/sample_rate:.2f} seconds) "
              f"remaining after chunking. These will be discarded.")
    
    return epochs


def save_epoch_as_npz(epoch_df, file_path):
    """
    Save an epoch DataFrame as an NPZ file.
    
    Parameters:
    -----------
    epoch_df : pandas.DataFrame
        DataFrame containing one epoch of data
    file_path : str
        Path where the NPZ file will be saved
    """
    # Convert DataFrame to dictionary of numpy arrays
    numpy_data = {}
    for column in epoch_df.columns:
        numpy_data[column] = epoch_df[column].values
    
    # Save as compressed numpy archive
    np.savez_compressed(file_path, **numpy_data)


def process_rhd_file(rhd_file_info, target_dir, channel_labels, target_sample_rate=256, epoch_duration=30.0):
    """
    Process a single RHD file: check for A-017/B-017 channels, resample, chunk, and save.
    
    Parameters:
    -----------
    rhd_file_info : tuple
        Tuple of (filename, result_dict, data_present_bool)
    target_dir : str
        Directory where NPZ files will be saved
    channel_labels : dict
        Dictionary mapping 'A-017' and 'B-017' to their labels (e.g., {'A-017': 'WT', 'B-017': 'KO+pilo'})
    target_sample_rate : float
        Target sampling rate in Hz (default: 256)
    epoch_duration : float
        Duration of each epoch in seconds (default: 30.0)
        
    Returns:
    --------
    num_epochs_saved : int
        Number of epochs saved for this file
    """
    filename, result, data_present = rhd_file_info
    
    if not data_present:
        print(f"Skipping {filename} - no data present")
        return 0
    
    # Extract channel data
    if 'amplifier_channels' not in result or 'amplifier_data' not in result:
        print(f"Error: No amplifier data found in {filename}")
        return 0
    
    # Check which channels (A-017 or B-017) exist in this file
    channels_to_process = []
    available_channels = [ch['native_channel_name'] for ch in result['amplifier_channels']]
    
    if 'A-017' in available_channels and 'A-017' in channel_labels:
        label = channel_labels['A-017']
        prefix = f"{label}_256"
        channels_to_process.append(('A-017', prefix, label))
    if 'B-017' in available_channels and 'B-017' in channel_labels:
        label = channel_labels['B-017']
        prefix = f"{label}_256"
        channels_to_process.append(('B-017', prefix, label))
    
    if not channels_to_process:
        print(f"Skipping {filename} - does not contain A-017 or B-017 channels")
        print(f"Available channels: {available_channels[:5]}..." if len(available_channels) > 5 else f"Available channels: {available_channels}")
        return 0
    
    print(f"\nProcessing {filename}")
    print(f"Found channels to process: {[ch[0] for ch in channels_to_process]}")
    
    # Get sample rate
    original_sample_rate = get_sample_rate(result)
    if original_sample_rate is None:
        print(f"Error: Could not determine sample rate for {filename}")
        return 0
    
    print(f"Original sample rate: {original_sample_rate} Hz")
    
    total_epochs_saved = 0
    
    # Process each channel found
    for channel_name, prefix, label in channels_to_process:
        print(f"\n  Processing channel: {channel_name} (Label: {label})")
        
        # Find the channel index
        channel_index = None
        for i, ch in enumerate(result['amplifier_channels']):
            if ch['native_channel_name'] == channel_name:
                channel_index = i
                break
        
        if channel_index is None:
            print(f"  Error: Channel {channel_name} not found (this shouldn't happen)")
            continue
        
        # Extract channel data
        amplifier_data = result['amplifier_data']
        if channel_index >= amplifier_data.shape[0]:
            print(f"  Error: Channel index {channel_index} out of range")
            continue
        
        channel_data = amplifier_data[channel_index, :]
        num_samples = len(channel_data)
        
        # Create time vector
        duration = num_samples / original_sample_rate
        time_vector = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_vector,
            channel_name: channel_data
        })
        
        print(f"  Loaded {num_samples} samples ({duration:.2f} seconds)")
        
        # Resample if needed
        if original_sample_rate != target_sample_rate:
            print(f"  Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz...")
            df = resample_dataframe(df, original_rate=original_sample_rate, target_rate=target_sample_rate)
            print(f"  Resampled to {len(df)} samples")
        
        # Chunk the data into epochs
        print(f"  Chunking data into {epoch_duration}-second epochs...")
        epochs = chunk_dataframe(df, epoch_duration=epoch_duration, sample_rate=target_sample_rate)
        print(f"  Created {len(epochs)} epochs")
        
        # Get the starting file number
        start_number = get_next_file_number(target_dir, prefix)
        
        # Save each epoch
        num_epochs_saved = 0
        for i, epoch_df in enumerate(epochs):
            file_number = start_number + i
            output_filename = f"{prefix}_{file_number}.npz"
            output_path = os.path.join(target_dir, output_filename)
            
            save_epoch_as_npz(epoch_df, output_path)
            num_epochs_saved += 1
        
        print(f"  Saved {num_epochs_saved} epochs for {channel_name} (Label: {label})")
        print(f"  File numbers: {start_number} to {start_number + num_epochs_saved - 1}")
        total_epochs_saved += num_epochs_saved
    
    return total_epochs_saved


def process_directory(source_dir, target_dir, channel_labels, target_sample_rate=256, epoch_duration=30.0):
    """
    Process all RHD files in a directory.
    
    Parameters:
    -----------
    source_dir : str
        Directory containing RHD files
    target_dir : str
        Directory where NPZ files will be saved
    channel_labels : dict
        Dictionary mapping 'A-017' and 'B-017' to their labels (e.g., {'A-017': 'WT', 'B-017': 'KO+pilo'})
    target_sample_rate : float
        Target sampling rate in Hz (default: 256)
    epoch_duration : float
        Duration of each epoch in seconds (default: 30.0)
    """
    print(f"\n{'='*70}")
    print(f"Processing RHD files from: {source_dir}")
    print(f"Saving NPZ files to: {target_dir}")
    print(f"Channel labels: A-017 = {channel_labels.get('A-017', 'Not specified')}, "
          f"B-017 = {channel_labels.get('B-017', 'Not specified')}")
    print(f"Target sample rate: {target_sample_rate} Hz")
    print(f"Epoch duration: {epoch_duration} seconds")
    print(f"{'='*70}\n")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Read all RHD files from the directory
    print("Reading RHD files...")
    rhd_files = read_rhd_directory(source_dir)
    
    if not rhd_files:
        print(f"No RHD files found in {source_dir}")
        return
    
    # Process each file
    total_epochs = 0
    files_processed = 0
    
    for rhd_file_info in rhd_files:
        num_epochs = process_rhd_file(
            rhd_file_info,
            target_dir,
            channel_labels=channel_labels,
            target_sample_rate=target_sample_rate,
            epoch_duration=epoch_duration
        )
        if num_epochs > 0:
            total_epochs += num_epochs
            files_processed += 1
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Files processed: {files_processed}")
    print(f"Total epochs saved: {total_epochs}")
    print(f"{'='*70}\n")


def main():
    """Main function to parse arguments and run the data builder."""
    parser = argparse.ArgumentParser(
        description='Process Intan RHD files and save as NPZ epochs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_builder.py --source_dir /path/to/rhd --target_dir /path/to/npz \\
      --label_a WT --label_b "KO+pilo"
  
  python data_builder.py --source_dir /path/to/rhd --target_dir /path/to/npz \\
      --label_a "WT+pilo" --label_b "KO+pilo"
        """
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Path to directory containing RHD files'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        help='Path to directory where NPZ files will be saved'
    )
    parser.add_argument(
        '--label_a',
        type=str,
        required=True,
        choices=['WT', 'WT+pilo', 'KO+pilo', 'KO+pilo2'],
        help='Label for A-017 channel (WT, WT+pilo, KO+pilo, or KO+pilo2)'
    )
    parser.add_argument(
        '--label_b',
        type=str,
        required=True,
        choices=['WT', 'WT+pilo', 'KO+pilo', 'KO+pilo2'],
        help='Label for B-017 channel (WT, WT+pilo, KO+pilo, or KO+pilo2)'
    )
    parser.add_argument(
        '--sample_rate',
        type=float,
        default=256,
        help='Target sampling rate in Hz (default: 256)'
    )
    parser.add_argument(
        '--epoch_duration',
        type=float,
        default=30.0,
        help='Duration of each epoch in seconds (default: 30.0)'
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist: {args.source_dir}")
        sys.exit(1)
    
    # Create channel labels dictionary
    channel_labels = {
        'A-017': args.label_a,
        'B-017': args.label_b
    }
    
    # Process the directory
    process_directory(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        channel_labels=channel_labels,
        target_sample_rate=args.sample_rate,
        epoch_duration=args.epoch_duration
    )


if __name__ == "__main__":
    main()
