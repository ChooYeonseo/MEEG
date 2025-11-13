"""
Data reader thread for EEG analysis.

This module provides the DataReaderThread class that handles reading
EEG data in a separate thread to prevent GUI freezing.
"""

import sys
import time
import traceback
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

try:
    import read_intan
    from cache_manager import cache_manager
except ImportError as e:
    print(f"Error importing read_intan utilities: {e}")
    traceback.print_exc()


class DataReaderThread(QThread):
    """Thread for reading EEG data without blocking the GUI."""
    
    progress_update = pyqtSignal(str)  # Status message
    data_loaded = pyqtSignal(object)   # Data results
    error_occurred = pyqtSignal(str)   # Error message
    finished_loading = pyqtSignal()    # Loading complete
    
    def __init__(self, path_or_files, use_cache=True, save_cache=True, file_type='rhd', sampling_rate=None):
        super().__init__()
        self.path_or_files = path_or_files  # Directory path for RHD, list of files for CSV
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.file_type = file_type  # 'rhd' or 'csv'
        self.sampling_rate = sampling_rate  # Required for CSV files
        self.output_capture = None
        
    def set_output_capture(self, output_capture):
        """Set the output capture object."""
        self.output_capture = output_capture
        
    def run(self):
        """Run the data reading process."""
        try:
            # Redirect stdout to capture print statements
            if self.output_capture:
                sys.stdout = self.output_capture
                sys.stderr = self.output_capture
            
            file_type_upper = self.file_type.upper()
            
            if self.file_type == 'rhd':
                print(f"Starting to read {file_type_upper} files from: {self.path_or_files}")
            else:
                print(f"Starting to read {len(self.path_or_files)} {file_type_upper} file(s)")
            print("=" * 50)
            
            # Check if we should use cache (only for RHD files for now)
            if self.use_cache and self.file_type == 'rhd':
                is_cached, cache_key = cache_manager.is_cached(self.path_or_files)
                if is_cached:
                    self.progress_update.emit("Loading from cache...")
                    print(f"Found cached data for directory. Loading from cache...")
                    results = cache_manager.load_from_cache(self.path_or_files)
                    print(f"Successfully loaded {len(results)} files from cache!")
                    print("=" * 50)
                    self.data_loaded.emit(results)
                    return
            
            # Read from original files based on file type
            self.progress_update.emit(f"Reading {file_type_upper} files...")
            
            if self.file_type == 'rhd':
                # Use the read_intan module to read RHD files
                results = read_intan.read_rhd_directory(self.path_or_files)
                file_description = "RHD files"
            elif self.file_type == 'csv':
                # Use CSV reading functionality with file list
                results = self.read_csv_files(self.path_or_files, self.sampling_rate)
                file_description = "CSV files"
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")
            
            if not results:
                print(f"No {file_description} found or no data could be read.")
                self.data_loaded.emit([])
            else:
                print(f"\nSuccessfully loaded {len(results)} {file_description}!")
                
                # Save to cache if requested (only for RHD files for now)
                if self.save_cache and self.file_type == 'rhd':
                    self.progress_update.emit("Saving to cache...")
                    print("Saving data to cache for faster future access...")
                    cache_manager.save_to_cache(self.path_or_files, results)
                    print("Cache saved successfully!")
                
                print("=" * 50)
                self.data_loaded.emit(results)
                
        except Exception as e:
            error_msg = f"Error reading data: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            
        finally:
            # Restore original stdout/stderr
            if self.output_capture:
                sys.stdout = self.output_capture.original_stdout
                sys.stderr = self.output_capture.original_stderr
            self.finished_loading.emit()
    
    def read_csv_files(self, csv_file_paths, sampling_rate):
        """Read CSV files from a list of file paths.
        Uses memory-efficient chunked reading and numpy arrays to prevent crashes.
        
        Args:
            csv_file_paths: List of absolute paths to CSV files
            sampling_rate: Sampling frequency in Hz
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        results = []
        
        if not csv_file_paths:
            print("No CSV files provided")
            return results
        
        print(f"Reading {len(csv_file_paths)} CSV file(s)")
        print(f"Using sampling rate: {sampling_rate} Hz")
        
        for csv_file_path in csv_file_paths:
            csv_file = Path(csv_file_path)
            try:
                print(f"Reading: {csv_file.name}...")
                
                # Read CSV file in chunks to avoid memory issues
                # First, get the column names and number of rows
                first_chunk = pd.read_csv(csv_file, nrows=1)
                channel_names = first_chunk.columns.tolist()
                n_channels = len(channel_names)
                
                print(f"  Found {n_channels} channels: {channel_names}")
                self.progress_update.emit(f"Found {n_channels} channels in {csv_file.name}")
                
                # Count total rows efficiently
                self.progress_update.emit(f"Counting samples in {csv_file.name}...")
                with open(csv_file, 'r') as f:
                    n_samples = sum(1 for _ in f) - 1  # -1 for header
                
                print(f"  Total samples: {n_samples:,}")
                self.progress_update.emit(f"Loading {n_samples:,} samples from {csv_file.name}...")
                
                # Pre-allocate numpy array for efficiency (float32 for memory optimization)
                amplifier_data = np.empty((n_channels, n_samples), dtype=np.float32)
                
                # Read CSV in chunks and populate the numpy array
                chunk_size = 50000  # Read 50k rows at a time
                row_offset = 0
                
                print(f"  Reading in chunks...")
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size, dtype=np.float32):
                    chunk_rows = len(chunk)
                    
                    # Round to 4 decimal places to reduce memory and computation load
                    # This is sufficient for EEG data and prevents numerical precision issues
                    chunk_data = np.round(chunk.values.T, decimals=4).astype(np.float32)
                    
                    # Place in pre-allocated array
                    amplifier_data[:, row_offset:row_offset + chunk_rows] = chunk_data
                    row_offset += chunk_rows
                    
                    # Emit progress signal to keep GUI responsive
                    progress_pct = (row_offset / n_samples) * 100
                    self.progress_update.emit(f"Loading {csv_file.name}: {progress_pct:.1f}% ({row_offset:,}/{n_samples:,} samples)")
                    
                    # Small sleep to allow PyQt6 event loop to process signals
                    # This prevents the GUI from freezing and PyQt6 from aborting
                    time.sleep(0.001)
                    
                    # Show progress in console less frequently
                    if row_offset % (chunk_size * 5) == 0:
                        print(f"    Progress: {row_offset:,}/{n_samples:,} samples ({progress_pct:.1f}%)")
                
                print(f"  ✓ Successfully loaded into numpy array")
                
                # Create channel info similar to RHD format
                amplifier_channels = [
                    {'native_channel_name': name, 'custom_channel_name': name}
                    for name in channel_names
                ]
                
                # Create time vector
                t_amplifier = np.arange(n_samples, dtype=np.float32) / sampling_rate
                
                # Create a result structure similar to RHD format
                result = {
                    'amplifier_data': amplifier_data,
                    'amplifier_channels': amplifier_channels,
                    't_amplifier': t_amplifier,
                    'frequency_parameters': {
                        'amplifier_sample_rate': sampling_rate
                    },
                    'notes': {
                        'note1': f'Loaded from CSV: {csv_file.name}',
                        'note2': f'Sampling rate: {sampling_rate} Hz'
                    },
                    'data_format': 'csv'  # Mark as CSV-originated
                }
                
                results.append((csv_file.name, result, True))
                duration = n_samples / sampling_rate
                memory_mb = amplifier_data.nbytes / (1024**2)
                print(f"  ✓ Loaded {n_channels} channels, {n_samples:,} samples ({duration:.2f}s)")
                print(f"  Memory usage: {memory_mb:.2f} MB")
                
            except Exception as e:
                print(f"  ✗ Error reading {csv_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append((csv_file.name, None, False))
        
        return results


class CacheLoaderThread(QThread):
    """Thread for loading cached EEG data."""
    
    progress_update = pyqtSignal(str)  # Status message
    data_loaded = pyqtSignal(object)   # Data results
    error_occurred = pyqtSignal(str)   # Error message
    finished_loading = pyqtSignal()    # Loading complete
    
    def __init__(self, cache_key):
        super().__init__()
        self.cache_key = cache_key
        self.output_capture = None
        
    def set_output_capture(self, output_capture):
        """Set the output capture object."""
        self.output_capture = output_capture
        
    def run(self):
        """Run the cache loading process."""
        try:
            # Redirect stdout to capture print statements
            if self.output_capture:
                sys.stdout = self.output_capture
                sys.stderr = self.output_capture
            
            self.progress_update.emit("Loading cached project...")
            print(f"Loading cached project: {self.cache_key}")
            print("=" * 50)
            
            # Load directly from cache using cache key
            results = cache_manager.load_from_cache_by_key(self.cache_key)
            
            print(f"Successfully loaded {len(results)} files from cache!")
            print("=" * 50)
            self.data_loaded.emit(results)
                
        except Exception as e:
            error_msg = f"Error loading cached data: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
            
        finally:
            # Restore original stdout/stderr
            if self.output_capture:
                sys.stdout = self.output_capture.original_stdout
                sys.stderr = self.output_capture.original_stderr
            self.finished_loading.emit()
