"""
Data reader thread for EEG analysis.

This module provides the DataReaderThread class that handles reading
EEG data in a separate thread to prevent GUI freezing.
"""

import sys
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
    
    def __init__(self, directory, use_cache=True, save_cache=True):
        super().__init__()
        self.directory = directory
        self.use_cache = use_cache
        self.save_cache = save_cache
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
            
            print(f"Starting to read RHD files from: {self.directory}")
            print("=" * 50)
            
            # Check if we should use cache
            if self.use_cache:
                is_cached, cache_key = cache_manager.is_cached(self.directory)
                if is_cached:
                    self.progress_update.emit("Loading from cache...")
                    print(f"Found cached data for directory. Loading from cache...")
                    results = cache_manager.load_from_cache(self.directory)
                    print(f"Successfully loaded {len(results)} files from cache!")
                    print("=" * 50)
                    self.data_loaded.emit(results)
                    return
            
            # Read from original files
            self.progress_update.emit("Reading RHD files...")
            
            # Use the read_intan module to read the directory
            results = read_intan.read_rhd_directory(self.directory)
            
            if not results:
                print("No RHD files found or no data could be read.")
                self.data_loaded.emit([])
            else:
                print(f"\nSuccessfully loaded {len(results)} RHD files!")
                
                # Save to cache if requested
                if self.save_cache:
                    self.progress_update.emit("Saving to cache...")
                    print("Saving data to cache for faster future access...")
                    cache_manager.save_to_cache(self.directory, results)
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
