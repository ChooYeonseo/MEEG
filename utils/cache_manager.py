"""
Cache manager for EEG data.

This module provides functionality to save and load processed EEG data
as numpy arrays to/from cache files for faster subsequent access.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
from datetime import datetime


class CacheManager:
    """Manages caching of EEG data for faster subsequent loading."""
    
    def __init__(self, cache_dir="./cache"):
        """
        Initialize the cache manager.
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cache files
        """
        # Always resolve to absolute path to avoid creating cache folders in wrong locations
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(exist_ok=True)
        
    def _generate_cache_key(self, directory_path):
        """
        Generate a unique cache key based on directory path and modification times.
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing RHD files
            
        Returns:
        --------
        cache_key : str
            Unique identifier for this dataset
        """
        # Get all RHD files in the directory
        rhd_files = list(Path(directory_path).glob("*.rhd"))
        
        # Create a string with file paths and modification times
        file_info = []
        for file_path in sorted(rhd_files):
            mtime = file_path.stat().st_mtime
            file_info.append(f"{file_path.name}:{mtime}")
        
        # Create hash of the combined information
        combined_info = "|".join(file_info) + f"|{directory_path}"
        cache_key = hashlib.md5(combined_info.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_paths(self, cache_key):
        """
        Get paths for cache files.
        
        Parameters:
        -----------
        cache_key : str
            Unique identifier for the dataset
            
        Returns:
        --------
        data_path : Path
            Path to the cached data file
        metadata_path : Path
            Path to the cached metadata file
        """
        data_path = self.cache_dir / f"{cache_key}_data.npz"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        return data_path, metadata_path
    
    def is_cached(self, directory_path):
        """
        Check if data for a directory is already cached.
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing RHD files
            
        Returns:
        --------
        is_cached : bool
            True if cached data exists and is valid
        cache_key : str
            Cache key for the dataset
        """
        try:
            cache_key = self._generate_cache_key(directory_path)
            data_path, metadata_path = self._get_cache_paths(cache_key)
            
            return data_path.exists() and metadata_path.exists(), cache_key
        except Exception as e:
            # If we can't generate cache key (e.g., directory doesn't exist),
            # return False but still check if we have any cached projects
            print(f"Warning: Cannot access directory {directory_path} for cache validation: {e}")
            return False, None
    
    def save_to_cache(self, directory_path, results):
        """
        Save processed data to cache.
        
        Parameters:
        -----------
        directory_path : str
            Path to the original directory
        results : list
            List of tuples containing (filename, result_dict, data_present_bool)
        """
        cache_key = self._generate_cache_key(directory_path)
        data_path, metadata_path = self._get_cache_paths(cache_key)
        
        print(f"Saving data to cache: {cache_key}")
        
        # Prepare data for saving
        cached_data = {}
        metadata = {
            "directory_path": str(directory_path),
            "cache_timestamp": datetime.now().isoformat(),
            "num_files": len(results),
            "files": []
        }
        
        for i, (filename, result, data_present) in enumerate(results):
            file_key = f"file_{i}"
            
            # Store metadata about the file
            file_metadata = {
                "filename": filename,
                "data_present": data_present,
                "index": i
            }
            
            if data_present and result:
                # Extract and store the main data arrays
                if 'amplifier_data' in result:
                    cached_data[f"{file_key}_amplifier_data"] = result['amplifier_data']
                    file_metadata["has_amplifier_data"] = True
                    file_metadata["amplifier_shape"] = result['amplifier_data'].shape
                else:
                    file_metadata["has_amplifier_data"] = False
                
                # Store other important data
                if 'aux_input_data' in result:
                    cached_data[f"{file_key}_aux_input_data"] = result['aux_input_data']
                    file_metadata["has_aux_input_data"] = True
                    file_metadata["aux_input_shape"] = result['aux_input_data'].shape
                else:
                    file_metadata["has_aux_input_data"] = False
                
                # Store sample rate and other metadata
                if 'frequency_parameters' in result:
                    file_metadata["frequency_parameters"] = result['frequency_parameters']
                
                # Store channel information
                if 'amplifier_channels' in result:
                    file_metadata["amplifier_channels"] = result['amplifier_channels']
                
                # Store other metadata (non-array data)
                non_array_data = {}
                for key, value in result.items():
                    if not isinstance(value, np.ndarray) and key not in ['amplifier_channels', 'frequency_parameters']:
                        try:
                            # Test if it's JSON serializable
                            json.dumps(value)
                            non_array_data[key] = value
                        except (TypeError, ValueError):
                            # Skip non-serializable data
                            pass
                
                file_metadata["other_data"] = non_array_data
            
            metadata["files"].append(file_metadata)
        
        # Save the numpy arrays
        np.savez_compressed(data_path, **cached_data)
        
        # Save the metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Cache saved successfully: {data_path}")
        print(f"Metadata saved: {metadata_path}")
    
    def load_from_cache_by_key(self, cache_key):
        """
        Load data from cache using cache key directly (for cached projects).
        
        Parameters:
        -----------
        cache_key : str
            The cache key of the project to load
            
        Returns:
        --------
        results : list
            List of tuples containing (filename, result_dict, data_present_bool)
        """
        data_path, metadata_path = self._get_cache_paths(cache_key)
        
        if not (data_path.exists() and metadata_path.exists()):
            raise FileNotFoundError(f"Cache files not found for key: {cache_key}")
        
        print(f"Loading data from cache: {cache_key}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load numpy arrays
        cached_data = np.load(data_path)
        
        # Reconstruct the results
        results = []
        for file_meta in metadata["files"]:
            filename = file_meta["filename"]
            data_present = file_meta["data_present"]
            file_index = file_meta["index"]
            file_key = f"file_{file_index}"
            
            if data_present:
                # Reconstruct the result dictionary
                result = {}
                
                # Restore array data
                if file_meta.get("has_amplifier_data", False):
                    result['amplifier_data'] = cached_data[f"{file_key}_amplifier_data"]
                
                if file_meta.get("has_aux_input_data", False):
                    result['aux_input_data'] = cached_data[f"{file_key}_aux_input_data"]
                
                # Restore metadata
                if "frequency_parameters" in file_meta:
                    result['frequency_parameters'] = file_meta["frequency_parameters"]
                
                if "amplifier_channels" in file_meta:
                    result['amplifier_channels'] = file_meta["amplifier_channels"]
                
                # Restore other data
                if "other_data" in file_meta:
                    result.update(file_meta["other_data"])
                
                results.append((filename, result, data_present))
            else:
                results.append((filename, {}, data_present))
        
        print(f"Cache loaded successfully: {len(results)} files")
        return results
    
    def load_from_cache(self, directory_path):
        """
        Load data from cache.
        
        Parameters:
        -----------
        directory_path : str
            Path to the original directory
            
        Returns:
        --------
        results : list
            List of tuples containing (filename, result_dict, data_present_bool)
        """
        cache_key = self._generate_cache_key(directory_path)
        data_path, metadata_path = self._get_cache_paths(cache_key)
        
        print(f"Loading data from cache: {cache_key}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load numpy arrays
        cached_data = np.load(data_path)
        
        # Reconstruct the results
        results = []
        for file_meta in metadata["files"]:
            filename = file_meta["filename"]
            data_present = file_meta["data_present"]
            file_index = file_meta["index"]
            file_key = f"file_{file_index}"
            
            if data_present:
                # Reconstruct the result dictionary
                result = {}
                
                # Restore array data
                if file_meta.get("has_amplifier_data", False):
                    result['amplifier_data'] = cached_data[f"{file_key}_amplifier_data"]
                
                if file_meta.get("has_aux_input_data", False):
                    result['aux_input_data'] = cached_data[f"{file_key}_aux_input_data"]
                
                # Restore metadata
                if "frequency_parameters" in file_meta:
                    result['frequency_parameters'] = file_meta["frequency_parameters"]
                
                if "amplifier_channels" in file_meta:
                    result['amplifier_channels'] = file_meta["amplifier_channels"]
                
                # Restore other data
                if "other_data" in file_meta:
                    result.update(file_meta["other_data"])
                
                results.append((filename, result, data_present))
            else:
                results.append((filename, {}, data_present))
        
        print(f"Cache loaded successfully: {len(results)} files")
        return results
    
    def get_cached_project_info(self, cache_key):
        """
        Get information about a cached project without needing the original directory.
        
        Parameters:
        -----------
        cache_key : str
            Cache key of the project
            
        Returns:
        --------
        project_info : dict
            Dictionary containing project information
        """
        data_path, metadata_path = self._get_cache_paths(cache_key)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for cache key: {cache_key}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            "cache_key": cache_key,
            "directory_path": metadata["directory_path"],
            "cache_timestamp": metadata["cache_timestamp"],
            "num_files": metadata["num_files"],
            "metadata_file": str(metadata_path),
            "data_file": str(data_path),
            "cache_exists": data_path.exists() and metadata_path.exists()
        }

    def list_cached_projects(self):
        """
        List all cached projects.
        
        Returns:
        --------
        projects : list
            List of dictionaries containing project information
        """
        projects = []
        
        # Find all metadata files
        metadata_files = list(self.cache_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cache_key = metadata_file.stem.replace("_metadata", "")
                
                project_info = {
                    "cache_key": cache_key,
                    "directory_path": metadata["directory_path"],
                    "cache_timestamp": metadata["cache_timestamp"],
                    "num_files": metadata["num_files"],
                    "metadata_file": str(metadata_file)
                }
                
                projects.append(project_info)
                
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        projects.sort(key=lambda x: x["cache_timestamp"], reverse=True)
        
        return projects
    
    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
        print("Cache cleared successfully")
    
    def remove_cached_project(self, cache_key):
        """
        Remove a specific cached project.
        
        Parameters:
        -----------
        cache_key : str
            Cache key of the project to remove
        """
        data_path, metadata_path = self._get_cache_paths(cache_key)
        
        if data_path.exists():
            data_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        print(f"Removed cached project: {cache_key}")

# Global cache manager instance
# Use absolute path relative to the project root (parent of utils directory)
_project_root = Path(__file__).parent.parent
_cache_path = _project_root / "cache"
cache_manager = CacheManager(str(_cache_path))