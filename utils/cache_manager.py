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
import shutil


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
    
    def _get_cache_path(self, cache_key):
        """
        Get path for cache directory.
        
        Parameters:
        -----------
        cache_key : str
            Unique identifier for the dataset
            
        Returns:
        --------
        cache_path : Path
            Path to the cached data directory
        """
        return self.cache_dir / cache_key

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
            cache_path = self._get_cache_path(cache_key)
            
            # Check for new structure (directory with metadata)
            if cache_path.is_dir():
                metadata_path = cache_path / "metadata.json"
                return metadata_path.exists(), cache_key
            
            # Check for legacy structure (single files)
            legacy_data = self.cache_dir / f"{cache_key}_data.npz"
            legacy_meta = self.cache_dir / f"{cache_key}_metadata.json"
            if legacy_data.exists() and legacy_meta.exists():
                return True, cache_key
                
            return False, None
        except Exception as e:
            print(f"Warning: Cannot access directory {directory_path} for cache validation: {e}")
            return False, None

    def init_cache_session(self, directory_path):
        """
        Initialize a new cache session. Creates the directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to the original directory
            
        Returns:
        --------
        cache_key : str
            The cache key
        cache_path : Path
            Path to the created cache directory
        """
        cache_key = self._generate_cache_key(directory_path)
        cache_path = self._get_cache_path(cache_key)
        
        # If directory exists, clear it for fresh start
        if cache_path.exists():
            shutil.rmtree(cache_path)
            
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized cache session: {cache_key} at {cache_path}")
        return cache_key, cache_path

    def save_file_to_cache(self, cache_path, filename, index, result, data_present):
        """
        Save a single file's data to the cache directory.
        
        Parameters:
        -----------
        cache_path : Path
            Path to the cache directory
        filename : str
            Name of the file being saved
        index : int
            Index of the file in the sequence
        result : dict
            The processed result dictionary
        data_present : bool
            Whether data is present
            
        Returns:
        --------
        file_metadata : dict
            Metadata about this specific file to be stored in the main metadata.json
        """
        file_key = f"file_{index}"
        data_file_path = cache_path / f"{file_key}.npz"
        
        file_metadata = {
            "filename": filename,
            "data_present": data_present,
            "index": index,
            "data_file": data_file_path.name
        }
        
        if not data_present or not result:
            return file_metadata

        cached_data = {}
        
        # Extract and store the main data arrays
        if 'amplifier_data' in result:
            cached_data["amplifier_data"] = result['amplifier_data']
            file_metadata["has_amplifier_data"] = True
            file_metadata["amplifier_shape"] = result['amplifier_data'].shape
        else:
            file_metadata["has_amplifier_data"] = False
        
        # Store other important data
        if 'aux_input_data' in result:
            cached_data["aux_input_data"] = result['aux_input_data']
            file_metadata["has_aux_input_data"] = True
            file_metadata["aux_input_shape"] = result['aux_input_data'].shape
        else:
            file_metadata["has_aux_input_data"] = False
        
        # Store sample rate and other metadata
        if 'frequency_parameters' in result:
            file_metadata["frequency_parameters"] = result['frequency_parameters']
        
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
                    pass
        file_metadata["other_data"] = non_array_data

        # Save the numpy arrays if we have data
        if cached_data:
            np.savez_compressed(data_file_path, **cached_data)
            
        return file_metadata

    def finalize_cache_session(self, cache_path, directory_path, files_metadata):
        """
        Finalize the cache session by writing the main metadata file.
        
        Parameters:
        -----------
        cache_path : Path
            Path to the cache directory
        directory_path : str
            Original data directory path
        files_metadata : list
            List of file metadata dictionaries
        """
        metadata = {
            "directory_path": str(directory_path),
            "cache_timestamp": datetime.now().isoformat(),
            "num_files": len(files_metadata),
            "files": files_metadata,
            "format": "directory_v1"
        }
        
        metadata_path = cache_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Cache session finalized. Metadata saved to {metadata_path}")

    def save_to_cache(self, directory_path, results):
        """
        Legacy/Convenience method: Save all results to cache.
        Uses the new directory structure.
        
        Parameters:
        -----------
        directory_path : str
            Path to the original directory
        results : list
            List of tuples containing (filename, result_dict, data_present_bool)
        """
        cache_key, cache_path = self.init_cache_session(directory_path)
        
        print(f"Saving data to cache: {cache_key}")
        
        files_metadata = []
        for i, (filename, result, data_present) in enumerate(results):
            file_meta = self.save_file_to_cache(cache_path, filename, i, result, data_present)
            files_metadata.append(file_meta)
            
        self.finalize_cache_session(cache_path, directory_path, files_metadata)
        
        print(f"Cache saved successfully: {cache_path}")
    
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
        return self.load_from_cache_by_key(cache_key)

    def load_from_cache_by_key(self, cache_key):
        """
        Load data from cache using cache key directly.
        Handles both legacy (single file) and new (directory) formats.
        
        Parameters:
        -----------
        cache_key : str
            The cache key of the project to load
            
        Returns:
        --------
        results : list
            List of tuples containing (filename, result_dict, data_present_bool)
        """
        cache_path = self._get_cache_path(cache_key)
        
        # Check for new format (directory)
        if cache_path.is_dir():
            metadata_path = cache_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Cache metadata not found in {cache_path}")
            
            print(f"Loading data from cache directory: {cache_key}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            results = []
            
            for file_meta in metadata["files"]:
                filename = file_meta["filename"]
                data_present = file_meta["data_present"]
                
                result = {}
                if data_present:
                    data_file_path = cache_path / file_meta.get("data_file", f"file_{file_meta['index']}.npz")
                    
                    if data_file_path.exists():
                        cached_data = np.load(data_file_path)
                        
                        if file_meta.get("has_amplifier_data", False):
                            result['amplifier_data'] = cached_data['amplifier_data']
                        
                        if file_meta.get("has_aux_input_data", False):
                            result['aux_input_data'] = cached_data['aux_input_data']
                            
                    # Restore metadata
                    if "frequency_parameters" in file_meta:
                        result['frequency_parameters'] = file_meta["frequency_parameters"]
                    
                    if "amplifier_channels" in file_meta:
                        result['amplifier_channels'] = file_meta["amplifier_channels"]
                    
                    # Restore other data
                    if "other_data" in file_meta:
                        result.update(file_meta["other_data"])
                
                results.append((filename, result, data_present))
                
            print(f"Cache loaded successfully: {len(results)} files")
            return results

        # Fallback to legacy format (check if file exists with _data.npz suffix)
        legacy_data_path = self.cache_dir / f"{cache_key}_data.npz"
        legacy_meta_path = self.cache_dir / f"{cache_key}_metadata.json"
        
        if legacy_data_path.exists() and legacy_meta_path.exists():
            print(f"Loading data from legacy cache file: {cache_key}")
            with open(legacy_meta_path, 'r') as f:
                metadata = json.load(f)
            
            cached_data = np.load(legacy_data_path)
            
            results = []
            for file_meta in metadata["files"]:
                filename = file_meta["filename"]
                data_present = file_meta["data_present"]
                file_index = file_meta["index"]
                file_key = f"file_{file_index}"
                
                if data_present:
                    result = {}
                    if file_meta.get("has_amplifier_data", False):
                        result['amplifier_data'] = cached_data[f"{file_key}_amplifier_data"]
                    if file_meta.get("has_aux_input_data", False):
                        result['aux_input_data'] = cached_data[f"{file_key}_aux_input_data"]
                    if "frequency_parameters" in file_meta:
                        result['frequency_parameters'] = file_meta["frequency_parameters"]
                    if "amplifier_channels" in file_meta:
                        result['amplifier_channels'] = file_meta["amplifier_channels"]
                    if "other_data" in file_meta:
                        result.update(file_meta["other_data"])
                    results.append((filename, result, data_present))
                else:
                    results.append((filename, {}, data_present))
            
            return results
            
        raise FileNotFoundError(f"Cache not found for key: {cache_key}")

    def get_cached_project_info(self, cache_key):
        """
        Get information about a cached project.
        """
        cache_path = self._get_cache_path(cache_key)
        
        # Check directory format
        if cache_path.is_dir():
             metadata_path = cache_path / "metadata.json"
             if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    "cache_key": cache_key,
                    "directory_path": metadata["directory_path"],
                    "cache_timestamp": metadata["cache_timestamp"],
                    "num_files": metadata["num_files"],
                    "metadata_file": str(metadata_path),
                    "data_file": str(cache_path), # It's a directory now
                    "cache_exists": True
                }

        # Check legacy format
        legacy_meta_path = self.cache_dir / f"{cache_key}_metadata.json"
        legacy_data_path = self.cache_dir / f"{cache_key}_data.npz"
        
        if legacy_meta_path.exists():
             with open(legacy_meta_path, 'r') as f:
                metadata = json.load(f)
             return {
                "cache_key": cache_key,
                "directory_path": metadata["directory_path"],
                "cache_timestamp": metadata["cache_timestamp"],
                "num_files": metadata["num_files"],
                "metadata_file": str(legacy_meta_path),
                "data_file": str(legacy_data_path),
                "cache_exists": legacy_data_path.exists()
            }
            
        raise FileNotFoundError(f"Metadata file not found for cache key: {cache_key}")

    def list_cached_projects(self):
        """List all cached projects."""
        projects = []
        
        # Check directory-based caches
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        projects.append({
                            "cache_key": item.name,
                            "directory_path": metadata["directory_path"],
                            "cache_timestamp": metadata["cache_timestamp"],
                            "num_files": metadata["num_files"],
                            "metadata_file": str(metadata_path)
                        })
                    except Exception:
                        continue
        
        # Check legacy - avoid duplicates if we upgraded one
        legacy_metas = list(self.cache_dir.glob("*_metadata.json"))
        for meta_file in legacy_metas:
             try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                cache_key = meta_file.stem.replace("_metadata", "")
                
                # Deduplicate
                if not any(p['cache_key'] == cache_key for p in projects):
                    projects.append({
                        "cache_key": cache_key,
                        "directory_path": metadata["directory_path"],
                        "cache_timestamp": metadata["cache_timestamp"],
                        "num_files": metadata["num_files"],
                        "metadata_file": str(meta_file)
                    })
             except Exception:
                 continue

        projects.sort(key=lambda x: x["cache_timestamp"], reverse=True)
        return projects
    
    def clear_cache(self):
        """Clear all cached data."""
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
        print("Cache cleared successfully")
    
    def remove_cached_project(self, cache_key):
        """Remove a specific cached project."""
        cache_path = self._get_cache_path(cache_key)
        
        # Directory format
        if cache_path.is_dir():
            shutil.rmtree(cache_path)
            print(f"Removed cached project directory: {cache_key}")
            return
            
        # Legacy format
        legacy_data = self.cache_dir / f"{cache_key}_data.npz"
        legacy_meta = self.cache_dir / f"{cache_key}_metadata.json"
        
        if legacy_data.exists(): legacy_data.unlink()
        if legacy_meta.exists(): legacy_meta.unlink()
        print(f"Removed cached project files: {cache_key}")

# Global cache manager instance
# Use absolute path relative to the project root (parent of utils directory)
_project_root = Path(__file__).parent.parent
_cache_path = _project_root / "cache"
cache_manager = CacheManager(str(_cache_path))