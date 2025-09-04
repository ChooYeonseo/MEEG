"""
Common utilities for the EEG Analysis GUI application.

This module contains utility functions and helper classes that are used
across multiple components of the application.
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def setup_python_path():
    """Add utils directory to Python path for importing read_intan module."""
    current_dir = Path(__file__).parent
    utils_dir = current_dir / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))


def format_timestamp():
    """Get formatted timestamp for terminal output."""
    return datetime.now().strftime("%H:%M:%S")


def format_terminal_message(text):
    """Format a message for terminal display with timestamp."""
    if text.strip():
        timestamp = format_timestamp()
        return f"[{timestamp}] {text}"
    return text


def get_default_directory():
    """Get the default directory for file selection."""
    return os.path.expanduser("~")


def validate_directory(directory):
    """Validate if directory exists and is accessible."""
    if not directory:
        return False, "No directory provided"
    
    if not os.path.exists(directory):
        return False, f"Directory does not exist: {directory}"
    
    if not os.path.isdir(directory):
        return False, f"Path is not a directory: {directory}"
    
    if not os.access(directory, os.R_OK):
        return False, f"Directory is not readable: {directory}"
    
    return True, "Directory is valid"


def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds):
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class DataSummaryFormatter:
    """Helper class for formatting data summaries."""
    
    @staticmethod
    def format_basic_info(total_files, files_with_data):
        """Format basic file information."""
        files_without_data = total_files - files_with_data
        return [
            f"Total files: {total_files}",
            f"Files with data: {files_with_data}",
            f"Files without data: {files_without_data}",
        ]
    
    @staticmethod
    def format_file_details(filename, data_present, result=None):
        """Format details for a single file."""
        status = "✓ Data" if data_present else "✗ No data"
        details = [f"{filename}: {status}"]
        
        if data_present and result:
            try:
                # Import here to avoid circular imports
                setup_python_path()
                import read_intan
                
                sample_rate = read_intan.get_sample_rate(result)
                if sample_rate:
                    details.append(f"  Sample rate: {sample_rate} Hz")
                    
                # Count channels
                if 'amplifier_data' in result:
                    num_channels = result['amplifier_data'].shape[0]
                    num_samples = result['amplifier_data'].shape[1]
                    duration = num_samples / sample_rate if sample_rate else None
                    
                    details.append(f"  Channels: {num_channels}")
                    details.append(f"  Samples: {num_samples}")
                    
                    if duration:
                        details.append(f"  Duration: {format_duration(duration)}")
                    else:
                        details.append("  Duration: Unknown")
                        
            except Exception as e:
                details.append(f"  Error getting details: {str(e)}")
                
        return details
