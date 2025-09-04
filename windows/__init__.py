"""
Windows package for EEG Analysis GUI application.

This package contains all the GUI window components for the EEG analysis tool.
"""

from .main_window import EEGMainWindow
from .data_reader_thread import DataReaderThread, CacheLoaderThread
from .output_capture import OutputCapture
from .plotting_window import PlottingWindow
from .preprocessing_pipeline import PreprocessingPipelineDialog
from .cached_project_dialog import CachedProjectDialog

__all__ = ['EEGMainWindow', 'DataReaderThread', 'CacheLoaderThread', 'OutputCapture', 'PlottingWindow', 'CachedProjectDialog']
