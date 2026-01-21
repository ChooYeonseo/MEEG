"""
Configuration file for EEG Analysis GUI application.

This module contains all the important configuration settings and constants
used throughout the application.
"""

# Application metadata
APP_NAME = "EEG Analysis Tool"
APP_VERSION = "1.1.2"
GITHUB_REPO = "ChooYeonseo/MEEG"
ORGANIZATION_NAME = "EEG Research"
ORGANIZATION_DOMAIN = "eeg-research.org"

# Window settings
DEFAULT_WINDOW_WIDTH = 1000
DEFAULT_WINDOW_HEIGHT = 700
DEFAULT_WINDOW_TITLE = "EEG Data Analysis Tool"

# UI Settings
TERMINAL_FONT_FAMILY = "Courier"
TERMINAL_FONT_SIZE = 10
TITLE_FONT_SIZE = 18

# File extensions and patterns
RHD_FILE_PATTERN = "*.rhd"
SUPPORTED_EXTENSIONS = [".rhd"]

# UI Colors (hex codes)
COLORS = {
    "primary_button": "#0078d4",
    "primary_button_hover": "#106ebe", 
    "primary_button_pressed": "#005a9e",
    "disabled_button_bg": "#cccccc",
    "disabled_button_text": "#666666",
    "terminal_bg": "#000000",
    "terminal_text": "#00ff00",
    "selected_dir": "#0000ff",
    "unselected_dir": "#808080"
}

# Progress and status messages
STATUS_MESSAGES = {
    "ready": "Ready",
    "selecting_directory": "Selecting directory...",
    "reading_files": "Reading RHD files...",
    "no_directory": "No directory selected",
    "no_data": "No data found",
    "error": "Error occurred"
}

# Layout settings
LAYOUT_SPACING = 10
LAYOUT_MARGINS = 15
BUTTON_MIN_HEIGHT = 40
INFO_TEXT_MIN_HEIGHT = 200
TERMINAL_MIN_HEIGHT = 200

# Splitter default sizes
DEFAULT_SPLITTER_SIZES = [300, 300]
