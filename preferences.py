"""
Preferences manager for EEG Analysis GUI application.

This module handles application preferences including theme selection
and other user settings.
"""

import json
import os
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal


class PreferencesManager(QObject):
    """Manager for application preferences and settings."""
    
    theme_changed = pyqtSignal(str)  # Signal emitted when theme changes
    
    def __init__(self):
        super().__init__()
        self.preferences_file = Path.home() / ".eeg_analysis_preferences.json"
        self.settings = self.load_preferences()
        
    def load_preferences(self):
        """Load preferences from file."""
        default_settings = {
            "theme": "normal",  # "normal" or "tokyo_night"
            "window_geometry": None,
            "last_directory": None,
            "plot_settings": {
                "default_duration": 10.0,
                "default_start_time": 0.0
            }
        }
        
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to handle new settings
                    default_settings.update(loaded_settings)
                    return default_settings
        except Exception as e:
            print(f"Error loading preferences: {e}")
            
        return default_settings
    
    def save_preferences(self):
        """Save preferences to file."""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def get_theme(self):
        """Get current theme setting."""
        return self.settings.get("theme", "normal")
    
    def set_theme(self, theme_name):
        """Set theme and emit signal."""
        if theme_name in ["normal", "tokyo_night"]:
            self.settings["theme"] = theme_name
            self.save_preferences()
            self.theme_changed.emit(theme_name)
    
    def get_setting(self, key, default=None):
        """Get a specific setting."""
        return self.settings.get(key, default)
    
    def set_setting(self, key, value):
        """Set a specific setting."""
        self.settings[key] = value
        self.save_preferences()


# Global preferences instance
preferences_manager = PreferencesManager()
