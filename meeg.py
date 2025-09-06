"""
EEG Analysis GUI Application - Main Entry Point

Program Developed by Yeonseo (Sean) Choo - Affiliation: Korea University, College of Medicine

Usage:
    python main.py

Requirements:
    - PyQt6
    - numpy
    - pandas
    - matplotlib
    - tqdm
    - Custom utils/read_intan module
"""

import sys
from PyQt6.QtWidgets import QApplication

# Import the main window from the windows package
from windows import EEGMainWindow
from tokyo_night_theme import apply_tokyo_night_theme
from normal_theme import apply_normal_theme
from preferences import preferences_manager


def setup_application():
    """Set up the PyQt6 application with proper configuration."""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("MEEG")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Korea University College of Medicine")
    app.setOrganizationDomain("http://link.korea.ac.kr/")
    
    # Apply initial theme based on preferences
    current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
    if current_theme == 'tokyo_night':
        apply_tokyo_night_theme(app)
    else:
        apply_normal_theme(app)
    
    return app


def main():
    """Main function to run the EEG Analysis GUI."""
    print("Starting EEG Analysis GUI...")
    
    # Create and configure the application
    app = setup_application()
    
    # Create and show the main window
    main_window = EEGMainWindow(app)
    main_window.show()
    
    # Start the application event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())