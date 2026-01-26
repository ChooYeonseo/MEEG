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
from theme import apply_tokyo_night_theme, apply_normal_theme, preferences_manager


def setup_application():
    """Set up the PyQt6 application with proper configuration."""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("MEEG")
    app.setApplicationDisplayName("MEEG")  # This ensures the menu shows "MEEG"
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Korea University College of Medicine")
    app.setOrganizationDomain("http://link.korea.ac.kr/")
    
    # Apply initial theme based on preferences
    current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
    if current_theme == 'tokyo_night':
        apply_tokyo_night_theme(app)
    elif current_theme == 'dark':
        from theme import apply_dark_theme
        apply_dark_theme(app)
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
    import traceback
    import os
    from pathlib import Path
    from datetime import datetime
    
    # Determine log file location (next to exe for frozen, in project dir for dev)
    if getattr(sys, 'frozen', False):
        log_dir = Path(sys.executable).parent
    else:
        log_dir = Path(__file__).parent
    
    log_file = log_dir / "meeg_crash.log"
    
    try:
        sys.exit(main())
    except Exception as e:
        # Write error to log file
        error_msg = f"""
{'='*60}
MEEG Crash Log - {datetime.now().isoformat()}
{'='*60}
Error Type: {type(e).__name__}
Error Message: {str(e)}

Full Traceback:
{traceback.format_exc()}
{'='*60}
"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg)
        
        print(f"ERROR: {e}")
        print(f"Full error written to: {log_file}")
        
        # Also show a message box if possible
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            if not QApplication.instance():
                app = QApplication(sys.argv)
            QMessageBox.critical(None, "MEEG Error", 
                f"An error occurred:\n\n{type(e).__name__}: {str(e)}\n\nSee {log_file} for details.")
        except:
            pass
        
        sys.exit(1)