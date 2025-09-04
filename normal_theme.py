"""
Normal (light) theme configuration for EEG Analysis GUI.

This module contains the standard/default styling for the application.
"""

# Normal theme styles (minimal styling, relies on system defaults)
NORMAL_THEME_STYLES = {
    'main_window': """
        QMainWindow {
            background-color: #f0f0f0;
        }
    """,
    
    'group_box': """
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
        }
    """,
    
    'button_primary': """
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """,
    
    'button_secondary': """
        QPushButton {
            border: 1px solid #ccc;
            padding: 8px 16px;
            border-radius: 4px;
        }
        
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
    """,
    
    'text_edit': """
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
        }
        
        QTextEdit:focus {
            border-color: #0078d4;
        }
    """,
    
    'terminal': """
        QTextEdit {
            background-color: #000000;
            color: #00ff00;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
        }
    """,
    
    'line_edit': """
        QLineEdit {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px 8px;
        }
        
        QLineEdit:focus {
            border-color: #0078d4;
        }
    """,
    
    'progress_bar': """
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f0f0f0;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
    """,
    
    'status_bar': """
        QStatusBar {
            border-top: 1px solid #ccc;
        }
    """,
    
    'splitter': """
        QSplitter::handle {
            background-color: #ccc;
        }
        
        QSplitter::handle:horizontal {
            width: 3px;
        }
        
        QSplitter::handle:vertical {
            height: 3px;
        }
    """,
}

def apply_normal_theme(app):
    """Apply normal theme to the entire application."""
    # Combine all styles
    complete_style = ""
    for style_name, style_css in NORMAL_THEME_STYLES.items():
        complete_style += style_css + "\n"
    
    # Apply to application
    app.setStyleSheet(complete_style)
