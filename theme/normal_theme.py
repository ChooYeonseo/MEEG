"""
Bright theme configuration for EEG Analysis GUI.

This module contains the bright/light styling for the application with gray tones
and high contrast for maximum readability.
"""

# Bright theme color palette (gray tones with high contrast)
NORMAL_COLORS = {
    'bg_primary': '#f5f5f5',      # Light gray background
    'bg_secondary': '#ffffff',    # White panels
    'bg_tertiary': '#e8e8e8',     # Slightly darker gray
    'fg_primary': '#2e2e2e',      # Dark gray text
    'fg_secondary': '#5a5a5a',    # Medium gray text
    'fg_disabled': '#a0a0a0',     # Disabled gray text
    'accent_primary': '#6b6b6b',  # Medium-dark gray accent
    'accent_hover': '#858585',    # Lighter gray for hover
    'accent_pressed': '#4a4a4a',  # Darker gray for pressed
    'border': '#d0d0d0',          # Border gray
    'terminal_bg': '#2e2e2e',     # Dark terminal background
    'terminal_fg': '#d4d4d4',     # Light terminal text
}

# Normal theme styles (gray tones)
NORMAL_THEME_STYLES = {
    'main_window': f"""
        QMainWindow {{
            background-color: {NORMAL_COLORS['bg_primary']};
            color: {NORMAL_COLORS['fg_primary']};
        }}
        
        QMainWindow > QWidget {{
            background-color: {NORMAL_COLORS['bg_primary']};
            color: {NORMAL_COLORS['fg_primary']};
        }}
    """,
    
    'group_box': f"""
        QGroupBox {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            font-weight: bold;
            border: 2px solid {NORMAL_COLORS['border']};
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            color: {NORMAL_COLORS['fg_primary']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: {NORMAL_COLORS['accent_primary']};
        }}
    """,
    
    'button_primary': f"""
        QPushButton {{
            background-color: {NORMAL_COLORS['accent_primary']};
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {NORMAL_COLORS['accent_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {NORMAL_COLORS['accent_pressed']};
        }}
        
        QPushButton:disabled {{
            background-color: {NORMAL_COLORS['border']};
            color: {NORMAL_COLORS['fg_disabled']};
        }}
    """,
    
    'button_secondary': f"""
        QPushButton {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            border: 1px solid {NORMAL_COLORS['border']};
            color: {NORMAL_COLORS['fg_primary']};
            padding: 8px 16px;
            border-radius: 4px;
        }}
        
        QPushButton:hover {{
            background-color: {NORMAL_COLORS['bg_tertiary']};
        }}
        
        QPushButton:pressed {{
            background-color: {NORMAL_COLORS['border']};
        }}
    """,
    
    'text_edit': f"""
        QTextEdit {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border: 1px solid {NORMAL_COLORS['border']};
            border-radius: 4px;
            padding: 4px;
        }}
        
        QTextEdit:focus {{
            border-color: {NORMAL_COLORS['accent_primary']};
        }}
    """,
    
    'terminal': f"""
        QTextEdit {{
            background-color: {NORMAL_COLORS['terminal_bg']};
            color: {NORMAL_COLORS['terminal_fg']};
            border: 1px solid {NORMAL_COLORS['border']};
            border-radius: 4px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
    """,
    
    'line_edit': f"""
        QLineEdit {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border: 1px solid {NORMAL_COLORS['border']};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        
        QLineEdit:focus {{
            border-color: {NORMAL_COLORS['accent_primary']};
        }}
    """,
    
    'progress_bar': f"""
        QProgressBar {{
            border: 1px solid {NORMAL_COLORS['border']};
            border-radius: 4px;
            background-color: {NORMAL_COLORS['bg_tertiary']};
            color: {NORMAL_COLORS['fg_primary']};
        }}
        
        QProgressBar::chunk {{
            background-color: {NORMAL_COLORS['accent_primary']};
            border-radius: 3px;
        }}
    """,
    
    'status_bar': f"""
        QStatusBar {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border-top: 1px solid {NORMAL_COLORS['border']};
        }}
        
        QStatusBar QLabel {{
            color: {NORMAL_COLORS['fg_primary']};
        }}
    """,
    
    'label': f"""
        QLabel {{
            color: {NORMAL_COLORS['fg_primary']};
            background-color: transparent;
        }}
    """,
    
    'menu_bar': f"""
        QMenuBar {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border-bottom: 1px solid {NORMAL_COLORS['border']};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 6px 12px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {NORMAL_COLORS['bg_tertiary']};
        }}
        
        QMenuBar::item:pressed {{
            background-color: {NORMAL_COLORS['accent_primary']};
            color: white;
        }}
    """,
    
    'menu': f"""
        QMenu {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border: 1px solid {NORMAL_COLORS['border']};
        }}
        
        QMenu::item {{
            padding: 6px 24px;
        }}
        
        QMenu::item:selected {{
            background-color: {NORMAL_COLORS['accent_primary']};
            color: white;
        }}
        
        QMenu::separator {{
            height: 1px;
            background-color: {NORMAL_COLORS['border']};
            margin: 4px 0px;
        }}
    """,
    
    'splitter': f"""
        QSplitter::handle {{
            background-color: {NORMAL_COLORS['border']};
        }}
        
        QSplitter::handle:horizontal {{
            width: 3px;
        }}
        
        QSplitter::handle:vertical {{
            height: 3px;
        }}
    """,
    
    'combo_box': f"""
        QComboBox {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            border: 1px solid {NORMAL_COLORS['border']};
            border-radius: 4px;
            padding: 6px 10px;
        }}
        
        QComboBox:hover {{
            border-color: {NORMAL_COLORS['accent_primary']};
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {NORMAL_COLORS['bg_secondary']};
            color: {NORMAL_COLORS['fg_primary']};
            selection-background-color: {NORMAL_COLORS['accent_primary']};
            selection-color: white;
            border: 1px solid {NORMAL_COLORS['border']};
        }}
    """,
}

def apply_normal_theme(app):
    """Apply bright theme to the entire application."""
    # Combine all styles
    complete_style = ""
    for style_name, style_css in NORMAL_THEME_STYLES.items():
        complete_style += style_css + "\n"
    
    # Apply to application
    app.setStyleSheet(complete_style)
