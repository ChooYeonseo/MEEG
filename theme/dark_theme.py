"""
Dark mode theme configuration for EEG Analysis GUI.

This module contains a dark black color scheme with white text for high contrast.
"""

# Dark mode color palette (dark blacks with white text)
DARK_COLORS = {
    'bg_primary': '#1a1a1a',      # Very dark black background
    'bg_secondary': '#252525',    # Dark black panels
    'bg_tertiary': '#303030',     # Slightly lighter black
    'bg_elevated': '#3a3a3a',     # Elevated elements
    'fg_primary': '#ffffff',      # Pure white text for maximum contrast
    'fg_secondary': '#e8e8e8',    # Very light gray text
    'fg_disabled': '#808080',     # Medium gray for disabled text
    'accent_primary': '#606060',  # Medium-dark gray accent
    'accent_hover': '#707070',    # Lighter gray for hover
    'accent_pressed': '#4a4a4a',  # Darker gray for pressed
    'accent_highlight': '#909090', # Highlight gray
    'border': '#404040',          # Dark border
    'terminal_bg': '#0a0a0a',     # Nearly black terminal background
    'terminal_fg': '#ffffff',     # White terminal text
}

# Dark mode styles
DARK_THEME_STYLES = {
    'main_window': f"""
        QMainWindow {{
            background-color: {DARK_COLORS['bg_primary']};
            color: {DARK_COLORS['fg_primary']};
        }}
        
        QMainWindow > QWidget {{
            background-color: {DARK_COLORS['bg_primary']};
            color: {DARK_COLORS['fg_primary']};
        }}
    """,
    
    'group_box': f"""
        QGroupBox {{
            background-color: {DARK_COLORS['bg_secondary']};
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 8px;
            font-weight: bold;
            font-size: 12px;
            color: {DARK_COLORS['accent_highlight']};
            margin-top: 8px;
            padding-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: {DARK_COLORS['accent_highlight']};
        }}
    """,
    
    'button_primary': f"""
        QPushButton {{
            background-color: {DARK_COLORS['accent_primary']};
            color: {DARK_COLORS['fg_primary']};
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
        }}
        
        QPushButton:hover {{
            background-color: {DARK_COLORS['accent_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {DARK_COLORS['accent_pressed']};
        }}
        
        QPushButton:disabled {{
            background-color: {DARK_COLORS['bg_tertiary']};
            color: {DARK_COLORS['fg_disabled']};
        }}
    """,
    
    'button_secondary': f"""
        QPushButton {{
            background-color: {DARK_COLORS['bg_tertiary']};
            color: {DARK_COLORS['fg_primary']};
            border: 1px solid {DARK_COLORS['border']};
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 11px;
        }}
        
        QPushButton:hover {{
            background-color: {DARK_COLORS['bg_elevated']};
            border-color: {DARK_COLORS['accent_primary']};
        }}
        
        QPushButton:pressed {{
            background-color: {DARK_COLORS['bg_tertiary']};
        }}
    """,
    
    'text_edit': f"""
        QTextEdit {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 6px;
            padding: 8px;
            selection-background-color: {DARK_COLORS['accent_primary']};
        }}
        
        QTextEdit:focus {{
            border-color: {DARK_COLORS['accent_highlight']};
        }}
    """,
    
    'terminal': f"""
        QTextEdit {{
            background-color: {DARK_COLORS['terminal_bg']};
            color: {DARK_COLORS['terminal_fg']};
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 6px;
            padding: 10px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            selection-background-color: {DARK_COLORS['accent_primary']};
        }}
    """,
    
    'line_edit': f"""
        QLineEdit {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 4px;
            padding: 6px 10px;
            selection-background-color: {DARK_COLORS['accent_primary']};
        }}
        
        QLineEdit:focus {{
            border-color: {DARK_COLORS['accent_highlight']};
        }}
    """,
    
    'combo_box': f"""
        QComboBox {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 4px;
            padding: 6px 10px;
        }}
        
        QComboBox:hover {{
            border-color: {DARK_COLORS['accent_primary']};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            selection-background-color: {DARK_COLORS['accent_primary']};
            border: 1px solid {DARK_COLORS['border']};
        }}
    """,
    
    'progress_bar': f"""
        QProgressBar {{
            border: 1px solid {DARK_COLORS['border']};
            border-radius: 6px;
            text-align: center;
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            font-weight: bold;
        }}
        
        QProgressBar::chunk {{
            background-color: {DARK_COLORS['accent_primary']};
            border-radius: 5px;
        }}
    """,
    
    'label': f"""
        QLabel {{
            color: {DARK_COLORS['fg_primary']};
            background-color: transparent;
        }}
    """,
    
    'status_bar': f"""
        QStatusBar {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border-top: 1px solid {DARK_COLORS['border']};
        }}
        
        QStatusBar QLabel {{
            color: {DARK_COLORS['fg_primary']};
        }}
    """,
    
    'menu_bar': f"""
        QMenuBar {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border-bottom: 1px solid {DARK_COLORS['border']};
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 6px 12px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {DARK_COLORS['bg_elevated']};
        }}
        
        QMenuBar::item:pressed {{
            background-color: {DARK_COLORS['accent_primary']};
        }}
    """,
    
    'menu': f"""
        QMenu {{
            background-color: {DARK_COLORS['bg_secondary']};
            color: {DARK_COLORS['fg_primary']};
            border: 1px solid {DARK_COLORS['border']};
        }}
        
        QMenu::item {{
            padding: 6px 24px;
        }}
        
        QMenu::item:selected {{
            background-color: {DARK_COLORS['accent_primary']};
        }}
        
        QMenu::separator {{
            height: 1px;
            background-color: {DARK_COLORS['border']};
            margin: 4px 0px;
        }}
    """,
    
    'splitter': f"""
        QSplitter::handle {{
            background-color: {DARK_COLORS['border']};
        }}
        
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        
        QSplitter::handle:hover {{
            background-color: {DARK_COLORS['accent_primary']};
        }}
    """,
    
    'scroll_bar': f"""
        QScrollBar:vertical {{
            background-color: {DARK_COLORS['bg_secondary']};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {DARK_COLORS['accent_primary']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {DARK_COLORS['accent_hover']};
        }}
        
        QScrollBar:horizontal {{
            background-color: {DARK_COLORS['bg_secondary']};
            height: 12px;
            border: none;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {DARK_COLORS['accent_primary']};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {DARK_COLORS['accent_hover']};
        }}
        
        QScrollBar::add-line, QScrollBar::sub-line {{
            background: none;
            border: none;
        }}
    """,
}

def apply_dark_theme(app):
    """Apply dark theme to the entire application."""
    # Combine all styles
    complete_style = ""
    for style_name, style_css in DARK_THEME_STYLES.items():
        complete_style += style_css + "\n"
    
    # Apply to application
    app.setStyleSheet(complete_style)
