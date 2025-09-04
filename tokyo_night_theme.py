"""
Tokyo Night theme configuration for EEG Analysis GUI.

This module contains the Tokyo Night color scheme and styling configurations
for the entire application.
"""

# Tokyo Night Color Palette
TOKYO_NIGHT_COLORS = {
    # Background colors
    'bg_primary': '#1a1b26',      # Main background
    'bg_secondary': '#24283b',    # Secondary panels
    'bg_tertiary': '#414868',     # Elevated elements
    'bg_float': '#16161e',        # Floating elements
    
    # Foreground colors
    'fg_primary': '#c0caf5',      # Primary text
    'fg_secondary': '#9aa5ce',    # Secondary text
    'fg_disabled': '#565f89',     # Disabled text
    
    # Accent colors
    'accent_blue': '#7aa2f7',     # Primary accent
    'accent_purple': '#bb9af7',   # Secondary accent
    'accent_cyan': '#7dcfff',     # Tertiary accent
    'accent_green': '#9ece6a',    # Success/positive
    'accent_yellow': '#e0af68',   # Warning
    'accent_red': '#f7768e',      # Error/negative
    'accent_orange': '#ff9e64',   # Special
    
    # UI Element colors
    'border': '#414868',          # Borders
    'selection': '#364a82',       # Selection highlight
    'hover': '#414868',           # Hover state
    'active': '#565f89',          # Active state
    
    # Terminal colors
    'terminal_bg': '#16161e',
    'terminal_fg': '#9ece6a',     # Green terminal text
    'terminal_cursor': '#c0caf5',
}

# Component-specific styles
TOKYO_NIGHT_STYLES = {
    'main_window': f"""
        QMainWindow {{
            background-color: {TOKYO_NIGHT_COLORS['bg_primary']};
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
        }}
        
        QWidget {{
            background-color: {TOKYO_NIGHT_COLORS['bg_primary']};
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
        }}
    """,
    
    'group_box': f"""
        QGroupBox {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            border-radius: 8px;
            font-weight: bold;
            font-size: 12px;
            color: {TOKYO_NIGHT_COLORS['accent_blue']};
            margin-top: 8px;
            padding-top: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: {TOKYO_NIGHT_COLORS['accent_blue']};
        }}
    """,
    
    'button_primary': f"""
        QPushButton {{
            background-color: {TOKYO_NIGHT_COLORS['accent_blue']};
            color: {TOKYO_NIGHT_COLORS['bg_primary']};
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
        }}
        
        QPushButton:hover {{
            background-color: {TOKYO_NIGHT_COLORS['accent_purple']};
        }}
        
        QPushButton:pressed {{
            background-color: {TOKYO_NIGHT_COLORS['accent_cyan']};
        }}
        
        QPushButton:disabled {{
            background-color: {TOKYO_NIGHT_COLORS['bg_tertiary']};
            color: {TOKYO_NIGHT_COLORS['fg_disabled']};
        }}
    """,
    
    'button_secondary': f"""
        QPushButton {{
            background-color: {TOKYO_NIGHT_COLORS['bg_tertiary']};
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 11px;
        }}
        
        QPushButton:hover {{
            background-color: {TOKYO_NIGHT_COLORS['hover']};
            border-color: {TOKYO_NIGHT_COLORS['accent_blue']};
        }}
        
        QPushButton:pressed {{
            background-color: {TOKYO_NIGHT_COLORS['active']};
        }}
        
        QPushButton:disabled {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            color: {TOKYO_NIGHT_COLORS['fg_disabled']};
            border-color: {TOKYO_NIGHT_COLORS['bg_tertiary']};
        }}
    """,
    
    'text_edit': f"""
        QTextEdit {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            border-radius: 6px;
            padding: 8px;
            font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            selection-background-color: {TOKYO_NIGHT_COLORS['selection']};
        }}
        
        QTextEdit:focus {{
            border-color: {TOKYO_NIGHT_COLORS['accent_blue']};
        }}
        
        QScrollBar:vertical {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {TOKYO_NIGHT_COLORS['bg_tertiary']};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {TOKYO_NIGHT_COLORS['hover']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
    """,
    
    'terminal': f"""
        QTextEdit {{
            background-color: {TOKYO_NIGHT_COLORS['terminal_bg']};
            color: {TOKYO_NIGHT_COLORS['terminal_fg']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            border-radius: 6px;
            padding: 12px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            selection-background-color: {TOKYO_NIGHT_COLORS['selection']};
        }}
    """,
    
    'line_edit': f"""
        QLineEdit {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 11px;
            selection-background-color: {TOKYO_NIGHT_COLORS['selection']};
        }}
        
        QLineEdit:focus {{
            border-color: {TOKYO_NIGHT_COLORS['accent_blue']};
        }}
        
        QLineEdit:disabled {{
            background-color: {TOKYO_NIGHT_COLORS['bg_tertiary']};
            color: {TOKYO_NIGHT_COLORS['fg_disabled']};
        }}
    """,
    
    'label': f"""
        QLabel {{
            color: {TOKYO_NIGHT_COLORS['fg_primary']};
            background: transparent;
        }}
    """,
    
    'progress_bar': f"""
        QProgressBar {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            border: 1px solid {TOKYO_NIGHT_COLORS['border']};
            border-radius: 4px;
            height: 6px;
        }}
        
        QProgressBar::chunk {{
            background-color: {TOKYO_NIGHT_COLORS['accent_blue']};
            border-radius: 3px;
        }}
    """,
    
    'status_bar': f"""
        QStatusBar {{
            background-color: {TOKYO_NIGHT_COLORS['bg_secondary']};
            color: {TOKYO_NIGHT_COLORS['fg_secondary']};
            border-top: 1px solid {TOKYO_NIGHT_COLORS['border']};
            font-size: 11px;
        }}
    """,
    
    'splitter': f"""
        QSplitter::handle {{
            background-color: {TOKYO_NIGHT_COLORS['border']};
        }}
        
        QSplitter::handle:horizontal {{
            width: 3px;
        }}
        
        QSplitter::handle:vertical {{
            height: 3px;
        }}
        
        QSplitter::handle:pressed {{
            background-color: {TOKYO_NIGHT_COLORS['accent_blue']};
        }}
    """,
    
    'form_layout': f"""
        QFormLayout QLabel {{
            color: {TOKYO_NIGHT_COLORS['fg_secondary']};
            font-weight: 500;
            font-size: 11px;
        }}
    """,
}

def get_matplotlib_style():
    """Get matplotlib style configuration for Tokyo Night theme."""
    return {
        'figure.facecolor': TOKYO_NIGHT_COLORS['bg_primary'],
        'axes.facecolor': TOKYO_NIGHT_COLORS['bg_secondary'],
        'axes.edgecolor': TOKYO_NIGHT_COLORS['border'],
        'axes.labelcolor': TOKYO_NIGHT_COLORS['fg_primary'],
        'axes.titlecolor': TOKYO_NIGHT_COLORS['accent_blue'],
        'xtick.color': TOKYO_NIGHT_COLORS['fg_secondary'],
        'ytick.color': TOKYO_NIGHT_COLORS['fg_secondary'],
        'text.color': TOKYO_NIGHT_COLORS['fg_primary'],
        'grid.color': TOKYO_NIGHT_COLORS['border'],
        'grid.alpha': 0.3,
        'savefig.facecolor': TOKYO_NIGHT_COLORS['bg_primary'],
        'savefig.edgecolor': 'none',
    }

def apply_tokyo_night_theme(app):
    """Apply Tokyo Night theme to the entire application."""
    # Combine all styles
    complete_style = ""
    for style_name, style_css in TOKYO_NIGHT_STYLES.items():
        complete_style += style_css + "\n"
    
    # Apply to application
    app.setStyleSheet(complete_style)
