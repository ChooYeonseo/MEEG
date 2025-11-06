"""
Theme package for EEG Analysis GUI.

Contains theme configurations and preferences management.
"""

from .preferences import preferences_manager
from .normal_theme import apply_normal_theme, NORMAL_THEME_STYLES, NORMAL_COLORS
from .dark_theme import apply_dark_theme, DARK_THEME_STYLES, DARK_COLORS
from .tokyo_night_theme import apply_tokyo_night_theme, TOKYO_NIGHT_STYLES, TOKYO_NIGHT_COLORS

__all__ = [
    'preferences_manager',
    'apply_normal_theme',
    'apply_dark_theme',
    'apply_tokyo_night_theme',
    'NORMAL_THEME_STYLES',
    'NORMAL_COLORS',
    'DARK_THEME_STYLES',
    'DARK_COLORS',
    'TOKYO_NIGHT_STYLES',
    'TOKYO_NIGHT_COLORS',
]
