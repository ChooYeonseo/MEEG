# Font Configuration System

## Overview

All three themes (Normal, Tokyo Night, Dark) now use unified font configuration dictionaries to ensure consistent typography across theme switches.

## Font Dictionary Structure

Each theme has its own `FONTS` dictionary with the following structure:

```python
THEME_FONTS = {
    'family': 'system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
    'mono_family': '"SF Mono", Monaco, "Cascadia Code", "Courier New", monospace',
    'size_small': '10px',
    'size_normal': '11px',
    'size_medium': '12px',
    'size_large': '14px',
}
```

## Font Variables

### Font Families

- **`family`**: UI font for regular interface elements (labels, buttons, menus, etc.)
  - Uses system-native fonts for best OS integration
  - Fallback chain: system-ui → -apple-system → Segoe UI → Helvetica Neue → Arial → sans-serif

- **`mono_family`**: Monospace font for code/terminal elements (text_edit, terminal)
  - Uses coding-optimized monospace fonts
  - Fallback chain: SF Mono → Monaco → Cascadia Code → Courier New → monospace

### Font Sizes

- **`size_small`** (10px): Progress bars, small indicators
- **`size_normal`** (11px): Standard UI elements (buttons, labels, line edits, status bar, menu bar)
- **`size_medium`** (12px): Group box titles, terminal text
- **`size_large`** (14px): Headers, emphasized text (reserved for future use)

## Theme-Specific Implementations

### Normal Theme
```python
NORMAL_FONTS = {
    'family': 'system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
    'mono_family': '"SF Mono", Monaco, "Cascadia Code", "Courier New", monospace',
    'size_small': '10px',
    'size_normal': '11px',
    'size_medium': '12px',
    'size_large': '14px',
}
```

### Tokyo Night Theme
```python
TOKYO_NIGHT_FONTS = {
    'family': 'system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
    'mono_family': '"SF Mono", Monaco, "Cascadia Code", "Courier New", monospace',
    'size_small': '10px',
    'size_normal': '11px',
    'size_medium': '12px',
    'size_large': '14px',
}
```

### Dark Theme
```python
DARK_FONTS = {
    'family': 'system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
    'mono_family': '"SF Mono", Monaco, "Cascadia Code", "Courier New", monospace',
    'size_small': '10px',
    'size_normal': '11px',
    'size_medium': '12px',
    'size_large': '14px',
}
```

## Usage in Stylesheets

All theme styles reference font variables instead of hardcoded values:

```python
'button_primary': f"""
    QPushButton {{
        font-family: {THEME_FONTS['family']};
        font-size: {THEME_FONTS['size_normal']};
        ...
    }}
""",

'terminal': f"""
    QTextEdit {{
        font-family: {THEME_FONTS['mono_family']};
        font-size: {THEME_FONTS['size_medium']};
        ...
    }}
""",
```

## Component Font Mapping

| Component | Font Family | Font Size | Usage |
|-----------|-------------|-----------|-------|
| group_box | family | size_medium (12px) | Group box titles |
| button_primary | family | size_normal (11px) | Primary buttons |
| button_secondary | family | size_normal (11px) | Secondary buttons |
| text_edit | family | size_normal (11px) | General text editing |
| terminal | mono_family | size_medium (12px) | Terminal/console output |
| line_edit | family | size_normal (11px) | Single-line text input |
| label | family | size_normal (11px) | Text labels |
| progress_bar | family | size_small (10px) | Progress bar text |
| status_bar | family | size_normal (11px) | Status bar messages |
| menu_bar | family | size_normal (11px) | Menu bar items |
| menu | family | size_normal (11px) | Dropdown menus |
| combo_box | family | size_normal (11px) | Combo boxes |
| form_layout | family | size_normal (11px) | Form labels |

## Benefits

1. **Consistency**: All themes use the same font sizes and families
2. **Maintainability**: Change font definitions in one place (FONTS dictionary)
3. **No Hardcoded Values**: All font references use variables
4. **Theme Independence**: Each theme can customize fonts if needed
5. **Easy Testing**: Switch themes and verify fonts remain consistent

## Testing

To verify font consistency:
1. Launch the application
2. Switch between themes (Normal → Tokyo Night → Dark)
3. Check that font sizes and families remain consistent across all UI elements
4. Verify terminal uses monospace fonts
5. Verify UI elements use system fonts

## Future Enhancements

- Can add theme-specific font customization if needed
- Can add `size_large` (14px) for headers when required
- Can add custom font weight variables (e.g., 'weight_normal', 'weight_bold')
- Can add line-height variables for better typography control
