# UI Restructure Summary

## Overview
Restructured the EEG seizure labeling interface with an initial configuration dialog and three separate control panels for better usability and organization.

## Major Changes

### 1. Initial Configuration Dialog
**File:** `windows/M2/epilepsy_label_main.py`

- **New Class:** `InitialConfigDialog`
- **Purpose:** Appears before main window to configure essential parameters
- **Features:**
  - Epoch time interval configuration (0.1 - 60 seconds, default 1 second)
  - Sampling rate configuration:
    - Auto-detects sampling rate from EEG data
    - Displays detected rate with checkbox to confirm
    - Manual override option if detection fails or user wants custom rate
  - "Go" button to accept and launch main window
  - "Cancel" button to abort

### 2. Removed Components
- ❌ Epoch Control Panel (epoch length, epochs to show, current epoch spinboxes)
- ❌ Navigation buttons (Previous/Next)
- ❌ Current epoch spinbox

**Rationale:** Epoch configuration moved to initial dialog; navigation now handled entirely by label widget

### 3. New Control Panel Structure

#### Panel 1: EEG Display Control
- **Purpose:** Control mosaic plot display
- **Features:**
  - Dropdown selector for epochs to show: 1, 3, 5, 7, 9, 11, 13, 15
  - User can only select from predefined values (no manual entry)
  - Independent from label panel's epoch count

#### Panel 2: Spectrogram Control Panel
- **Purpose:** Control spectrogram visualization
- **Features:**
  - **Brightness Controls:**
    - "Brighter" button: Reduces color dynamic range (makes bright areas dimmer → overall brighter look)
    - "Dimmer" button: Increases color dynamic range (makes bright areas brighter → overall dimmer look)
    - Resets zoom acceleration when clicked
  
  - **Zoom Controls:**
    - "+" button (Show Less): Decrease epochs shown → more detail
    - "-" button (Show More): Increase epochs shown → wider view
    - **Acceleration Feature:** Consecutive clicks increase step size (1→2→3...→10)
    - Acceleration resets when limit reached or when switching to brightness controls
  
  - **Status Display:**
    - Label showing current epoch count: "Showing N epochs"
    - Updates dynamically with zoom changes

#### Panel 3: Label Control Panel
- **Purpose:** Control label widget display and manage label files
- **Features:**
  - Dropdown selector for epochs to show: 1, 3, 5, 7, 9, 11, 13, 15
  - User can only select from predefined values (no manual entry)
  - Independent from EEG panel's epoch count
  - Label file management buttons:
    - Create Label File
    - Import Label File
    - Save Labels

### 4. Data Summary Display
Updated to show configuration from initial dialog:
- Electrodes count
- Sampling Rate (from config dialog)
- Duration (calculated)
- Epoch Length (from config dialog)
- Total Epochs (calculated)

## Technical Implementation

### Modified Files

#### 1. `windows/M2/epilepsy_label_main.py`

**New Classes:**
- `InitialConfigDialog`: Configuration dialog with epoch length and sampling rate inputs

**Modified Class: `EpilepsyLabelWindow`**

**New Attributes:**
```python
self.detected_sampling_rate  # Auto-detected from EEG data
self.epoch_length           # From config dialog
self.sampling_rate          # From config dialog (detected or manual)
self.mosaic_epochs_to_show  # Independent control for mosaic (default: 5)
self.label_epochs_to_show   # Independent control for label (default: 5)
self.spectrogram_epochs_to_show  # Independent control for spectrogram (default: 15)
self.spectrogram_vmin       # Brightness control min
self.spectrogram_vmax       # Brightness control max
self.zoom_acceleration      # Tracks consecutive zoom operations
```

**New Methods:**
- `on_mosaic_epochs_changed(value)`: Handle mosaic epoch count change
- `on_label_epochs_changed(value)`: Handle label epoch count change
- `on_brighter_clicked()`: Increase spectrogram brightness
- `on_dimmer_clicked()`: Decrease spectrogram brightness
- `on_zoom_in_clicked()`: Show fewer epochs with acceleration
- `on_zoom_out_clicked()`: Show more epochs with acceleration

**Removed Methods:**
- `on_epoch_length_changed()`
- `on_epochs_to_show_changed()`
- `on_current_epoch_changed()`
- `previous_epoch()`
- `next_epoch()`

**Modified Methods:**
- `__init__()`: Shows config dialog before initialization
- `create_control_panel()`: Restructured to 3 panels
- `on_label_changed()`: Removed current_epoch_spin reference

#### 2. `windows/M2/spectrogram_label_widgets.py`

**Modified Class: `SpectrogramWidget`**

**New Attributes:**
```python
self.vmin                   # Color scale minimum (None = auto)
self.vmax                   # Color scale maximum (None = auto)
self.last_computed_vmin     # Last auto-computed min
self.last_computed_vmax     # Last auto-computed max
```

**New Methods:**
- `get_color_limits()`: Returns current (vmin, vmax) tuple
- `set_color_limits(vmin, vmax)`: Sets color scale limits for brightness control

**Modified Methods:**
- `update_plot()`: Applies custom color limits when set, stores auto-computed values

## User Workflow

### Application Startup
1. User opens seizure labeling window
2. **Configuration Dialog Appears:**
   - Shows detected sampling rate (if available)
   - User confirms or enters custom sampling rate
   - User sets epoch time interval
   - User clicks "Go" to continue (or "Cancel" to abort)
3. Main window opens with configured parameters

### During Analysis

#### Adjusting Display Density
- **Mosaic Plot:** Select from Panel 1 dropdown (1-15 epochs)
- **Label Widget:** Select from Panel 3 dropdown (1-15 epochs)
- **Spectrogram:** Use Panel 2 zoom buttons
  - Click "+" multiple times for rapid zoom in
  - Click "-" multiple times for rapid zoom out

#### Adjusting Spectrogram Appearance
- **Too Dark:** Click "Brighter" button (Panel 2)
- **Too Bright:** Click "Dimmer" button (Panel 2)
- Multiple clicks for larger adjustments

#### Navigation
- Use arrow keys in label widget
- Click epochs in label widget
- Click epochs in mosaic plot
- All widgets stay synchronized automatically

## Benefits

### 1. Simplified Initial Setup
- Clear, focused dialog for essential configuration
- Prevents accidental epoch length changes during analysis
- Sampling rate validation before main window loads

### 2. Reduced Clutter
- Removed redundant navigation controls
- Removed epoch configuration from main window
- Cleaner, more organized interface

### 3. Independent View Controls
- Mosaic and label widgets can show different epoch counts
- Users can optimize each view independently
- Spectrogram has dynamic zoom with acceleration

### 4. Enhanced Spectrogram Control
- Brightness adjustment without editing code
- Smooth zoom with acceleration for large datasets
- Visual feedback of current zoom level

### 5. Better User Experience
- One-time configuration at startup
- Contextual controls grouped by function
- Intuitive zoom acceleration (faster when clicking repeatedly)

## Configuration Examples

### Example 1: High Temporal Resolution
**Initial Config:**
- Epoch Length: 0.5 seconds
- Sampling Rate: 2000 Hz (detected)

**Panel Settings:**
- EEG Display: 15 epochs (7.5 seconds)
- Label Widget: 7 epochs (3.5 seconds)
- Spectrogram: 20 epochs (10 seconds)

### Example 2: Overview Analysis
**Initial Config:**
- Epoch Length: 2.0 seconds
- Sampling Rate: 2000 Hz (detected)

**Panel Settings:**
- EEG Display: 3 epochs (6 seconds)
- Label Widget: 11 epochs (22 seconds)
- Spectrogram: 50 epochs (100 seconds, zoomed out)

## Migration Notes

### For Users
- **First Launch:** Will see new configuration dialog
- **Epoch Length:** Now set once at startup (was adjustable during analysis)
- **Navigation:** Use label widget or mosaic clicks (Previous/Next buttons removed)
- **Epochs to Show:** Now separate controls for mosaic, label, and spectrogram

### For Developers
- Configuration dialog can be extended for other parameters
- Brightness/zoom controls demonstrate pattern for spectrogram enhancements
- Acceleration mechanism can be applied to other continuous adjustments
- Independent epoch counts allow widget-specific optimizations
