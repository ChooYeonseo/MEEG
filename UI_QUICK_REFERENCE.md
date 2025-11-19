# Quick Reference: New UI Controls

## Initial Setup Dialog

When you open the seizure labeling window, you'll first see a configuration dialog:

### Epoch Time Interval
- **Range:** 0.1 - 60 seconds
- **Default:** 1.0 second
- **Purpose:** Sets the duration of each analysis epoch
- **Note:** This cannot be changed after window opens

### Sampling Rate
- **Auto-Detection:** Shows detected rate if available
- **Checkbox:** "Use detected sampling rate" (checked by default if detected)
- **Manual Override:** Uncheck to enter custom sampling rate
- **Range:** 1 - 100,000 Hz

### Buttons
- **Go:** Accept configuration and open main window
- **Cancel:** Abort and close window

---

## Control Panels (Left Side)

### Panel 1: EEG Display Control
Controls the mosaic plot (top-right panel)

**Epochs to Show:** Dropdown menu
- Options: 1, 3, 5, 7, 9, 11, 13, 15
- Default: 5
- Effect: Number of epochs visible in mosaic plot

### Panel 2: Spectrogram Control Panel
Controls the spectrogram (middle-right panel)

**Brightness:**
- **Brighter:** Makes spectrogram brighter (reduces color contrast)
- **Dimmer:** Makes spectrogram dimmer (increases color contrast)

**Zoom:**
- **+ (Show Less):** Zoom in - shows fewer epochs with more detail
- **- (Show More):** Zoom out - shows more epochs for wider view
- **Acceleration:** Clicking repeatedly increases step size (faster zoom)
- **Status:** Shows current epoch count ("Showing N epochs")

**Tips:**
- Click brightness buttons to reset zoom acceleration
- Zoom acceleration resets when limit is reached
- Max acceleration: 10x

### Panel 3: Label Control Panel
Controls the label widget (bottom-right panel)

**Epochs to Show:** Dropdown menu
- Options: 1, 3, 5, 7, 9, 11, 13, 15
- Default: 5
- Effect: Number of epochs visible in label widget
- **Independent from Panel 1**

**Label File Management:**
- **Create Label File:** Creates new CSV for seizure labels
- **Import Label File:** Loads existing label CSV
- **Save Labels:** Saves current labels to CSV

---

## Navigation

### Keyboard (in Label Widget)
- **Left Arrow:** Previous epoch
- **Right Arrow:** Next epoch
- **0-8 Keys:** Assign Racine score to current epoch

### Mouse
- **Click epoch in mosaic plot:** Jump to that epoch
- **Click epoch in label widget:** Jump to that epoch

### Synchronization
All widgets automatically stay synchronized:
- Mosaic plot
- Spectrogram
- Topography (left-top)
- Label widget

---

## Common Tasks

### Adjust Mosaic View Density
1. Go to **Panel 1: EEG Display Control**
2. Select epochs to show from dropdown (1-15)

### Adjust Label View Density
1. Go to **Panel 3: Label Control Panel**
2. Select epochs to show from dropdown (1-15)

### Make Spectrogram Easier to Read
**Too dark:**
1. Click "Brighter" button in Panel 2

**Too bright:**
1. Click "Dimmer" button in Panel 2

### Zoom Spectrogram for Detail
**Zoom in (more detail):**
1. Click "+" button in Panel 2
2. Keep clicking for faster zoom

**Zoom out (wider view):**
1. Click "-" button in Panel 2
2. Keep clicking for faster zoom

### Label Seizures
1. Navigate to epoch using arrow keys or clicks
2. Press 0-8 to assign Racine score
3. Save with "Save Labels" button in Panel 3

---

## Racine Score Reference

- **0:** No seizure activity
- **1:** Facial movements
- **2:** Head nodding
- **3:** Forelimb clonus
- **4:** Rearing
- **5:** Rearing and falling
- **6:** Severe tonic-clonic seizure
- **7:** Wild running/jumping
- **8:** Death

---

## Tips

### For Large Datasets
1. Start with high spectrogram zoom (show many epochs)
2. Identify regions of interest
3. Zoom in on suspicious regions
4. Use mosaic/label panels for detailed analysis

### For Efficient Labeling
1. Set label widget to show 7-11 epochs
2. Keep mosaic at 3-5 epochs for context
3. Use keyboard shortcuts for labeling
4. Save frequently

### For Finding Seizures
1. Use spectrogram overview (zoom out)
2. Look for frequency/power changes
3. Click suspicious epochs in spectrogram
4. Confirm with mosaic and topography
5. Label and continue

---

## Troubleshooting

### Config Dialog Doesn't Show Detected Rate
- The EEG data may not contain sampling rate metadata
- Use manual override with your known sampling rate
- Common values: 1000 Hz, 2000 Hz, 5000 Hz

### Spectrogram Too Bright/Dark After Zoom
- Click "Brighter" or "Dimmer" to adjust
- Zoom and brightness are independent controls

### Can't Change Epoch Length
- Epoch length is set in initial config dialog only
- Restart window to change epoch length

### Widgets Not Synchronized
- This should not happen - it's automatic
- If it does, report as a bug with steps to reproduce
