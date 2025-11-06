# EEG Power Visualization - Updated Features

## Recent Changes

### 1. **Fixed Electrode Coordinates**
- Created new electrode map (`Version2_Fixed.json`) with proper 10-20 system coordinates
- Electrodes now fit correctly on the head shape
- Coordinates are optimized for standard head mapping

### 2. **Channel Name Display**
- Electrodes now display with their **EEG channel names** (P4, C2, F4, etc.) instead of electrode numbers
- Much easier to identify specific brain regions
- Labels are clear and positioned directly on the electrodes

### 3. **Improved Visualization**
- Power values are now **interpolated across the entire head** using scipy's griddata
- Empty spaces between electrodes are filled with interpolated power values
- Creates a smooth, continuous power map across the scalp
- Color gradients flow naturally from high power (red) to low power (blue)

### 4. **Better Head Boundary**
- Elliptical head outline properly defined
- Interpolation is masked to stay within the head boundary
- Contour lines show power distribution patterns

## Electrode Layout

The 16 electrodes follow the standard 10-20 system:

```
        Fp1  Fp2
    F3  Fz   F4
    C1  (center)  C2
    P3  Pz   P4
    T5  T1   T2  T6
    O1  (center)  O2
```

## File Structure

- `electrode_map/Version2_Fixed.json` - Updated electrode coordinates with channel names
- `Figure/Figure1E.py` - Main visualization script
- `EEG_Power_Visualization.ipynb` - Interactive Jupyter notebook
- `test_visualization.py` - Quick test script

## Usage

### Command Line
```bash
cd /Users/sean/LINK/MEEG
python Figure/Figure1E.py
```

### Jupyter Notebook
1. Open `EEG_Power_Visualization.ipynb`
2. Run cells in order
3. Adjust parameters in each visualization cell:
   - `start_time`: Start time in seconds
   - `duration`: Window length in seconds
   - `freq_band`: Frequency range (low_hz, high_hz)

### Test Script
```bash
python test_visualization.py
```

## Visualization Features

### Interpolation
- **Linear interpolation** fills empty space between electrodes
- Smooth power gradient across the scalp
- Realistic representation of brain activity

### Color Coding
- **Red**: High power (active regions)
- **Yellow**: Medium power
- **Blue**: Low power (quiet regions)

### Information Display
- Electrode labels show channel names (P4, C2, etc.)
- Colorbar shows power scale
- Statistics box shows power range and mean

## Example Parameters

**Alpha Band (8-12 Hz)**
```python
start_time = 0
duration = 10
freq_band = (8, 12)
```

**Theta Band (4-8 Hz)**
```python
start_time = 0
duration = 10
freq_band = (4, 8)
```

**Beta Band (12-30 Hz)**
```python
start_time = 0
duration = 10
freq_band = (12, 30)
```

## Output

- **PNG files**: Saved to `Figure/results/`
- **CSV files**: Power values exported to `eeg_power_*.csv`
- **Statistics**: Printed to console (mean, min, max power)

## Next Steps

1. Run `test_visualization.py` to verify everything works
2. Open the Jupyter notebook for interactive exploration
3. Adjust time windows and frequency bands to analyze different aspects
4. Export results for further analysis
