# EEG Power Visualization on Electrode Map

## Overview

This project provides tools to visualize EEG power spectral density (PSD) across electrode locations on a head-shaped map. The power is color-coded with a red-yellow-blue colormap:
- **Red**: High power in the specified frequency band
- **Blue**: Low power in the specified frequency band

## Files

### Main Script: `Figure/Figure1E.py`
A standalone Python script that can be run from the command line. It includes the `EEGPowerVisualizer` class with all necessary functionality.

**Usage:**
```bash
cd /Users/sean/LINK/MEEG
python Figure/Figure1E.py
```

**Modify the configuration section** at the bottom of the file to change:
- `start_time`: Start time of the analysis window (seconds)
- `duration`: Length of the time window (seconds)
- `freq_band`: Frequency band to analyze (Hz), e.g., (5, 15) for 5-15 Hz

### Interactive Notebook: `EEG_Power_Visualization.ipynb`
A Jupyter notebook with multiple visualizations and analysis options. It's interactive and allows you to:
- Extract custom time windows
- Compare multiple frequency bands
- Analyze power changes over time
- Export results to CSV
- View summary statistics

**Key Features:**
1. **Basic Power Map**: Single frequency band visualization
2. **Multi-Band Comparison**: Compare power across Delta, Theta, Alpha, and Beta bands
3. **Time-Varying Analysis**: See how power changes across 6 consecutive time windows
4. **Data Export**: Save power values to CSV for further analysis
5. **Statistics**: Summary statistics for each frequency band

## Installation

The required packages are already in your `environment.yml`. Ensure you're using the correct conda environment:

```bash
conda activate eeg
```

## How It Works

### 1. Data Loading
- Loads CSV file from: `/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv`
- File contains 14 EEG channels sampled at 2000 Hz
- Total duration: ~10 minutes (600 seconds)

### 2. Time Window Extraction
- User specifies start time and duration
- Data is extracted from the CSV for that window

### 3. Power Spectral Density (PSD) Computation
- Uses Welch's method for robust PSD estimation
- Averages power within the specified frequency band (e.g., 5-15 Hz)
- Results in one power value per channel

### 4. Electrode Mapping
- Maps channel power values to physical electrode locations
- Uses coordinates from `electrode_map/Version2.json`
- 16 total electrodes based on standard 10-20 electrode system

### 5. Head-Shaped Visualization
- Draws a head outline (ellipse shape)
- Marks nose position and ear references
- Plots electrodes at their mapped coordinates
- Colors each electrode based on power value
- Includes colorbar for reference

## Frequency Bands

Standard EEG frequency bands:
- **Delta**: 0.5-4 Hz (deep sleep)
- **Theta**: 4-8 Hz (drowsiness, meditation)
- **Alpha**: 8-12 Hz (relaxed wakefulness)
- **Beta**: 12-30 Hz (active thinking)

## Electrode Coordinates

The 16 electrodes follow a modified 10-20 system with coordinates from `Version2.json`:

```
1   O1  O2
2   T5  T6
3   P3  P4 (Parietal)
4   C1  C2 (Central)
5   T1  T2
6   F3  F4 (Frontal)
7   Fp1 Fp2 (Frontal Pole)
8   Fz  Pz  Cz (Midline)
```

## Example Usage

### Command Line (Figure1E.py)
```python
# Edit the configuration section at the bottom:
start_time = 10  # Start at 10 seconds
duration = 5     # Analyze 5 seconds
freq_band = (8, 12)  # Alpha band

python Figure/Figure1E.py
```

### Jupyter Notebook (EEG_Power_Visualization.ipynb)
```python
# Cell 3: Configuration
csv_path = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv"
visualizer = EEGPowerVisualizer(csv_path, electrode_map_path, sampling_rate)

# Cell 5: Visualization 1 - Basic Power Map
start_time = 0
duration = 5
freq_band = (5, 15)
fig, ax, power_values = visualizer.visualize(...)

# Cell 6: Visualization 2 - Compare bands
# Automatically compares Delta, Theta, Alpha, Beta

# Cell 7: Visualization 3 - Time-Varying Analysis
# Shows power evolution over 6 consecutive windows
```

## Output

### Results Directory
Results are saved to: `/Users/sean/LINK/MEEG/Figure/results/`

### Output Files
- `eeg_power_5-15Hz.csv`: Power values for each channel
  ```
  Channel,Power_uV2_Hz
  P4,105.176440
  C2,87.700545
  ...
  ```

### Plots
- PNG files can be saved by modifying the code:
  ```python
  fig.savefig('/Users/sean/LINK/MEEG/Figure/results/power_map.png', dpi=300, bbox_inches='tight')
  ```

## Class Reference: `EEGPowerVisualizer`

### Initialization
```python
visualizer = EEGPowerVisualizer(
    csv_path,              # Path to CSV file
    electrode_map_path,    # Path to JSON electrode map
    sampling_rate=2000     # Sampling rate in Hz
)
```

### Key Methods

#### `extract_window(start_time, duration)`
Extract a time window from the data.
- **Parameters:**
  - `start_time`: Start time in seconds
  - `duration`: Duration in seconds
- **Returns:** DataFrame with extracted data

#### `compute_psd_power(data, freq_band=(5,15), method='welch')`
Compute power in a frequency band.
- **Parameters:**
  - `data`: EEG data DataFrame
  - `freq_band`: Tuple (low_hz, high_hz)
  - `method`: 'welch' or 'periodogram'
- **Returns:** Dictionary with channel names and power values

#### `plot_power_on_head(power_values, title, freq_band, cmap, head_scale, electrode_size)`
Plot power values on a head map.
- **Parameters:**
  - `power_values`: Dictionary of channel power values
  - `title`: Plot title
  - `freq_band`: Frequency band tuple
  - `cmap`: Colormap ('RdYlBu_r' recommended)
  - `head_scale`: Head size scale factor
  - `electrode_size`: Size of electrode markers
- **Returns:** Figure and axes objects

#### `visualize(start_time, duration, freq_band, title)`
Complete analysis pipeline.
- **Parameters:** Same as above methods
- **Returns:** Figure, axes, and power values dictionary

#### `save_results(output_dir, power_values, freq_band)`
Save results to CSV.
- **Parameters:**
  - `output_dir`: Directory for output
  - `power_values`: Dictionary of power values
  - `freq_band`: Frequency band tuple
- **Returns:** Path to saved file

## Customization

### Change Frequency Band
```python
# Modify in Figure1E.py or notebook:
freq_band = (1, 4)  # Delta band
freq_band = (4, 8)  # Theta band
freq_band = (8, 12)  # Alpha band
freq_band = (12, 30)  # Beta band
```

### Adjust Visualization
```python
fig, ax, power_values = visualizer.plot_power_on_head(
    power_values,
    title="Custom Title",
    cmap='jet',  # Change colormap
    head_scale=10,  # Larger head
    electrode_size=400  # Larger electrodes
)
```

### Change Electrode Map
If you have a different electrode layout, update:
- `/Users/sean/LINK/MEEG/electrode_map/Version2.json`
- Or point to a different JSON file in the initialization

## Troubleshooting

### File Not Found Error
- Ensure the volume "/Volumes/CHOO'S SSD" is mounted
- Check file paths are correct

### Channel Not Mapping to Electrodes
- Verify channel names in CSV match extraction logic
- Check `_extract_electrode_number()` method if channel naming is different

### Power Values Seem Wrong
- Verify frequency band is within data range (0.5-80 Hz after filtering)
- Check that data is actually preprocessed (filtered)
- Consider longer windows for more stable estimates (>= 5 seconds recommended)

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data handling
- **matplotlib**: Plotting
- **scipy**: Signal processing (Welch's method)
- **json**: Electrode map loading

All included in your `eeg` environment.

## References

- Welch, P. (1967). The use of fast Fourier transform for estimation of power spectra. IEEE Transactions on Audio and Electroacoustics.
- 10-20 electrode system: https://en.wikipedia.org/wiki/10%E2%80%9320_electrode_system

## Next Steps

1. Run the basic script: `python Figure/Figure1E.py`
2. Explore the Jupyter notebook for interactive analysis
3. Adjust parameters to analyze different time windows and frequency bands
4. Export results and integrate with your analysis pipeline
