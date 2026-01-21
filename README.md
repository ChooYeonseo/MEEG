# NeuroBridge MEEG

**Mouse EEG Behavior Labeling & Topography**

<p align="center">
  <img src="assets/img/Logo.png" alt="NeuroBridge MEEG Logo" width="400">
</p>

> Developed by **Yeonseo (Sean) Choo**  
> Affiliation: Korea University, College of Medicine  
> Lab: [LINK LAB](http://link.korea.ac.kr/)

---

## About

**MEEG** (Mouse EEG) is a comprehensive GUI application for analyzing, visualizing, and labeling mouse EEG data. It provides an integrated environment for epilepsy research, seizure detection, and brain activity analysis.

---

## Features

### 📁 Data Input & Management
- **RHD File Support**: Native Intan RHD file format reader
- **CSV Import**: Load generic EEG data from CSV files
- **Project Caching**: Fast reload of previously processed data
- **Preprocessing Pipeline**: Built-in preprocessing with customizable parameters

### 🧠 Signal Preprocessing
- Bandpass, lowpass, highpass filtering
- Notch filter (50/60 Hz) for power line noise
- DC offset removal
- Z-score normalization
- IQR-based artifact removal
- Resampling to target sampling rate

### 📊 Visualization
- **Data Preview**: Raw signal visualization with multi-channel display
- **STFT Spectrogram**: Time-frequency analysis with adjustable parameters
- **EEG Topography**: Power distribution across electrode positions
- **Mosaic Plots**: Differential signals between electrode pairs

### 🏷️ Seizure Labeling (5-Panel Interface)
1. **Topography Panel**: Frequency band power visualization (Delta, Theta, Alpha, Beta, Gamma)
2. **Spectrogram Panel**: STFT analysis with adjustable power range
3. **Mosaic Panel**: Multi-channel EEG display with epoch navigation
4. **Video Panel**: Synchronized video playback for behavior correlation
5. **Label Panel**: Keyboard-based labeling (0-8 severity scale)

### 🎨 User Interface
- **Multiple Themes**: Tokyo Night, Dark Mode, Bright Theme
- **Keyboard Shortcuts**: Arrow keys for navigation, 0-8 for labeling
- **Auto-save**: Labels saved automatically in JSON format

### 🔄 Auto-Update
- Checks GitHub releases on startup
- One-click update from within the application
- Works both for source and executable distribution

---

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/ChooYeonseo/MEEG.git
cd MEEG

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate meeg

# Run the application
python meeg.py
```

### From Executable (Windows)

1. Download the latest release from [GitHub Releases](https://github.com/ChooYeonseo/MEEG/releases)
2. Extract the zip file
3. Run `MEEG.exe`

---

## Requirements

See [environment.yml](environment.yml) for full dependency list:

| Package | Version |
|---------|---------|
| Python | 3.12+ |
| PyQt6 | 6.0+ |
| NumPy | 2.3+ |
| SciPy | 1.16+ |
| Pandas | 2.3+ |
| Matplotlib | 3.8+ |
| OpenCV | 4.0+ |
| scikit-learn | 1.5+ |

---

## Usage

### Quick Start

1. **Launch Application**: Run `python meeg.py` or double-click `MEEG.exe`
2. **Load Data**: Click "📁 Select Folder && Read Data" or load cached project
3. **Configure Electrodes**: Set up electrode positions for topography
4. **Start Analysis**: Click "🧠 MEEG Analysis" to open the labeling interface

### Keyboard Shortcuts (Labeling Window)

| Key | Action |
|-----|--------|
| `←` / `→` | Navigate epochs |
| `0-8` | Label current epoch (severity scale) |
| `Space` | Switch to video view |
| `↑` / `↓` | Adjust channel spacing |

---

## Project Structure

```
MEEG/
├── meeg.py                 # Main entry point
├── config.py               # Application configuration
├── environment.yml         # Conda environment
├── MEEG.spec              # PyInstaller build spec
│
├── windows/               # GUI windows
│   ├── main_window.py     # Main application window
│   └── M2/                # Analysis modules
│       ├── epilepsy_label_main.py   # Seizure labeling
│       ├── topowindow.py            # Topography widget
│       ├── mosaicwindow.py          # Mosaic plotter
│       ├── spectrogram_label_widgets.py
│       └── video_sync_widget.py
│
├── utils/                 # Utility modules
│   ├── read_intan.py      # RHD file reader
│   ├── signal_preprocessing.py  # Signal processing
│   ├── cache_manager.py   # Data caching
│   └── auto_updater.py    # GitHub update checker
│
├── theme/                 # UI themes
├── electrode_map/         # Electrode configurations
└── assets/               # Images and icons
```

---

## Building Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build
python -m PyInstaller MEEG.spec --noconfirm

# Output: dist/MEEG/MEEG.exe
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.0.2 | 2026-01 | Auto-update, improved preprocessing |
| 0.0.1 | 2025-11 | Initial release |

---

## License

See [LICENSE](LICENSE) for details.

---

## Contact

- **Developer**: Yeonseo (Sean) Choo
- **Lab**: LINK LAB, Korea University College of Medicine
- **Website**: http://link.korea.ac.kr/