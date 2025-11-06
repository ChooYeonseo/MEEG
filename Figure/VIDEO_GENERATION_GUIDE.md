# EEG Power Animation Video Generator

Generate animated videos showing how EEG power evolves over time across electrode positions!

## Features

✨ **Key Capabilities:**
- 🎬 Creates smooth animations showing power changes over configurable time periods
- 🎨 Uses the same interpolation and boundary masking as your static visualizations
- 📊 Consistent color scaling across all frames (red = high power, blue = low power)
- ⚡ Configurable frame rate, resolution, and window sizes
- 🔄 Overlapping windows for smooth transitions
- 💾 Outputs standard MP4 video files

## Quick Start

### Option 1: From Jupyter Notebook (Easiest)

1. Open `EEG_Power_Visualization.ipynb`
2. Go to **Visualization 4: Generate Animated Video of Power Changes Over Time**
3. Configure parameters:
   ```python
   video_start_time = 50              # When to start (seconds)
   video_total_duration = 30           # How long to animate (seconds)
   video_window_duration = 2           # Analysis window size (seconds)
   video_freq_band = (5, 15)           # Frequency band (Hz)
   video_fps = 10                      # Animation speed
   ```
4. Run the cell
5. Video will be saved to: `/Users/sean/LINK/MEEG/Figure/results/eeg_power_animation.mp4`

### Option 2: Command Line

```bash
# Basic usage
python /Users/sean/LINK/MEEG/generate_eeg_video.py \
  --start 50 \
  --duration 30 \
  --band 5-15 \
  --output my_video.mp4

# High quality video (larger file size, slower generation)
python /Users/sean/LINK/MEEG/generate_eeg_video.py \
  --start 0 \
  --duration 60 \
  --fps 20 \
  --dpi 150 \
  --output high_quality.mp4

# Full head visualization (no boundary masking)
python /Users/sean/LINK/MEEG/generate_eeg_video.py \
  --start 0 \
  --duration 30 \
  --no-boundary \
  --output full_head.mp4
```

## Parameters Explained

### Time Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_time` | 0 | When to start the animation (in seconds) |
| `total_duration` | 30 | How many seconds of data to animate |
| `window_duration` | 2 | Duration of analysis window for each frame |

**Example:** `start=50, duration=30, window=2` will animate from 50s-80s, computing power in overlapping 2-second windows.

### Video Quality

| Parameter | Default | Impact |
|-----------|---------|--------|
| `fps` | 10 | Frames per second (higher = smoother but larger files) |
| `dpi` | 100 | Resolution (higher = sharper but slower) |

**Recommendations:**
- Fast preview: `fps=5, dpi=75`
- Good quality: `fps=10, dpi=100` (balanced)
- High quality: `fps=20, dpi=150` (slower, larger files)

### Frequency Bands

Common choices:
- `(0.5, 4)` - Delta (deep sleep)
- `(4, 8)` - Theta (drowsiness)
- `(8, 13)` - Alpha (relaxed)
- `(13, 30)` - Beta (alert)
- `(30, 50)` - Gamma (high cognitive)

## How It Works

1. **Data Extraction**: Divides the time range into overlapping windows (50% overlap for smooth transitions)
2. **Power Computation**: Computes power in specified frequency band for each window using Welch's method
3. **Interpolation**: Uses cubic spline interpolation to create smooth surfaces between electrodes
4. **Boundary Masking**: Optionally restricts visualization to specified electrode boundary
5. **Frame Generation**: Creates animation frames with synchronized colorbars
6. **Video Encoding**: Encodes frames to MP4 using FFmpeg

## Example Animations

### 1. Monitor Sleep Stage Transitions
```python
# Watch power changes during sleep stage transition
visualizer.generate_video(
    start_time=120,           # 2 minutes into recording
    total_duration=60,         # 1 minute animation
    freq_band=(0.5, 4),       # Delta band
    window_duration=5,         # 5-second analysis windows
    fps=10
)
```

### 2. Alpha Power During Task
```python
# Visualize attention-related alpha band activity
visualizer.generate_video(
    start_time=0,
    total_duration=120,        # 2 minutes
    freq_band=(8, 13),        # Alpha band
    window_duration=2,         # Fine-grained 2-second windows
    fps=15                    # Smoother animation
)
```

### 3. Artifact Detection
```python
# Quickly scan for muscle artifacts in beta band
visualizer.generate_video(
    start_time=0,
    total_duration=300,        # 5 minutes scan
    freq_band=(13, 30),       # Beta band
    window_duration=3,         # Coarse 3-second windows
    fps=5                     # Slow playback to watch for spikes
)
```

## Troubleshooting

### "FFmpeg not found"
**Solution:** Install FFmpeg
```bash
# On macOS with Homebrew
brew install ffmpeg

# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On Windows with Chocolatey
choco install ffmpeg
```

### Video generation is slow
- Reduce `dpi` (e.g., 75 instead of 100)
- Reduce `fps` (e.g., 5 or 8 instead of 10)
- Reduce `total_duration` to test with shorter clips
- Use fewer frames: increase `window_duration` to 3-5 seconds

### Video file is huge
- Reduce `dpi` (defaults to 100)
- Reduce `fps` (5-10 is usually sufficient)
- The MP4 codec should compress it reasonably

### Poor interpolation quality
- Ensure `window_duration` is long enough for stable PSD (minimum 1-2 seconds recommended)
- Electrode positions and data might not be properly aligned
- Check that `boundary_electrodes` are correct electrode names

## Output Format

Generated MP4 files include:
- ✅ Smooth interpolated surface (pcolormesh)
- ✅ Contour lines for power levels
- ✅ Electrode markers with labels
- ✅ Boundary region outline (if specified)
- ✅ Synchronized colorbar
- ✅ Frame counter and time information
- ✅ Frequency band and time window labels

## Advanced Usage

### Programmatic Video Generation (Python)

```python
from Figure.Figure1E import EEGPowerVisualizer

# Initialize
visualizer = EEGPowerVisualizer(
    csv_path="/path/to/data.csv",
    electrode_map_path="/path/to/electrode_map.json"
)

# Generate with custom parameters
video_path = visualizer.generate_video(
    start_time=0,
    total_duration=60,
    window_duration=2,
    freq_band=(5, 15),
    output_path="/path/to/output.mp4",
    fps=10,
    boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3'],
    cmap='viridis',  # Try different colormaps!
    dpi=100
)

print(f"Video saved to: {video_path}")
```

### Different Colormaps

Try different colormaps for different visualizations:
```python
# Red-hot spectrum
cmap='hot'

# Viridis (perceptually uniform)
cmap='viridis'

# Plasma
cmap='plasma'

# Custom red-yellow-blue
cmap='RdYlBu_r'  # (current default)

# Grayscale
cmap='gray'
```

## Performance Tips

| What | Time | Notes |
|------|------|-------|
| 30s @ 10 FPS, 100 DPI | ~2 min | Default settings |
| 60s @ 10 FPS, 100 DPI | ~4 min | 2x duration |
| 30s @ 5 FPS, 100 DPI | ~1 min | Half the frames |
| 30s @ 10 FPS, 150 DPI | ~4-5 min | Higher quality |

## See Also

- `Figure1E.py` - Main visualization engine
- `EEG_Power_Visualization.ipynb` - Interactive notebook with examples
- `Figure1E.py.generate_video()` - Full API documentation

---

**Happy animating! 🎬📊**
