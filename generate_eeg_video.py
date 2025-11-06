#!/usr/bin/env python3
"""
EEG Power Animation Video Generator

Generate animated videos showing how EEG power evolves over time with smooth
interpolation across electrode positions.

Usage:
    python generate_eeg_video.py --start 0 --duration 30 --band 5-15 --output video.mp4
    
Requirements:
    - matplotlib, numpy, pandas, scipy
    - FFmpeg installed on your system (for MP4 encoding)
"""

import argparse
import sys
from pathlib import Path

# Add the Figure directory to path
sys.path.insert(0, '/Users/sean/LINK/MEEG/Figure')

from Figure1E import EEGPowerVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Generate animated EEG power video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30-second video of 5-15 Hz band starting at 50 seconds
  python generate_eeg_video.py --start 50 --duration 30 --band 5-15 --output my_video.mp4
  
  # Generate video with custom window size and frame rate
  python generate_eeg_video.py --start 0 --duration 60 --window 3 --fps 15 --output high_quality.mp4
  
  # Generate video without boundary masking (full head)
  python generate_eeg_video.py --start 0 --duration 30 --no-boundary --output full_head.mp4
        """
    )
    
    # Time parameters
    parser.add_argument('--start', type=float, default=0,
                        help='Start time in seconds (default: 0)')
    parser.add_argument('--duration', type=float, default=30,
                        help='Total duration to visualize in seconds (default: 30)')
    parser.add_argument('--window', type=float, default=2,
                        help='Analysis window duration in seconds (default: 2)')
    
    # Frequency band
    parser.add_argument('--band', type=str, default='5-15',
                        help='Frequency band as "low-high" in Hz (default: "5-15")')
    
    # Video parameters
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Figure DPI/resolution (default: 100)')
    parser.add_argument('--output', type=str, default='eeg_power_animation.mp4',
                        help='Output video file path (default: eeg_power_animation.mp4)')
    
    # Boundary masking
    parser.add_argument('--no-boundary', action='store_true',
                        help='Disable boundary masking (show entire head map)')
    
    # Colormap
    parser.add_argument('--cmap', type=str, default='RdYlBu_r',
                        help='Colormap name (default: RdYlBu_r)')
    
    # Data paths
    parser.add_argument('--csv', type=str,
                        default="/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv",
                        help='Path to EEG CSV data file')
    parser.add_argument('--electrode-map', type=str,
                        default='/Users/sean/LINK/MEEG/electrode_map/Version2_Fixed.json',
                        help='Path to electrode map JSON file')
    
    args = parser.parse_args()
    
    # Parse frequency band
    try:
        freq_low, freq_high = map(float, args.band.split('-'))
        freq_band = (freq_low, freq_high)
    except:
        print(f"Error: Invalid frequency band format '{args.band}'. Use 'low-high' (e.g., '5-15')")
        sys.exit(1)
    
    # Setup boundary electrodes
    if args.no_boundary:
        boundary_electrodes = None
        print("Note: Boundary masking disabled (will show entire head map)")
    else:
        boundary_electrodes = ['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
    
    # Check if data file exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        print("\nTip: Update the --csv path to point to your EEG data file")
        sys.exit(1)
    
    if not Path(args.electrode_map).exists():
        print(f"Error: Electrode map file not found: {args.electrode_map}")
        sys.exit(1)
    
    # Initialize visualizer
    print("Initializing EEG Power Visualizer...")
    visualizer = EEGPowerVisualizer(
        csv_path=args.csv,
        electrode_map_path=args.electrode_map,
        sampling_rate=2000,
        channel_mapping_path='/Users/sean/LINK/MEEG/electrode_map/channel_mapping.json'
    )
    
    # Generate video
    try:
        video_path = visualizer.generate_video(
            start_time=args.start,
            total_duration=args.duration,
            window_duration=args.window,
            freq_band=freq_band,
            output_path=args.output,
            fps=args.fps,
            boundary_electrodes=boundary_electrodes,
            cmap=args.cmap,
            dpi=args.dpi
        )
        
        print(f"\n✅ Success! Video saved to: {video_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Video generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
