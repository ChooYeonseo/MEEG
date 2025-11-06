#!/usr/bin/env python3
"""
EEG Power Video Generation Examples
====================================

This script demonstrates various ways to generate EEG power animation videos.
Run this interactively to see all the examples.
"""

import sys
sys.path.insert(0, '/Users/sean/LINK/MEEG/Figure')

from Figure1E import EEGPowerVisualizer
from pathlib import Path


# Configuration
CSV_PATH = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv"
ELECTRODE_MAP = '/Users/sean/LINK/MEEG/electrode_map/Version2_Fixed.json'
OUTPUT_DIR = Path('/Users/sean/LINK/MEEG/Figure/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def example_1_basic():
    """Example 1: Basic video generation with default parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Video Generation")
    print("="*70)
    print("\nGenerating a simple 30-second video of alpha band power...")
    print("Parameters:")
    print("  - Time range: 0-30 seconds")
    print("  - Frequency band: 5-15 Hz (Alpha)")
    print("  - Window duration: 2 seconds")
    print("  - Frame rate: 10 FPS")
    print("  - Resolution: 100 DPI\n")
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    output_path = str(OUTPUT_DIR / "example1_basic_alpha.mp4")
    visualizer.generate_video(
        start_time=0,
        total_duration=30,
        window_duration=2,
        freq_band=(5, 15),
        output_path=output_path,
        fps=10,
        boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
    )
    print(f"✅ Video saved: {output_path}\n")


def example_2_different_bands():
    """Example 2: Generate videos for different frequency bands"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Different Frequency Bands")
    print("="*70)
    
    bands = [
        ((0.5, 4), "Delta (Deep Sleep)", "delta"),
        ((4, 8), "Theta (Drowsiness)", "theta"),
        ((8, 13), "Alpha (Relaxed)", "alpha"),
        ((13, 30), "Beta (Alert)", "beta"),
    ]
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    for freq_band, description, name in bands:
        print(f"\nGenerating {description} band video...")
        
        output_path = str(OUTPUT_DIR / f"example2_band_{name}.mp4")
        visualizer.generate_video(
            start_time=50,
            total_duration=30,
            window_duration=2,
            freq_band=freq_band,
            output_path=output_path,
            fps=10,
            boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
        )
        print(f"✅ {description} video saved: {output_path}")


def example_3_smooth_animation():
    """Example 3: High frame rate for smooth animation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Smooth Animation (High Frame Rate)")
    print("="*70)
    print("\nGenerating smooth 30-second video at 20 FPS...")
    print("Note: This will take longer but result in smoother animation\n")
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    output_path = str(OUTPUT_DIR / "example3_smooth_animation.mp4")
    visualizer.generate_video(
        start_time=0,
        total_duration=30,
        window_duration=1.5,  # Shorter windows for smoother transitions
        freq_band=(5, 15),
        output_path=output_path,
        fps=20,  # Higher frame rate
        boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
    )
    print(f"✅ Smooth animation saved: {output_path}\n")


def example_4_high_quality():
    """Example 4: High quality video with better resolution"""
    print("\n" + "="*70)
    print("EXAMPLE 4: High Quality Video")
    print("="*70)
    print("\nGenerating high-quality video...")
    print("Note: Higher DPI = better quality but slower generation\n")
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    output_path = str(OUTPUT_DIR / "example4_high_quality.mp4")
    visualizer.generate_video(
        start_time=50,
        total_duration=30,
        window_duration=2,
        freq_band=(5, 15),
        output_path=output_path,
        fps=15,
        dpi=150,  # Higher resolution
        boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
    )
    print(f"✅ High quality video saved: {output_path}\n")


def example_5_no_boundary():
    """Example 5: Full head visualization without boundary masking"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Full Head Map (No Boundary Masking)")
    print("="*70)
    print("\nGenerating video showing entire electrode head map...\n")
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    output_path = str(OUTPUT_DIR / "example5_full_head.mp4")
    visualizer.generate_video(
        start_time=0,
        total_duration=30,
        window_duration=2,
        freq_band=(5, 15),
        output_path=output_path,
        fps=10,
        boundary_electrodes=None  # No boundary masking
    )
    print(f"✅ Full head video saved: {output_path}\n")


def example_6_long_recording():
    """Example 6: Long recording scan"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Long Recording Scan")
    print("="*70)
    print("\nGenerating 2-minute video for quick scanning...")
    print("Note: Coarser windows and lower FPS for speed\n")
    
    visualizer = EEGPowerVisualizer(CSV_PATH, ELECTRODE_MAP, sampling_rate=2000)
    
    output_path = str(OUTPUT_DIR / "example6_long_scan.mp4")
    visualizer.generate_video(
        start_time=0,
        total_duration=120,  # 2 minutes
        window_duration=4,  # Coarser windows
        freq_band=(5, 15),
        output_path=output_path,
        fps=5,  # Lower frame rate for speed
        dpi=75,  # Lower resolution
        boundary_electrodes=['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
    )
    print(f"✅ Long scan video saved: {output_path}\n")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("EEG POWER VIDEO GENERATION EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various video generation options.")
    print("All videos will be saved to:")
    print(f"  {OUTPUT_DIR}\n")
    
    # Ask which examples to run
    print("Select examples to run:")
    print("  1. Basic video generation")
    print("  2. Different frequency bands")
    print("  3. Smooth animation (high FPS)")
    print("  4. High quality video")
    print("  5. Full head map (no boundary)")
    print("  6. Long recording scan")
    print("  0. Run ALL examples\n")
    
    choice = input("Enter your choice (0-6, or comma-separated for multiple): ").strip()
    
    examples = {
        '1': example_1_basic,
        '2': example_2_different_bands,
        '3': example_3_smooth_animation,
        '4': example_4_high_quality,
        '5': example_5_no_boundary,
        '6': example_6_long_recording,
    }
    
    if choice == '0':
        # Run all
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    else:
        # Run selected
        for num in choice.split(','):
            num = num.strip()
            if num in examples:
                try:
                    examples[num]()
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
            else:
                print(f"Invalid choice: {num}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print(f"Videos saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
