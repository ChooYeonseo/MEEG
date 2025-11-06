#!/usr/bin/env python3
"""
Quick test script to verify the electrode visualization works
"""

import sys
sys.path.insert(0, '/Users/sean/LINK/MEEG')

from Figure.Figure1E import EEGPowerVisualizer
import matplotlib.pyplot as plt

# Initialize
csv_path = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv"
electrode_map_path = "/Users/sean/LINK/MEEG/electrode_map/Version2_Fixed.json"

visualizer = EEGPowerVisualizer(csv_path, electrode_map_path, sampling_rate=2000)

# Test visualization
print("\n" + "="*60)
print("Testing EEG Power Visualization")
print("="*60)

fig, ax, power_values = visualizer.visualize(
    start_time=0,
    duration=5,
    freq_band=(5, 15),
    title="EEG Power Map Test\n5-15 Hz Band"
)

print("\nVisualization created successfully!")
print(f"Electrodes mapped: {len(power_values)}")
print(f"\nChannel powers:")
for ch, power in sorted(power_values.items()):
    print(f"  {ch}: {power:.2f} µV²/Hz")

plt.savefig('/Users/sean/LINK/MEEG/Figure/results/eeg_power_test.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: /Users/sean/LINK/MEEG/Figure/results/eeg_power_test.png")

plt.show()
