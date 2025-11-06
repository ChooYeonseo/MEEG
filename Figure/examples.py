"""
Example Usage Scenarios for EEG Power Visualization
=====================================================

This file demonstrates various use cases and how to use the EEGPowerVisualizer.
Copy and paste examples into your scripts or notebooks.
"""

import sys
sys.path.insert(0, '/Users/sean/LINK/MEEG')

from Figure.Figure1E import EEGPowerVisualizer
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# SETUP: Initialize the visualizer once at the beginning
# ============================================================================

def initialize_visualizer():
    """Initialize the visualizer with your data."""
    csv_path = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv"
    electrode_map_path = "/Users/sean/LINK/MEEG/electrode_map/Version2.json"
    sampling_rate = 2000
    
    visualizer = EEGPowerVisualizer(csv_path, electrode_map_path, sampling_rate)
    return visualizer


# ============================================================================
# EXAMPLE 1: Basic Power Analysis
# ============================================================================

def example_1_basic_power_analysis():
    """
    Analyze power in 5-15 Hz band for first 5 seconds.
    Simplest use case.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Power Analysis")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Extract and analyze
    fig, ax, power_values = visualizer.visualize(
        start_time=0,
        duration=5,
        freq_band=(5, 15),
        title="Basic Power Analysis (5-15 Hz)"
    )
    
    # Show plot
    plt.show()
    
    # Print results
    print(f"\nAnalyzed {len(power_values)} channels")
    print(f"Power range: {min(power_values.values()):.2f} - {max(power_values.values()):.2f} µV²/Hz")


# ============================================================================
# EXAMPLE 2: Analyze Different Frequency Bands
# ============================================================================

def example_2_frequency_band_comparison():
    """
    Compare power across different frequency bands (Delta, Theta, Alpha, Beta).
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Frequency Band Comparison")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Define frequency bands
    frequency_bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-12 Hz)': (8, 12),
        'Beta (12-30 Hz)': (12, 30),
    }
    
    # Extract data once
    window_data = visualizer.extract_window(start_time=0, duration=10)
    
    # Compute power for each band
    results = {}
    for band_name, freq_range in frequency_bands.items():
        print(f"\nAnalyzing {band_name}...")
        power_values = visualizer.compute_psd_power(window_data, freq_band=freq_range)
        
        # Calculate statistics
        avg_power = sum(power_values.values()) / len(power_values)
        max_power = max(power_values.values())
        min_power = min(power_values.values())
        
        results[band_name] = {
            'avg': avg_power,
            'max': max_power,
            'min': min_power,
            'data': power_values
        }
        
        print(f"  Average: {avg_power:.2f}")
        print(f"  Max: {max_power:.2f}")
        print(f"  Min: {min_power:.2f}")
    
    # Summary
    print("\n" + "-"*60)
    print("SUMMARY:")
    for band_name, stats in results.items():
        print(f"{band_name}: avg={stats['avg']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")


# ============================================================================
# EXAMPLE 3: Time-Varying Analysis
# ============================================================================

def example_3_time_varying_analysis():
    """
    Track how power changes over time by analyzing multiple consecutive windows.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Time-Varying Analysis")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Parameters
    window_duration = 5  # seconds
    num_windows = 6
    freq_band = (8, 12)  # Alpha band
    
    # Analyze each window
    time_evolution = []
    for i in range(num_windows):
        start_time = i * window_duration
        print(f"\nWindow {i+1}: {start_time}-{start_time+window_duration} seconds")
        
        try:
            window_data = visualizer.extract_window(start_time, window_duration)
            power_values = visualizer.compute_psd_power(window_data, freq_band=freq_band)
            
            avg_power = sum(power_values.values()) / len(power_values)
            time_evolution.append({
                'window': i+1,
                'start_time': start_time,
                'avg_power': avg_power,
                'max_power': max(power_values.values()),
                'min_power': min(power_values.values())
            })
            print(f"  Average Power: {avg_power:.2f} µV²/Hz")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Create summary
    df = pd.DataFrame(time_evolution)
    print("\n" + "-"*60)
    print("TIME EVOLUTION:")
    print(df.to_string(index=False))


# ============================================================================
# EXAMPLE 4: Find Most and Least Active Regions
# ============================================================================

def example_4_spatial_analysis():
    """
    Identify which brain regions have the highest and lowest power.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Spatial Analysis (Brain Region Activity)")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Analyze
    window_data = visualizer.extract_window(start_time=0, duration=10)
    power_values = visualizer.compute_psd_power(window_data, freq_band=(5, 15))
    
    # Sort by power
    sorted_channels = sorted(power_values.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTOP 5 MOST ACTIVE CHANNELS (Highest Power):")
    for i, (channel, power) in enumerate(sorted_channels[:5], 1):
        print(f"  {i}. {channel}: {power:.2f} µV²/Hz")
    
    print("\nTOP 5 LEAST ACTIVE CHANNELS (Lowest Power):")
    for i, (channel, power) in enumerate(sorted_channels[-5:], 1):
        print(f"  {i}. {channel}: {power:.2f} µV²/Hz")
    
    # Regional analysis
    print("\n" + "-"*60)
    print("REGIONAL BREAKDOWN:")
    
    regions = {
        'Frontal': ['F3', 'F4', 'Fz', 'Fp1', 'Fp2'],
        'Central': ['C1', 'C2', 'Cz'],
        'Parietal': ['P3', 'P4', 'Pz'],
        'Temporal': ['T1', 'T2', 'T5', 'T6'],
        'Occipital': ['O1', 'O2'],
    }
    
    for region, channels in regions.items():
        region_power = [power_values[ch] for ch in channels if ch in power_values]
        if region_power:
            avg = sum(region_power) / len(region_power)
            print(f"  {region}: {avg:.2f} µV²/Hz (avg of {len(region_power)} channels)")


# ============================================================================
# EXAMPLE 5: Export Results to CSV
# ============================================================================

def example_5_export_results():
    """
    Export power analysis results to CSV file for further processing.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Export Results")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Analyze
    window_data = visualizer.extract_window(start_time=0, duration=5)
    power_values = visualizer.compute_psd_power(window_data, freq_band=(5, 15))
    
    # Export
    output_dir = "/Users/sean/LINK/MEEG/Figure/results"
    freq_band = (5, 15)
    
    visualizer.save_results(output_dir, power_values, freq_band)
    
    print(f"\nResults exported to: {output_dir}")
    print("CSV file format: Channel, Power_uV2_Hz")


# ============================================================================
# EXAMPLE 6: Custom Visualization Settings
# ============================================================================

def example_6_custom_visualization():
    """
    Customize the appearance of the power map visualization.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Visualization")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Extract data
    window_data = visualizer.extract_window(start_time=0, duration=5)
    power_values = visualizer.compute_psd_power(window_data, freq_band=(8, 12))
    
    # Create custom plot with different settings
    fig, ax = visualizer.plot_power_on_head(
        power_values,
        title="Alpha Band (8-12 Hz)\nCustom Styling",
        freq_band=(8, 12),
        cmap='hot',              # Different colormap
        head_scale=10,           # Larger head
        electrode_size=500       # Larger electrodes
    )
    
    plt.tight_layout()
    plt.show()
    
    print("Custom visualization displayed!")
    print("Try other colormaps: 'viridis', 'plasma', 'cool', 'jet', 'RdYlBu_r'")


# ============================================================================
# EXAMPLE 7: Batch Processing Multiple Time Windows
# ============================================================================

def example_7_batch_processing():
    """
    Analyze and save results for multiple time windows automatically.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Batch Processing")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    # Parameters
    time_windows = [0, 60, 120, 180]  # seconds
    freq_bands = [(5, 15), (8, 12), (4, 8)]
    
    results_summary = []
    
    for start_time in time_windows:
        for freq_band in freq_bands:
            print(f"\nProcessing: {start_time}s, {freq_band[0]}-{freq_band[1]} Hz")
            
            try:
                # Analyze
                window_data = visualizer.extract_window(start_time, duration=5)
                power_values = visualizer.compute_psd_power(window_data, freq_band=freq_band)
                
                # Calculate statistics
                avg_power = sum(power_values.values()) / len(power_values)
                
                results_summary.append({
                    'time': start_time,
                    'freq_band': f"{freq_band[0]}-{freq_band[1]}",
                    'avg_power': avg_power,
                    'max_power': max(power_values.values()),
                    'min_power': min(power_values.values())
                })
                
                print(f"  Average Power: {avg_power:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results_summary)
    print("\n" + "-"*60)
    print("BATCH PROCESSING SUMMARY:")
    print(df.to_string(index=False))


# ============================================================================
# EXAMPLE 8: Power Lateralization Analysis
# ============================================================================

def example_8_lateralization_analysis():
    """
    Compare power between left and right hemispheres.
    """
    print("\n" + "="*60)
    print("EXAMPLE 8: Lateralization Analysis")
    print("="*60)
    
    visualizer = initialize_visualizer()
    
    window_data = visualizer.extract_window(start_time=0, duration=10)
    power_values = visualizer.compute_psd_power(window_data, freq_band=(5, 15))
    
    # Split channels into hemispheres
    left_channels = ['F3', 'C1', 'P3', 'T5', 'O1']
    right_channels = ['F4', 'C2', 'P4', 'T6', 'O2']
    
    # Calculate average power for each hemisphere
    left_power = [power_values[ch] for ch in left_channels if ch in power_values]
    right_power = [power_values[ch] for ch in right_channels if ch in power_values]
    
    left_avg = sum(left_power) / len(left_power) if left_power else 0
    right_avg = sum(right_power) / len(right_power) if right_power else 0
    
    # Calculate asymmetry index
    asymmetry = (right_avg - left_avg) / ((right_avg + left_avg) / 2) * 100
    
    print(f"\nLeft Hemisphere Average Power: {left_avg:.2f} µV²/Hz")
    print(f"Right Hemisphere Average Power: {right_avg:.2f} µV²/Hz")
    print(f"Asymmetry Index: {asymmetry:.1f}%")
    
    if asymmetry > 5:
        print("  → Right hemisphere more active")
    elif asymmetry < -5:
        print("  → Left hemisphere more active")
    else:
        print("  → Relatively symmetric")


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EEG POWER VISUALIZATION - EXAMPLE SCRIPTS")
    print("="*60)
    print("\nUncomment the example you want to run at the bottom of this file")
    print("\nAvailable examples:")
    print("  1. Basic Power Analysis")
    print("  2. Frequency Band Comparison")
    print("  3. Time-Varying Analysis")
    print("  4. Spatial Analysis (Regional Activity)")
    print("  5. Export Results")
    print("  6. Custom Visualization")
    print("  7. Batch Processing")
    print("  8. Lateralization Analysis")
    
    # Uncomment to run examples:
    # example_1_basic_power_analysis()
    # example_2_frequency_band_comparison()
    # example_3_time_varying_analysis()
    # example_4_spatial_analysis()
    # example_5_export_results()
    # example_6_custom_visualization()
    # example_7_batch_processing()
    # example_8_lateralization_analysis()
    
    print("\n" + "="*60)
    print("To run an example, uncomment it in the main block and run this file")
    print("="*60)
