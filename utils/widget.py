"""
EEG plotting widgets and utility functions.

This module contains reusable plotting functions for EEG data visualization,
separated from the main GUI components for better modularity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def calculate_global_statistics(cached_segments, channels, df_data=None):
    """
    Calculate global statistics from all cached data.
    
    Parameters:
    -----------
    cached_segments : dict
        Dictionary of cached data segments
    channels : list
        List of channel names
    df_data : pd.DataFrame, optional
        Current data frame (fallback if no cached data)
        
    Returns:
    --------
    global_sigma : float
        Global standard deviation across all data
    """
    all_data_values = []
    
    # Collect data from all cached segments
    for cached_data in cached_segments.values():
        for ch in channels:
            if ch in cached_data:
                all_data_values.extend(cached_data[ch])
    
    # If no cached data, use current data for statistics
    if not all_data_values and df_data is not None:
        for ch in channels:
            if ch in df_data.columns:
                all_data_values.extend(df_data[ch].values)
    
    # Calculate global sigma
    global_sigma = np.std(all_data_values) if all_data_values else 100.0
    
    return global_sigma


def add_scale_bars(ax, time_data, global_sigma, y_center, total_height):
    """
    Add scale bars for amplitude and time to the EEG plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The plot axes
    time_data : pd.Series
        Time data
    global_sigma : float
        Global standard deviation
    y_center : float
        Y-axis center position
    total_height : float
        Total height of the plot
    """
    # Scale bar size: sigma rounded up to nearest 100 units
    scale_bar_size = np.ceil(global_sigma / 100) * 100
    
    # Y-scale bar (amplitude)
    scale_x = time_data.iloc[0] + 0.02 * (time_data.iloc[-1] - time_data.iloc[0])
    scale_y_bottom = y_center + total_height/2 - scale_bar_size - 20
    scale_y_top = scale_y_bottom + scale_bar_size
    
    ax.plot([scale_x, scale_x], [scale_y_bottom, scale_y_top], 'k-', linewidth=2)
    ax.text(scale_x - 0.01 * (time_data.iloc[-1] - time_data.iloc[0]), 
           (scale_y_bottom + scale_y_top) / 2, 
           f'{int(scale_bar_size)}μV', 
           ha='right', va='center', fontsize=8, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # X-scale bar (time) - Fixed to show correct physical length
    time_duration = time_data.iloc[-1] - time_data.iloc[0]
    
    # Choose appropriate scale bar duration based on total duration
    if time_duration >= 20:
        time_scale = 5.0  # 5s for long duration
    elif time_duration >= 10:
        time_scale = 2.0  # 2s for medium duration  
    elif time_duration >= 5:
        time_scale = 1.0  # 1s for short duration
    else:
        time_scale = 0.5  # 0.5s for very short duration
    
    # Calculate the physical positions for the scale bar
    # The bar should be positioned near the bottom right and have correct length
    bar_end_x = time_data.iloc[-1] - 0.05 * time_duration  # 5% from right edge
    bar_start_x = bar_end_x - time_scale  # Subtract actual time duration
    time_bar_y = y_center - total_height/2 + 20
    
    # Only draw the bar if it fits within the visible time range
    if bar_start_x >= time_data.iloc[0]:
        ax.plot([bar_start_x, bar_end_x], [time_bar_y, time_bar_y], 'k-', linewidth=2)
        ax.text((bar_start_x + bar_end_x) / 2, 
               time_bar_y - 15, 
               f'{time_scale}s', 
               ha='center', va='top', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def add_midline(ax, time_data):
    """
    Add a vertical dashed line at the middle time point.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The plot axes
    time_data : pd.Series
        Time data
    """
    mid_time = (time_data.iloc[0] + time_data.iloc[-1]) / 2
    ax.axvline(x=mid_time, color='red', linestyle='--', alpha=1.0, linewidth=1)


def clean_axes(ax):
    """
    Remove axes spines and ticks for a clean EEG plot appearance.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The plot axes
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


##############################################################
############### Various EEG Plotting Functions ###############
##############################################################

def plot_standard_eeg_data(figure, df_data, cached_segments, 
                          sigma_multiplier=5, y_range_multiplier=5,
                          line_color='black', line_width=0.8, line_alpha=0.8,
                          global_sigma=None):
    """
    Plot standard EEG data with consistent scaling and spacing.
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The matplotlib figure to plot on
    df_data : pd.DataFrame
        DataFrame containing the EEG data with 'time' column
    cached_segments : dict
        Dictionary of cached data segments for global statistics
    sigma_multiplier : float, default=1.5
        Multiplier for channel spacing (channels spaced by sigma_multiplier * sigma)
    y_range_multiplier : float, default=3
        Multiplier for Y-axis range (range = y_range_multiplier * sigma)
    line_color : str, default='blue'
        Color of the EEG traces
    line_width : float, default=0.8
        Width of the EEG traces
    line_alpha : float, default=0.8
        Transparency of the EEG traces
        
    Returns:
    --------
    global_sigma : float
        The calculated global standard deviation
    """
    # Clear figure
    figure.clear()
    
    # Get available channels (exclude 'time')
    channels = [ch for ch in df_data.columns if ch != 'time']
    if not channels:
        return None
    
    time_data = df_data['time']
    
    # Use provided global sigma or calculate it
    if global_sigma is not None:
        global_sigma_value = global_sigma
    else:
        global_sigma_value = calculate_global_statistics(cached_segments, channels, df_data)
    
    # Channel spacing and Y-axis range
    channel_spacing = sigma_multiplier * global_sigma_value
    y_range = y_range_multiplier * global_sigma_value
    
    # Create single plot with all channels
    ax = figure.add_subplot(1, 1, 1)
    
    # Plot each channel with vertical offset
    for i, channel_name in enumerate(channels):
        channel_values = df_data[channel_name]
        
        # Apply vertical offset (center channels around 0)
        offset = (i - (len(channels) - 1) / 2) * channel_spacing
        offset_signal = channel_values + offset
        
        # Plot signal
        ax.plot(time_data, offset_signal, color=line_color, 
               linewidth=line_width, alpha=line_alpha)
        
        # Add channel label on the right side
        ax.text(time_data.iloc[-1], offset, f'  {channel_name}', 
               va='center', ha='left', fontsize=8, fontweight='bold')
    
    # Set Y-axis limits and remove Y-axis ticks
    total_height = (len(channels) - 1) * channel_spacing + 2 * y_range
    y_center = 0
    ax.set_ylim(y_center - total_height/2, y_center + total_height/2)
    
    # Set X-axis limits to fit the data exactly
    ax.set_xlim(time_data.iloc[0], time_data.iloc[-1])
    
    # Add visual elements
    add_midline(ax, time_data)
    add_scale_bars(ax, time_data, global_sigma_value, y_center, total_height)
    clean_axes(ax)
    
    # Set title
    start_time = time_data.iloc[0] if len(time_data) > 0 else 0
    end_time = time_data.iloc[-1] if len(time_data) > 0 else 0
    figure.suptitle(f'EEG Data: {start_time:.2f}s - {end_time:.2f}s (σ={global_sigma_value:.1f}μV)', 
                   fontsize=12, fontweight='bold')
    
    return global_sigma_value


def plot_enhanced_eeg_data(figure, df_data, cached_segments, global_sigma=None):
    """
    Plot enhanced EEG view with improved styling and statistics.
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The matplotlib figure to plot on
    df_data : pd.DataFrame
        DataFrame containing the EEG data
    cached_segments : dict
        Dictionary of cached data segments
        
    Returns:
    --------
    global_sigma : float
        The calculated global standard deviation
    """
    # Clear figure
    figure.clear()
    
    channels = [ch for ch in df_data.columns if ch != 'time']
    if not channels:
        return None
    
    time_data = df_data['time']
    num_channels = len(channels)
    
    # Use provided global sigma or calculate it
    if global_sigma is not None:
        global_sigma_value = global_sigma
    else:
        global_sigma_value = calculate_global_statistics(cached_segments, channels, df_data)
    
    # Color palette for different channels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, channel_name in enumerate(channels):
        ax = figure.add_subplot(num_channels, 1, i + 1)
        
        channel_values = df_data[channel_name]
        color = colors[i % len(colors)]
        
        # Enhanced plotting with better styling
        ax.plot(time_data, channel_values, color=color, linewidth=1.2, alpha=0.8)
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Enhanced styling
        ax.set_ylabel(f'{channel_name}\n(μV)', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        # Add statistics
        y_mean = channel_values.mean()
        y_std = channel_values.std()
        stats_text = f'μ={y_mean:.1f} σ={y_std:.1f}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Optimize y-axis range
        padding = max(y_std * 0.1, 1)
        ax.set_ylim(y_mean - 3*y_std - padding, y_mean + 3*y_std + padding)
        
        if i == num_channels - 1:
            ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    # Enhanced title
    start_time = time_data.iloc[0] if len(time_data) > 0 else 0
    end_time = time_data.iloc[-1] if len(time_data) > 0 else 0
    figure.suptitle(f'Enhanced EEG Data: {start_time:.2f}s - {end_time:.2f}s (σ={global_sigma_value:.1f}μV)', 
                   fontsize=12, fontweight='bold')
    
    return global_sigma_value


def plot_differential_eeg_data(figure, df_data, cached_segments, global_sigma=None):
    """
    Plot differential signals if multiple channels are available.
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The matplotlib figure to plot on
    df_data : pd.DataFrame
        DataFrame containing the EEG data
    cached_segments : dict
        Dictionary of cached data segments
        
    Returns:
    --------
    global_sigma : float
        The calculated global standard deviation
    """
    # Clear figure
    figure.clear()
    
    channels = [ch for ch in df_data.columns if ch != 'time']
    if len(channels) < 2:
        # Fall back to standard plot if not enough channels
        return plot_standard_eeg_data(figure, df_data, cached_segments)
    
    time_data = df_data['time']
    
    # Use provided global sigma or calculate it
    if global_sigma is not None:
        global_sigma_value = global_sigma
    else:
        global_sigma_value = calculate_global_statistics(cached_segments, channels, df_data)
    
    # Create differential pairs (adjacent channels)
    diff_pairs = []
    for i in range(len(channels) - 1):
        diff_pairs.append((channels[i], channels[i + 1]))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (ch1, ch2) in enumerate(diff_pairs):
        ax = figure.add_subplot(len(diff_pairs), 1, i + 1)
        
        # Calculate differential signal
        diff_signal = df_data[ch1] - df_data[ch2]
        color = colors[i % len(colors)]
        
        ax.plot(time_data, diff_signal, color=color, linewidth=1.2, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax.set_ylabel(f'{ch1} - {ch2}\n(μV)', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
        
        if i == len(diff_pairs) - 1:
            ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    start_time = time_data.iloc[0] if len(time_data) > 0 else 0
    end_time = time_data.iloc[-1] if len(time_data) > 0 else 0
    figure.suptitle(f'Differential EEG Signals: {start_time:.2f}s - {end_time:.2f}s (σ={global_sigma_value:.1f}μV)', 
                   fontsize=12, fontweight='bold')
    
    return global_sigma_value


def plot_clinical_scale_eeg_data(figure, df_data, cached_segments, global_sigma=None):
    """
    Plot with clinical scale (7μV = 1mm).
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The matplotlib figure to plot on
    df_data : pd.DataFrame
        DataFrame containing the EEG data
    cached_segments : dict
        Dictionary of cached data segments
        
    Returns:
    --------
    global_sigma : float
        The calculated global standard deviation
    """
    # Clear figure
    figure.clear()
    
    channels = [ch for ch in df_data.columns if ch != 'time']
    if not channels:
        return None
    
    time_data = df_data['time']
    
    # Use provided global sigma or calculate it
    if global_sigma is not None:
        global_sigma_value = global_sigma
    else:
        global_sigma_value = calculate_global_statistics(cached_segments, channels, df_data)
    
    # Clinical scaling: 7μV = 1mm
    uv_per_mm = 7.0
    channel_spacing_mm = 10.0  # 1cm between channels
    channel_spacing_uv = channel_spacing_mm * uv_per_mm
    
    # Single plot with all channels offset vertically
    ax = figure.add_subplot(1, 1, 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    y_min, y_max = float('inf'), float('-inf')
    
    for i, channel_name in enumerate(channels):
        channel_values = df_data[channel_name]
        color = colors[i % len(colors)]
        
        # Apply vertical offset
        offset = i * channel_spacing_uv
        offset_signal = channel_values + offset
        
        ax.plot(time_data, offset_signal, color=color, linewidth=1.0, 
               label=channel_name, alpha=0.8)
        
        # Add baseline for each channel
        ax.axhline(y=offset, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        y_min = min(y_min, offset_signal.min())
        y_max = max(y_max, offset_signal.max())
    
    # Set up clinical scale y-axis
    tick_spacing_uv = uv_per_mm  # 7μV per tick
    y_tick_start = np.floor(y_min / tick_spacing_uv) * tick_spacing_uv
    y_tick_end = np.ceil(y_max / tick_spacing_uv) * tick_spacing_uv
    
    y_ticks = np.arange(y_tick_start, y_tick_end + tick_spacing_uv, tick_spacing_uv)
    ax.set_yticks(y_ticks)
    
    # Custom y-tick labels
    y_tick_labels = [f'{int(tick)}μV\n({tick/uv_per_mm:.1f}mm)' for tick in y_ticks]
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    
    ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Clinical Scale (7μV = 1mm)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    start_time = time_data.iloc[0] if len(time_data) > 0 else 0
    end_time = time_data.iloc[-1] if len(time_data) > 0 else 0
    figure.suptitle(f'Clinical Scale EEG: {start_time:.2f}s - {end_time:.2f}s (7μV = 1mm, σ={global_sigma_value:.1f}μV)', 
                   fontsize=12, fontweight='bold')
    
    return global_sigma_value
    
    return global_sigma
