"""
EEG Power Visualization on Electrode Map
=========================================

This script loads EEG data from a CSV file, computes power spectral density (PSD)
in a specified frequency band (e.g., 5-15 Hz), and visualizes the power values
across electrode locations in a head-shaped plot.

The power is color-coded: red for high power, blue for low power.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import json
from scipy.signal import welch
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from pathlib import Path


class EEGPowerVisualizer:
    """Visualize EEG power across electrode map with head shape."""
    
    def __init__(self, csv_path, electrode_map_path, sampling_rate=2000, channel_mapping_path=None):
        """
        Initialize the EEG Power Visualizer.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing EEG data
        electrode_map_path : str
            Path to the JSON file containing electrode coordinates
        sampling_rate : float
            Sampling rate of the EEG data in Hz
        channel_mapping_path : str, optional
            Path to the channel-to-electrode mapping JSON file
        """
        self.csv_path = csv_path
        self.electrode_map_path = electrode_map_path
        self.sampling_rate = sampling_rate
        
        # Load data
        self.data = self._load_csv()
        self.electrode_coords = self._load_electrode_map()
        
        # Load channel mapping if provided
        if channel_mapping_path is None:
            # Try default location
            channel_mapping_path = "/Users/sean/LINK/MEEG/electrode_map/channel_mapping.json"
        
        try:
            self.channel_mapping = self._load_channel_mapping(channel_mapping_path)
        except:
            self.channel_mapping = {}
            print("Warning: Could not load channel mapping, will use fallback method")
        
        print(f"Data loaded: {self.data.shape[0]} samples at {sampling_rate} Hz")
        print(f"Duration: {self.data.shape[0] / sampling_rate:.2f} seconds")
        print(f"Channels: {len(self.data.columns)}")
        print(f"Electrodes in map: {len(self.electrode_coords)}")
        print(f"Channel mapping available: {len(self.channel_mapping) > 0}")
    
    def _load_csv(self):
        """Load CSV file into a pandas DataFrame."""
        df = pd.read_csv(self.csv_path)
        return df
    
    def _load_electrode_map(self):
        """Load electrode coordinates from JSON file."""
        with open(self.electrode_map_path, 'r') as f:
            electrode_map = json.load(f)
        
        # Convert to dictionary with electrode number as key
        coords_dict = {}
        for electrode in electrode_map:
            coords_dict[electrode['number']] = {
                'x': electrode['x'],
                'y': electrode['y'],
                'channel': electrode.get('channel', '')  # Get channel name if available
            }
        return coords_dict
    
    def _load_channel_mapping(self, channel_mapping_path):
        """Load channel to electrode number mapping from JSON file."""
        with open(channel_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        return mapping_data.get('channel_to_electrode', {})
    
    def extract_window(self, start_time, duration):
        """
        Extract a time window from the data.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        duration : float
            Duration of the window in seconds
            
        Returns:
        --------
        window_data : pd.DataFrame
            Data in the specified time window
        """
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int((start_time + duration) * self.sampling_rate)
        
        # Clip to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self.data), end_sample)
        
        window_data = self.data.iloc[start_sample:end_sample].reset_index(drop=True)
        
        actual_duration = (end_sample - start_sample) / self.sampling_rate
        print(f"Extracted window: {start_sample} to {end_sample} samples ({actual_duration:.2f}s)")
        
        return window_data
    
    def compute_psd_power(self, data, freq_band=(5, 15), method='welch'):
        """
        Compute power spectral density for a frequency band.
        
        Parameters:
        -----------
        data : pd.DataFrame
            EEG data (channels as columns)
        freq_band : tuple
            Frequency band (low_hz, high_hz)
        method : str
            Method for PSD computation ('welch' or 'periodogram')
            
        Returns:
        --------
        power_values : dict
            Dictionary with all electrode channels as keys and power values as values.
            Missing channels are interpolated from available data.
        """
        low_hz, high_hz = freq_band
        power_values_raw = {}
        
        # First, compute power for available channels
        for channel in data.columns:
            if channel == 'time':  # Skip time column if present
                continue
                
            signal = data[channel].values
            
            if method == 'welch':
                # Use Welch's method for more stable PSD estimate
                frequencies, psd = welch(signal, fs=self.sampling_rate, nperseg=min(1024, len(signal)//4))
            else:
                # Use periodogram
                from scipy.signal import periodogram
                frequencies, psd = periodogram(signal, fs=self.sampling_rate)
            
            # Find power in the specified frequency band
            band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            band_power = np.mean(psd[band_mask])
            
            power_values_raw[channel] = band_power
            print(f"{channel}: {band_power:.6f} µV²/Hz (avg in {low_hz}-{high_hz} Hz)")
        
        # Now fill in missing electrodes by interpolating from available data
        power_values = self._fill_missing_electrodes(power_values_raw, freq_band)
        
        return power_values
    
    def _create_boundary_mask(self, grid_x, grid_y, boundary_electrode_names):
        """
        Create a mask for a region bounded by specified electrodes.
        
        Parameters:
        -----------
        grid_x : np.ndarray
            X coordinates of the grid
        grid_y : np.ndarray
            Y coordinates of the grid
        boundary_electrode_names : list
            List of electrode channel names that form the boundary (e.g., ['Fp1', 'Fp2', 'F4', ...])
            
        Returns:
        --------
        mask : np.ndarray
            Boolean mask where True indicates inside the boundary region
        """
        # Extract boundary electrode coordinates
        boundary_coords = []
        for elec_name in boundary_electrode_names:
            for elec_num, coords in self.electrode_coords.items():
                if coords['channel'] == elec_name:
                    boundary_coords.append([coords['x'], coords['y']])
                    break
        
        if len(boundary_coords) < 3:
            print(f"Warning: Could not find enough boundary electrodes. Found {len(boundary_coords)}, need at least 3.")
            return np.ones(grid_x.shape, dtype=bool)
        
        boundary_coords = np.array(boundary_coords)
        
        # Use Delaunay triangulation to determine if points are inside the boundary
        try:
            tri = Delaunay(boundary_coords)
            # Reshape grid for point checking
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            mask_flat = tri.find_simplex(grid_points) >= 0
            mask = mask_flat.reshape(grid_x.shape)
        except Exception as e:
            print(f"Error creating Delaunay triangulation: {e}")
            # Fallback: create a simple polygon mask
            from matplotlib.path import Path as MPLPath
            boundary_path = MPLPath(boundary_coords)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            mask_flat = boundary_path.contains_points(grid_points)
            mask = mask_flat.reshape(grid_x.shape)
        
        return mask
    
    def plot_power_on_head(self, power_values, title="EEG Power Map", 
                          freq_band=(5, 15), cmap='RdYlBu_r',
                          boundary_electrodes=None,
                          electrode_size=300):
        """
        Plot power values on electrode map with interpolation within a boundary region.
        
        Parameters:
        -----------
        power_values : dict
            Dictionary with channel names and power values (includes interpolated values)
        title : str
            Title for the plot
        freq_band : tuple
            Frequency band for reference in title
        cmap : str
            Colormap to use (red for high, blue for low)
        boundary_electrodes : list, optional
            List of electrode channel names defining the boundary region for interpolation
            e.g., ['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
            If None, shows entire region
        electrode_size : int
            Size of electrode markers
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Get ALL electrode coordinates from the map (not just those with data)
        electrode_positions = []
        electrode_powers = []
        electrode_channels = []
        electrode_numbers = []
        
        for elec_num, coords in sorted(self.electrode_coords.items()):
            channel = coords['channel']
            
            # Get power value (should be available now due to interpolation)
            if channel in power_values:
                x = coords['x']
                y = coords['y']
                electrode_positions.append([x, y])
                electrode_powers.append(power_values[channel])
                electrode_channels.append(channel)
                electrode_numbers.append(elec_num)
            else:
                print(f"Warning: No power value found for channel {channel} (electrode {elec_num})")
        
        electrode_positions = np.array(electrode_positions)
        electrode_powers = np.array(electrode_powers)
        
        # Normalize power to [0, 1] for color mapping
        power_min, power_max = electrode_powers.min(), electrode_powers.max()
        if power_max > power_min:
            norm = Normalize(vmin=power_min, vmax=power_max)
        else:
            norm = Normalize(vmin=0, vmax=1)
        
        colormap = cm.get_cmap(cmap)
        
        # Create interpolation grid based on electrode positions
        grid_resolution = 150
        x_min, x_max = electrode_positions[:, 0].min() - 1, electrode_positions[:, 0].max() + 1
        y_min, y_max = electrode_positions[:, 1].min() - 1, electrode_positions[:, 1].max() + 1
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution)
        )
        
        # Interpolate power values onto the grid using cubic interpolation for smoother surface
        grid_power = griddata(
            electrode_positions, 
            electrode_powers, 
            (grid_x, grid_y), 
            method='cubic'
        )
        
        # Fill NaN values at the edges with nearest-neighbor interpolation
        grid_power_filled = griddata(
            electrode_positions, 
            electrode_powers, 
            (grid_x, grid_y), 
            method='nearest'
        )
        
        # Use cubic interpolation where available, nearest where not
        grid_power = np.where(np.isnan(grid_power), grid_power_filled, grid_power)
        
        # Apply boundary mask if specified
        if boundary_electrodes is not None:
            boundary_mask = self._create_boundary_mask(grid_x, grid_y, boundary_electrodes)
            grid_power = np.where(boundary_mask, grid_power, np.nan)
        
        # Plot the interpolated power field as filled surface using pcolormesh for smooth visualization
        mesh = ax.pcolormesh(
            grid_x, grid_y, grid_power,
            cmap=colormap,
            norm=norm,
            shading='auto',
            zorder=1,
            alpha=0.9
        )
        
        # Add contourf for better visual definition
        levels = np.linspace(power_min, power_max, 25)
        contourf = ax.contourf(
            grid_x, grid_y, grid_power, 
            levels=levels, 
            cmap=colormap, 
            norm=norm,
            extend='both',
            zorder=2,
            alpha=0.7
        )
        
        # Add contour lines for reference
        contour = ax.contour(
            grid_x, grid_y, grid_power,
            levels=8,
            colors='gray',
            alpha=0.4,
            linewidths=0.8,
            zorder=3
        )
        
        # Plot ALL electrodes on top with markers and labels
        for i, (pos, power, channel, elec_num) in enumerate(zip(electrode_positions, electrode_powers, electrode_channels, electrode_numbers)):
            x_scaled, y_scaled = pos
            color = colormap(norm(power))
            
            # Plot electrode as a circle with edge - larger and more prominent
            ax.scatter(x_scaled, y_scaled, s=electrode_size, c=[color], 
                      edgecolors='black', linewidth=3, zorder=10, alpha=1.0)
            
            # Add a small highlight ring around the electrode
            circle = plt.Circle((x_scaled, y_scaled), 0.25, fill=False, 
                               edgecolor='white', linewidth=1.5, zorder=9, alpha=0.8)
            ax.add_patch(circle)
            
            # Add channel name label (e.g., P4, C2, etc.)
            ax.text(x_scaled, y_scaled, channel, 
                   ha='center', va='center', fontsize=11, fontweight='bold', 
                   color='white', zorder=11, bbox=dict(boxstyle='round,pad=0.4', 
                                                       facecolor='black', alpha=0.7,
                                                       edgecolor='white', linewidth=1))
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Power (µV²/Hz)', shrink=0.85, pad=0.02)
        
        # Labels and formatting - use electrode coordinates for margins
        margin = 1.0
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title with frequency band info
        full_title = f"{title}\n{freq_band[0]}-{freq_band[1]} Hz Band"
        plt.title(full_title, fontsize=14, fontweight='bold', pad=20)
        
        # Add statistics box
        stats_text = f"Power Range: {power_min:.2f} - {power_max:.2f} µV²/Hz\n"
        stats_text += f"Mean: {electrode_powers.mean():.2f} µV²/Hz\n"
        stats_text += f"Electrodes: {len(electrode_channels)}/{len(self.electrode_coords)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def _fill_missing_electrodes(self, power_values_raw, freq_band):
        """
        Fill in missing electrode channels by interpolating from available data.
        Uses inverse distance weighting with distance decay to ensure interpolated 
        values don't exceed measured values and decay appropriately.
        
        Parameters:
        -----------
        power_values_raw : dict
            Dictionary with power values for available channels
        freq_band : tuple
            Frequency band info for reporting
            
        Returns:
        --------
        power_values : dict
            Dictionary with all electrode channels as keys, including interpolated values
        """
        power_values = dict(power_values_raw)  # Start with available data
        
        # Get all electrode positions
        electrode_positions = {}
        for elec_num, coords in self.electrode_coords.items():
            electrode_positions[coords['channel']] = (coords['x'], coords['y'])
        
        # Find missing channels
        available_channels = set(power_values_raw.keys())
        all_channels = set(electrode_positions.keys())
        missing_channels = all_channels - available_channels
        
        if missing_channels:
            print(f"\nMissing channels: {missing_channels}")
            print("Interpolating power values for missing channels with distance decay...")
            
            # Convert available data to numpy arrays for interpolation
            available_pos = np.array([electrode_positions[ch] for ch in available_channels])
            available_powers = np.array([power_values_raw[ch] for ch in available_channels])
            
            # Get statistics of available data
            max_power = available_powers.max()
            min_power = available_powers.min()
            mean_power = available_powers.mean()
            
            # Interpolate missing values using inverse distance weighting with decay
            for missing_channel in missing_channels:
                missing_pos = np.array(electrode_positions[missing_channel])
                
                # Calculate distances from missing electrode to all available electrodes
                distances = np.sqrt(np.sum((available_pos - missing_pos) ** 2, axis=1))
                
                # Use inverse distance weighting with exponential decay
                # Decay factor ensures values decrease as distance increases
                decay_factor = 2.0  # Controls how quickly values decay with distance
                weights = np.exp(-decay_factor * distances)
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Weighted average of available power values
                interpolated_power = np.sum(weights * available_powers)
                
                # Apply additional damping: interpolated values should not exceed max of measured values
                # but should be reasonable estimates based on surrounding electrodes
                interpolated_power = np.clip(interpolated_power, min_power, max_power * 0.95)
                
                power_values[missing_channel] = float(interpolated_power)
                print(f"  {missing_channel}: {interpolated_power:.6f} µV²/Hz (interpolated, distance decay applied)")
        
        return power_values
    
    def _extract_electrode_number(self, channel_name):
        """
        Extract electrode number from channel name.
        Uses the channel mapping if available, otherwise falls back to 10-20 label mapping.
        
        Parameters:
        -----------
        channel_name : str
            Channel name (e.g., "P4", "C2", "Fz", etc.)
            
        Returns:
        --------
        int
            Electrode number (1-16), or -1 if not found
        """
        # First try using the loaded channel mapping
        if self.channel_mapping and channel_name in self.channel_mapping:
            return self.channel_mapping[channel_name]
        
        # Fallback: mapping from 10-20 labels to electrode numbers
        label_to_num = {
            'fp1': 15,   # Frontal Pole Left
            'fp2': 14,   # Frontal Pole Right
            'f3': 13,    # Frontal Left
            'f4': 3,     # Frontal Right
            'fz': 11,    # Frontal Central
            'c1': 9,     # Central Left
            'c2': 2,     # Central Right
            'cz': 12,    # Central Central (approximate)
            'p3': 5,     # Parietal Left
            'p4': 1,     # Parietal Right
            'pz': 4,     # Parietal Central
            't1': 6,     # Temporal Left (lower)
            't2': 16,    # Temporal Right (lower)
            't5': 7,     # Temporal Left (upper)
            't6': 8,     # Temporal Right (upper)
            'o1': 10,    # Occipital Left
            'o2': 12,    # Occipital Right
            'oz': 11,    # Occipital Central (approximate)
        }
        
        # Normalize the channel name
        name = channel_name.lower().strip()
        
        # Try to find in mapping
        if name in label_to_num:
            return label_to_num[name]
        
        # If not found, return -1
        print(f"Warning: Channel '{channel_name}' not found in mapping")
        return -1
    
    def visualize(self, start_time=0, duration=5, freq_band=(5, 15), 
                  title="EEG Power Spectral Density", boundary_electrodes=None):
        """
        Complete pipeline: extract window, compute PSD, and visualize.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        duration : float
            Duration of the window in seconds
        freq_band : tuple
            Frequency band (low_hz, high_hz) for PSD analysis
        title : str
            Title for the plot
        boundary_electrodes : list, optional
            List of electrode channel names defining the boundary region
            e.g., ['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
        """
        print(f"\n{'='*60}")
        print(f"Visualizing EEG Power: {freq_band[0]}-{freq_band[1]} Hz")
        print(f"{'='*60}\n")
        
        # Extract time window
        window_data = self.extract_window(start_time, duration)
        
        # Compute power in frequency band
        print(f"\nComputing power in {freq_band[0]}-{freq_band[1]} Hz band using Welch's method:")
        power_values = self.compute_psd_power(window_data, freq_band=freq_band)
        
        # Plot
        print(f"\nGenerating plot...")
        fig, ax = self.plot_power_on_head(power_values, title=title, freq_band=freq_band, 
                                          boundary_electrodes=boundary_electrodes)
        
        return fig, ax, power_values
    
    def generate_video(self, start_time=0, total_duration=30, window_duration=2, freq_band=(5, 15),
                      output_path="eeg_power_animation.mp4", fps=10, 
                      boundary_electrodes=None, cmap='RdYlBu_r', dpi=100, power_range=None):
        """
        Generate an animated video showing power changes over time.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        total_duration : float
            Total duration to animate (in seconds)
        window_duration : float
            Duration of each analysis window (in seconds)
        freq_band : tuple
            Frequency band (low_hz, high_hz) for PSD analysis
        output_path : str
            Output file path for the video (e.g., "animation.mp4")
        fps : int
            Frames per second for the video
        boundary_electrodes : list, optional
            List of electrode channel names defining the boundary region
            e.g., ['Fp1', 'Fp2', 'F4', 'T6', 'O2', 'O1', 'T5', 'F3']
        cmap : str
            Colormap to use (red for high, blue for low)
        dpi : int
            DPI for figure rendering
            
        Returns:
        --------
        output_path : str
            Path to the generated video file
        """
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        
        print(f"\n{'='*60}")
        print(f"Generating EEG Power Video Animation")
        print(f"{'='*60}")
        print(f"Duration: {total_duration}s with {window_duration}s windows")
        print(f"Frequency band: {freq_band[0]}-{freq_band[1]} Hz")
        print(f"Output: {output_path}")
        print(f"FPS: {fps}")
        print(f"{'='*60}\n")
        
        # Calculate number of frames to match desired video duration
        # Video duration (in seconds) = num_frames / fps
        # So: num_frames = video_duration_seconds * fps
        # But we want to cover total_duration seconds of data with window_duration windows
        # Using 50% overlap: step_size = window_duration / 2
        
        # Calculate step size for 50% overlap between windows
        step_size = window_duration / 2
        
        # Generate time points that cover the entire total_duration
        # Last window should end at start_time + total_duration
        time_points = []
        t = start_time
        while t + window_duration <= start_time + total_duration:
            time_points.append(t)
            t += step_size
        
        # Add final frame if needed to reach the end
        if len(time_points) == 0 or time_points[-1] + window_duration < start_time + total_duration:
            time_points.append(start_time + total_duration - window_duration)
        
        time_points = np.array(time_points)
        num_frames = len(time_points)
        
        # Calculate actual FPS to achieve desired video duration
        actual_video_duration = num_frames / fps
        print(f"Creating {num_frames} animation frames (video will be ~{actual_video_duration:.1f}s at {fps} FPS)...")
        
        # Get electrode positions for static setup
        electrode_positions = []
        electrode_channels = []
        electrode_numbers = []
        
        for elec_num, coords in sorted(self.electrode_coords.items()):
            channel = coords['channel']
            electrode_positions.append([coords['x'], coords['y']])
            electrode_channels.append(channel)
            electrode_numbers.append(elec_num)
        
        electrode_positions = np.array(electrode_positions)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
        
        # Precompute power values for all time points
        print("Precomputing power values for all time points...")
        all_power_values = []
        power_min_global = float('inf')
        power_max_global = float('-inf')
        
        for i, t in enumerate(time_points):
            if i % max(1, num_frames // 10) == 0:
                print(f"  Computing power: {i+1}/{num_frames} frames...")
            
            window_data = self.extract_window(t, window_duration)
            power_values = self.compute_psd_power(window_data, freq_band=freq_band)
            
            # Get powers in consistent order
            powers = np.array([power_values.get(ch, 0) for ch in electrode_channels])
            all_power_values.append(powers)
            
            # Track global min/max for consistent color scaling
            power_min_global = min(power_min_global, powers.min())
            power_max_global = max(power_max_global, powers.max())
        
        all_power_values = np.array(all_power_values)
        
        # Determine power range for color scaling
        if power_range is not None and power_range[0] is not None and power_range[1] is not None:
            # Use fixed power range
            power_min, power_max = power_range[0], power_range[1]
            print(f"Using fixed power range: {power_min}-{power_max} µV²/Hz")
        else:
            # Auto-scale based on data
            power_range_calc = power_max_global - power_min_global
            
            # Safety check: if power_range is 0 (all values are the same), add a small range
            if power_range_calc == 0:
                print("⚠️  Warning: All power values are identical (power_range = 0)")
                print(f"   This may indicate the window_duration is too short for stable PSD estimation.")
                print(f"   Recommend using window_duration >= 1 second (current: {window_duration}s)")
                # Add a small artificial range to prevent normalization errors
                power_range_calc = power_min_global * 0.1 if power_min_global != 0 else 1.0
            
            power_min = power_min_global - 0.05 * power_range_calc
            power_max = power_max_global + 0.05 * power_range_calc
            print(f"Auto-scaled power range: {power_min:.2f}-{power_max:.2f} µV²/Hz")
        
        norm = Normalize(vmin=power_min, vmax=power_max)
        colormap = cm.get_cmap(cmap)
        
        # Create grid for interpolation (computed once)
        grid_resolution = 150
        x_min, x_max = electrode_positions[:, 0].min() - 1, electrode_positions[:, 0].max() + 1
        y_min, y_max = electrode_positions[:, 1].min() - 1, electrode_positions[:, 1].max() + 1
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution)
        )
        
        # Create boundary mask (computed once)
        if boundary_electrodes is not None:
            boundary_mask = self._create_boundary_mask(grid_x, grid_y, boundary_electrodes)
        else:
            boundary_mask = np.ones(grid_x.shape, dtype=bool)
        
        # Create a dummy ScalarMappable for the colorbar (so it appears only once)
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        
        # Initialize plot elements
        mesh = None
        contourf = None
        contour_lines = None
        electrode_scatter = None
        electrode_labels = None
        colorbar = None
        
        def animate(frame_idx):
            nonlocal mesh, contourf, contour_lines, electrode_scatter, electrode_labels, colorbar
            
            ax.clear()
            
            # Get power values for this frame
            electrode_powers = all_power_values[frame_idx]
            
            # Interpolate power values onto the grid using cubic interpolation
            grid_power = griddata(
                electrode_positions,
                electrode_powers,
                (grid_x, grid_y),
                method='cubic'
            )
            
            # Fill NaN values at edges with nearest-neighbor interpolation
            grid_power_filled = griddata(
                electrode_positions,
                electrode_powers,
                (grid_x, grid_y),
                method='nearest'
            )
            
            grid_power = np.where(np.isnan(grid_power), grid_power_filled, grid_power)
            
            # Apply boundary mask
            grid_power = np.where(boundary_mask, grid_power, np.nan)
            
            # Plot the interpolated power field
            mesh = ax.pcolormesh(
                grid_x, grid_y, grid_power,
                cmap=colormap,
                norm=norm,
                shading='auto',
                zorder=1,
                alpha=0.9
            )
            
            # Add contourf for better visual definition
            levels = np.linspace(power_min, power_max, 25)
            contourf = ax.contourf(
                grid_x, grid_y, grid_power,
                levels=levels,
                cmap=colormap,
                norm=norm,
                extend='both',
                zorder=2,
                alpha=0.7
            )
            
            # Add contour lines
            contour_lines = ax.contour(
                grid_x, grid_y, grid_power,
                levels=8,
                colors='gray',
                alpha=0.4,
                linewidths=0.8,
                zorder=3
            )
            
            # Plot electrodes on top
            electrode_scatter = ax.scatter(
                electrode_positions[:, 0],
                electrode_positions[:, 1],
                c=electrode_powers,
                s=300,
                cmap=colormap,
                norm=norm,
                edgecolors='white',
                linewidths=2,
                zorder=5,
                alpha=1.0
            )
            
            # Add electrode labels
            for pos, channel in zip(electrode_positions, electrode_channels):
                ax.text(
                    pos[0], pos[1],
                    channel,
                    ha='center', va='center',
                    fontsize=20,
                    fontweight='bold',
                    color='black',
                    zorder=6
                )
            
            # Draw boundary line if specified
            if boundary_electrodes is not None:
                boundary_coords = []
                for elec_name in boundary_electrodes:
                    for elec_num, coords in self.electrode_coords.items():
                        if coords['channel'] == elec_name:
                            boundary_coords.append([coords['x'], coords['y']])
                            break
                
                if len(boundary_coords) >= 2:
                    # Close the polygon
                    boundary_coords_array = np.array(boundary_coords)
                    boundary_coords_closed = np.vstack([boundary_coords_array, boundary_coords_array[0]])
                    ax.plot(
                        boundary_coords_closed[:, 0],
                        boundary_coords_closed[:, 1],
                        'k-',
                        linewidth=2,
                        zorder=4,
                        alpha=0.5
                    )
            
            # Update title with current time
            current_time = time_points[frame_idx]
            window_end = current_time + window_duration
            ax.set_title(
                f"EEG Power Map: {freq_band[0]}-{freq_band[1]} Hz\n"
                f"Time: {current_time:.1f}s - {window_end:.1f}s (Frame {frame_idx+1}/{num_frames})",
                fontsize=14,
                fontweight='bold'
            )
            
            ax.set_aspect('equal')
            ax.axis('on')
            
            # Build return list of artists for animation (blit=False, so we don't need to return all artists)
            # Just return an empty list since we're not using blit mode
            return []
        
        # Create animation
        print("Creating animation...")
        anim = animation.FuncAnimation(
            fig, animate,
            frames=num_frames,
            interval=1000/fps,  # milliseconds between frames
            blit=False,
            repeat=True
        )
        
        # Add a single colorbar to the figure (outside animation loop)
        cbar = plt.colorbar(
            sm,
            ax=ax,
            label=f'Power (µV²/Hz)',
            pad=0.02,
            fraction=0.046
        )
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
        anim.save(output_path, writer=writer, dpi=dpi)
        
        plt.close(fig)
        
        print(f"\n{'='*60}")
        print(f"Video generation complete!")
        print(f"Output: {output_path}")
        print(f"Duration: ~{num_frames/fps:.1f} seconds at {fps} FPS")
        print(f"{'='*60}\n")
        
        return output_path
    
    def save_results(self, output_dir, power_values, freq_band):
        """
        Save power values to a CSV file for reference.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        power_values : dict
            Dictionary with power values
        freq_band : tuple
            Frequency band info for filename
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Channel': list(power_values.keys()),
            'Power_uV2_Hz': list(power_values.values())
        })
        
        # Save
        freq_str = f"{freq_band[0]}-{freq_band[1]}Hz"
        output_file = Path(output_dir) / f"eeg_power_{freq_str}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return output_file

    def generate_montage_video_with_marker(self, start_time=0, total_duration=30, output_path="montage_animation.mp4",
                                          fps=10, dpi=100, spacing=400, spacing_cluster=800, limit=500,
                                          window_width=10, playback_speed=0.2):
        """
        Generate a video showing EEG montage plot with a synchronized moving vertical line marker.
        
        Parameters:
        -----------
        start_time : float
            Start time in seconds
        total_duration : float
            Total duration of data to show (in seconds)
        output_path : str
            Output file path for the video
        fps : int
            Frames per second for the video
        dpi : int
            DPI for figure rendering
        spacing : int
            Spacing between montage pairs in uV
        spacing_cluster : int
            Spacing between montage clusters in uV
        limit : int
            Y-axis limit padding in uV
        window_width : float
            Width of the visible time window in seconds (default: 10s)
        playback_speed : float
            How fast time advances per video second (default: 0.2 means 0.2s data per 1s video)
            For 110s of data with 0.2 playback speed, video will be ~550s (9 min) long
            
        Returns:
        --------
        output_path : str
            Path to the generated video file
        """
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        
        print(f"\n{'='*60}")
        print(f"Generating EEG Montage Video with Moving Marker")
        print(f"{'='*60}")
        print(f"Data duration: {total_duration}s")
        print(f"Time range: {start_time}s to {start_time + total_duration}s")
        print(f"Window width: {window_width}s")
        print(f"Playback speed: {playback_speed}s data per 1s video")
        print(f"Video duration: ~{total_duration / playback_speed}s")
        print(f"Output: {output_path}")
        print(f"FPS: {fps}")
        print(f"{'='*60}\n")
        
        # Extract time window for the entire duration
        window_data = self.extract_window(start_time, total_duration)
        
        # Define montage groups
        montage_groups = [
            [('C1', 'P3'), ('P3', 'T5'), ('T5', 'O1')],
            [('C1', 'Pz'), ('Pz', 'O1')],
            [('C2', 'Pz'), ('Pz', 'O2')],
            [('C2', 'P4'), ('P4', 'T6'), ('T6', 'O2')],
        ]
        
        # Get available channels
        available_channels = set(window_data.columns)
        
        # Build list of valid montage pairs
        valid_pairs = []
        for group in montage_groups:
            for ch1, ch2 in group:
                if ch1 in available_channels and ch2 in available_channels:
                    valid_pairs.append((ch1, ch2))
        
        print(f"Found {len(valid_pairs)} valid montage pairs")
        
        if len(valid_pairs) == 0:
            print("Error: No valid montage pairs found!")
            return None
        
        # Prepare data for all pairs
        time_array = np.arange(len(window_data)) / self.sampling_rate
        
        montage_data_dict = {}
        for ch1, ch2 in valid_pairs:
            data_ch1 = window_data[ch1].values
            data_ch2 = window_data[ch2].values
            montage_data_dict[(ch1, ch2)] = data_ch1 - data_ch2
        
        # Calculate y-axis positions
        num_pairs = len(valid_pairs)
        y_offsets = []
        y_labels = []
        
        # Start from top
        current_y = num_pairs * spacing
        for i, (ch1, ch2) in enumerate(valid_pairs):
            y_offsets.append(current_y)
            y_labels.append(f'{ch1}-{ch2}')
            current_y -= spacing
        
        y_min = min(y_offsets) - limit * 1.2
        y_max = max(y_offsets) + limit * 1.2
        
        # Create figure with 2:1 aspect ratio (width:height)
        fig, ax = plt.subplots(figsize=(20, 8), dpi=dpi)
        
        # Calculate number of frames based on playback speed
        # If playback_speed = 0.2, then 1 second of video shows 0.2 seconds of data
        # So video_duration = total_duration / playback_speed
        video_duration = total_duration / playback_speed
        num_frames = int(video_duration * fps)
        
        print(f"Creating {num_frames} animation frames...")
        print(f"Video will be {video_duration:.1f} seconds long at {fps} FPS")
        
        # Initialize plot elements
        trace_lines = []
        marker_line = None
        title_text = None
        grid_lines = []
        
        def animate(frame_idx):
            nonlocal marker_line, title_text, trace_lines, grid_lines
            
            # Calculate current time position in the data
            current_time = (frame_idx / num_frames) * total_duration
            current_time_abs = start_time + current_time
            
            # Calculate window boundaries
            # Center the marker in the window
            window_start = current_time - window_width / 2
            window_end = current_time + window_width / 2
            
            # Adjust if we're at the beginning or end
            if window_start < 0:
                window_start = 0
                window_end = window_width
            elif window_end > total_duration:
                window_end = total_duration
                window_start = total_duration - window_width
            
            # Find indices for this window
            idx_start = int(window_start * self.sampling_rate)
            idx_end = int(window_end * self.sampling_rate)
            idx_start = max(0, idx_start)
            idx_end = min(len(time_array), idx_end)
            
            # Clear old plot elements
            for line in trace_lines:
                line.remove()
            trace_lines.clear()
            
            for line in grid_lines:
                line.remove()
            grid_lines.clear()
            
            if marker_line is not None:
                marker_line.remove()
            
            if title_text is not None:
                title_text.remove()
            
            # Plot traces in the current window
            time_window = time_array[idx_start:idx_end] - window_start
            
            for i, (ch1, ch2) in enumerate(valid_pairs):
                montage = montage_data_dict[(ch1, ch2)][idx_start:idx_end]
                line, = ax.plot(time_window, montage + y_offsets[i], 
                              color='black', linewidth=0.8, alpha=0.8, zorder=3)
                trace_lines.append(line)
            
            # Add grid lines
            for t in np.arange(0, window_width + 1, 1):
                if t <= window_width:
                    if int(window_start + t) % 5 == 0:
                        line = ax.axvline(x=t, color='#1a1a1a', alpha=0.8, 
                                        linewidth=1.5, linestyle='-', zorder=1)
                    else:
                        line = ax.axvline(x=t, color='#404040', alpha=0.4, 
                                        linewidth=0.8, linestyle='-', zorder=1)
                    grid_lines.append(line)
            
            # Draw marker line at the center (current position)
            marker_position = current_time - window_start
            marker_line = ax.axvline(x=marker_position, color='red', 
                                    linewidth=2.5, alpha=0.9, zorder=10)
            
            # Add time indicator title
            title_text = ax.text(0.5, 0.98, 
                               f"Time: {current_time_abs:.1f}s (Frame {frame_idx+1}/{num_frames})",
                               transform=ax.transAxes, ha='center', va='top',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Update axes
            ax.set_xlim(0, window_width)
            
            return trace_lines + grid_lines + [marker_line, title_text]
        
        # Set up static elements
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Voltage (μV)', fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_offsets)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.grid(True, alpha=0.2, zorder=0)
        
        # Add scale bar (static, positioned in data coordinates)
        scale_bar_height = 200
        scale_bar_x = 0.5  # Fixed position in window
        scale_bar_y_start = max(y_offsets) + 500
        scale_bar_y_end = scale_bar_y_start - scale_bar_height
        
        ax.plot([scale_bar_x, scale_bar_x],
               [scale_bar_y_start, scale_bar_y_end],
               'r-', linewidth=2, zorder=5)
        ax.text(scale_bar_x + 0.3, (scale_bar_y_start + scale_bar_y_end) / 2,
               f'{scale_bar_height} μV', fontsize=10, va='center')
        
        # Create animation
        print("Creating animation...")
        anim = animation.FuncAnimation(
            fig, animate,
            frames=num_frames,
            interval=1000/fps,
            blit=False,
            repeat=True
        )
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
        anim.save(output_path, writer=writer, dpi=dpi)
        
        plt.close(fig)
        
        print(f"\n{'='*60}")
        print(f"Video generation complete!")
        print(f"Output: {output_path}")
        print(f"Video duration: ~{num_frames/fps:.1f} seconds at {fps} FPS")
        print(f"Data shown: {total_duration}s at {playback_speed}x speed")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    """Main execution function with example usage."""
    
    # ============================================================
    # CONFIGURATION - Modify these parameters
    # ============================================================
    
    # File paths
    csv_path = "/Volumes/CHOO'S SSD/LINK/1. 박진봉 교수님 데이터/1st_trial/processed_b_channels_2000hz_80hz_filtered.csv"
    electrode_map_path = "/Users/sean/LINK/MEEG/electrode_map/Version2_Fixed.json"
    
    # EEG parameters
    sampling_rate = 2000  # Hz - from the filename
    
    # Analysis parameters
    start_time = 0  # Start time in seconds
    duration = 5    # Duration in seconds
    freq_band = (5, 15)  # Frequency band for PSD analysis (Hz)
    
    # Output
    output_dir = "/Users/sean/LINK/MEEG/Figure/results"
    
    # ============================================================
    # EXECUTION
    # ============================================================
    
    try:
        # Initialize visualizer
        visualizer = EEGPowerVisualizer(csv_path, electrode_map_path, sampling_rate)
        
        # Generate visualization
        fig, ax, power_values = visualizer.visualize(
            start_time=start_time,
            duration=duration,
            freq_band=freq_band,
            title=f"EEG Power Map\nWindow: {start_time}-{start_time+duration}s"
        )
        
        # Save results
        visualizer.save_results(output_dir, power_values, freq_band)
        
        # Display
        plt.show()
        
        print(f"\n{'='*60}")
        print("Visualization complete!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
