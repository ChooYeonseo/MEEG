"""
Topography Window for MEEG Seizure Labeling.

This module displays the EEG power topography visualization with frequency band selection.
Based on Figure1E.py implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.signal import welch
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, ConvexHull
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QDialog, QFormLayout,
                             QDoubleSpinBox, QDialogButtonBox, QButtonGroup,
                             QRadioButton, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class FixedScaleDialog(QDialog):
    """Dialog for setting fixed power scale."""
    
    def __init__(self, current_min=None, current_max=None, unit="μV²/Hz", theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.setWindowTitle("Fixed Power Scale")
        self.current_min = current_min
        self.current_max = current_max
        self.unit = unit
        
        # Set background color to match theme
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(self.theme_colors['bg_primary']))
        self.setPalette(palette)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QFormLayout(self)
        
        # Min power spinbox
        self.min_power_spin = QDoubleSpinBox()
        self.min_power_spin.setRange(-1000000.0, 1000000.0)
        self.min_power_spin.setValue(self.current_min if self.current_min is not None else 0.0)
        self.min_power_spin.setSuffix(f" {self.unit}")
        self.min_power_spin.setDecimals(2)
        layout.addRow("Min Power:", self.min_power_spin)
        
        # Max power spinbox
        self.max_power_spin = QDoubleSpinBox()
        self.max_power_spin.setRange(-1000000.0, 1000000.0)
        self.max_power_spin.setValue(self.current_max if self.current_max is not None else 200.0)
        self.max_power_spin.setSuffix(f" {self.unit}")
        self.max_power_spin.setDecimals(2)
        layout.addRow("Max Power:", self.max_power_spin)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
    def get_scale_range(self):
        """Get the selected scale range."""
        return (self.min_power_spin.value(), self.max_power_spin.value())


class CustomFrequencyDialog(QDialog):
    """Dialog for entering custom frequency range."""
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.setWindowTitle("Custom Frequency Range")
        
        # Set background color to match theme
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(self.theme_colors['bg_primary']))
        self.setPalette(palette)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QFormLayout(self)
        
        # Low frequency spinbox
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(0.1, 10000.0)
        self.low_freq_spin.setValue(5.0)
        self.low_freq_spin.setSuffix(" Hz")
        self.low_freq_spin.setDecimals(1)
        layout.addRow("Low Frequency:", self.low_freq_spin)
        
        # High frequency spinbox
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(0.1, 10000.0)
        self.high_freq_spin.setValue(15.0)
        self.high_freq_spin.setSuffix(" Hz")
        self.high_freq_spin.setDecimals(1)
        layout.addRow("High Frequency:", self.high_freq_spin)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
    def get_frequency_range(self):
        """Get the selected frequency range."""
        return (self.low_freq_spin.value(), self.high_freq_spin.value())


class TopographyWidget(QWidget):
    """Widget for displaying EEG power topography with frequency band selection."""
    
    # Predefined frequency bands
    FREQ_BANDS = {
        'Delta': (0.5, 4, 'Deep sleep, unconsciousness'),
        'Theta': (4, 8, 'Drowsiness, meditation, memory'),
        'Alpha': (8, 12, 'Relaxed wakefulness, eyes closed'),
        'Beta': (12, 30, 'Active thinking, problem solving'),
        'Gamma': (30, 100, 'High-level cognition, learning'),
    }
    
    def __init__(self, electrode_positions=None, theme_colors=None, parent=None):
        super().__init__(parent)
        self.electrode_positions = electrode_positions or []
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.current_epoch = 0
        self.epoch_data = None
        self.sampling_rate = 2000  # Default sampling rate
        self.epoch_length = 10  # Default epoch length in seconds
        
        # Current frequency band
        self.current_freq_band = (5, 15)  # Default to Alpha-like range
        
        # Fixed scale settings
        self.fixed_scale_enabled = False
        self.fixed_scale_min = None
        self.fixed_scale_max = None
        
        # Display settings
        self.calculation_method = 'mean'  # 'sum' or 'mean' (Default to mean as requested)
        self.interpolation_enabled = True # Default to True
        
        # Cache for boundary electrodes (computed once)
        self._cached_boundary = None
        
        # Electrode coordinates dictionary
        self.electrode_coords = {}
        self._prepare_electrode_coords()
        
        self.init_ui()
        
    def _prepare_electrode_coords(self):
        """Convert electrode positions to coordinate dictionary."""
        for electrode in self.electrode_positions:
            channel = electrode.get('channel', electrode.get('name', f"E{electrode['number']}"))
            self.electrode_coords[electrode['number']] = {
                'x': electrode['x'],
                'y': electrode['y'],
                'channel': channel
            }
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("EEG Power Topography")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Frequency band selection panel
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)
        
        # Fixed Scale button
        self.fixed_scale_btn = QPushButton("Fixed Scale")
        self.fixed_scale_btn.setCheckable(True)
        self.fixed_scale_btn.setChecked(False)
        self.fixed_scale_btn.clicked.connect(self.on_fixed_scale_clicked)
        self.fixed_scale_btn.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
        control_layout.addWidget(self.fixed_scale_btn)
        
        # Create radio buttons for predefined bands
        self.band_group = QButtonGroup(self)
        self.band_buttons = {}  # Store buttons for styling
        
        for band_name in self.FREQ_BANDS.keys():
            radio = QRadioButton(band_name)
            self.band_group.addButton(radio)
            self.band_buttons[band_name] = radio
            radio.clicked.connect(lambda checked, name=band_name: self.on_band_selected(name))
            # Apply theme colors to radio button text
            radio.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
            control_layout.addWidget(radio)
        
        # Default selection (Alpha)
        alpha_button = [btn for btn in self.band_group.buttons() if btn.text() == 'Alpha']
        if alpha_button:
            alpha_button[0].setChecked(True)
            self._highlight_selected_button(alpha_button[0])
        
        # Custom button
        self.custom_radio = QRadioButton("Custom")
        self.band_group.addButton(self.custom_radio)
        self.band_buttons['Custom'] = self.custom_radio
        self.custom_radio.clicked.connect(self.on_custom_clicked)
        self.custom_radio.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
        control_layout.addWidget(self.custom_radio)
        
        control_layout.addStretch()
        
        # New Settings Group (Interpolation & Calculation)
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(2)
        
        # Interpolation Toggle
        self.interp_check = QCheckBox("Interpolate")
        self.interp_check.setChecked(self.interpolation_enabled)
        self.interp_check.clicked.connect(self.on_interpolation_toggled)
        self.interp_check.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
        settings_layout.addWidget(self.interp_check)
        
        # Calculation Method Combo
        calc_layout = QHBoxLayout()
        calc_label = QLabel("Calc:")
        calc_label.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
        self.calc_combo = QComboBox()
        self.calc_combo.addItems(["Mean", "Sum"])
        self.calc_combo.setCurrentText(self.calculation_method.capitalize())
        self.calc_combo.currentTextChanged.connect(self.on_calc_method_changed)
        calc_layout.addWidget(calc_label)
        calc_layout.addWidget(self.calc_combo)
        settings_layout.addLayout(calc_layout)
        
        control_layout.addLayout(settings_layout)
        
        layout.addLayout(control_layout)
        
        # Create matplotlib figures - main plot and colorbar as separate images
        plot_colorbar_layout = QHBoxLayout()
        plot_colorbar_layout.setSpacing(5)
        
        # Main topography plot
        self.figure = Figure(figsize=(5, 5), facecolor=self.theme_colors['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        plot_colorbar_layout.addWidget(self.canvas, stretch=8)
        
        # Colorbar as separate figure - wider to show text
        self.colorbar_figure = Figure(figsize=(1.2, 5), facecolor=self.theme_colors['bg_primary'])
        self.colorbar_canvas = FigureCanvas(self.colorbar_figure)
        self.colorbar_ax = self.colorbar_figure.add_subplot(111)
        plot_colorbar_layout.addWidget(self.colorbar_canvas, stretch=2)
        
        layout.addLayout(plot_colorbar_layout)
        
        # Initialize the plot
        self.setup_topography_plot()
        
    def _highlight_selected_button(self, selected_button):
        """Highlight the selected frequency band button."""
        # Reset all buttons to default style with theme colors
        for btn in self.band_buttons.values():
            btn.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
        
        # Highlight selected button with background color
        selected_button.setStyleSheet(f"""
            QRadioButton {{
                background-color: #4a6fa5;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QRadioButton::indicator {{
                width: 13px;
                height: 13px;
            }}
        """)
    
    def on_band_selected(self, band_name):
        """Handle frequency band selection."""
        if band_name in self.FREQ_BANDS:
            freq_info = self.FREQ_BANDS[band_name]
            self.current_freq_band = (freq_info[0], freq_info[1])
            print(f"Selected band: {band_name} ({freq_info[0]}-{freq_info[1]} Hz) - {freq_info[2]}")
            # Highlight this button
            self._highlight_selected_button(self.band_buttons[band_name])
            self.update_topography()
    
    def on_custom_clicked(self):
        """Handle custom frequency band selection."""
        dialog = CustomFrequencyDialog(self.theme_colors, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_freq_band = dialog.get_frequency_range()
            print(f"Custom frequency band: {self.current_freq_band[0]}-{self.current_freq_band[1]} Hz")
            # Highlight custom button
            self._highlight_selected_button(self.custom_radio)
            self.update_topography()
    
    def on_fixed_scale_clicked(self, checked):
        """Handle fixed scale button toggle."""
        if checked:
            # Determine unit based on method
            unit = "μV²" if self.calculation_method == 'sum' else "μV²/Hz"
            
            # Show dialog to set scale range
            dialog = FixedScaleDialog(self.fixed_scale_min, self.fixed_scale_max, unit, self.theme_colors, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.fixed_scale_min, self.fixed_scale_max = dialog.get_scale_range()
                self.fixed_scale_enabled = True
                self.fixed_scale_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #4a6fa5;
                        color: white;
                        padding: 5px 10px;
                        border-radius: 3px;
                        font-weight: bold;
                    }}
                """)
                print(f"Fixed scale enabled: {self.fixed_scale_min} to {self.fixed_scale_max} {unit}")
                self.update_topography()
            else:
                # User cancelled, uncheck the button
                self.fixed_scale_btn.setChecked(False)
        else:
            # Disable fixed scale
            self.fixed_scale_enabled = False
            self.fixed_scale_btn.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
            print("Fixed scale disabled - using auto scale")
            self.update_topography()
            
    def on_interpolation_toggled(self, checked):
        """Handle interpolation toggle."""
        self.interpolation_enabled = checked
        self.update_topography()
        
    def on_calc_method_changed(self, text):
        """Handle calculation method change."""
        self.calculation_method = text.lower()
        # Reset fixed scale if switching methods as units change
        if self.fixed_scale_enabled:
            self.fixed_scale_enabled = False
            self.fixed_scale_btn.setChecked(False)
            self.fixed_scale_btn.setStyleSheet(f"color: {self.theme_colors['fg_primary']};")
            print("Fixed scale disabled due to method change")
        self.update_topography()
    
    def setup_topography_plot(self):
        """Setup the initial topography plot."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Show placeholder text
        self.ax.text(0.5, 0.5, 'Waiting for data...', 
                    transform=self.ax.transAxes,
                    ha='center', va='center', 
                    fontsize=12, color=self.theme_colors['fg_primary'])
        
        # Clear colorbar
        self.colorbar_ax.clear()
        self.colorbar_ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.colorbar_figure.tight_layout()
        self.colorbar_canvas.draw()
    
    def set_data(self, epoch_data, sampling_rate, epoch_length):
        """
        Set the data for the topography plot.
        
        Parameters:
        -----------
        epoch_data : pd.DataFrame
            EEG data for the current epoch (channels as columns)
        sampling_rate : float
            Sampling rate in Hz
        epoch_length : float
            Epoch length in seconds
        """
        self.epoch_data = epoch_data
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.update_topography()
    
    def compute_psd_power(self, data, freq_band=(5, 15)):
        """
        Compute power spectral density for a frequency band.
        
        Parameters:
        -----------
        data : pd.DataFrame
            EEG data (channels as columns)
        freq_band : tuple
            Frequency band (low_hz, high_hz)
            
        Returns:
        --------
        power_values : dict
            Dictionary with channel names as keys and power values as values
        """
        low_hz, high_hz = freq_band
        power_values_raw = {}
        
        # Compute power for available channels
        for channel in data.columns:
            if channel == 'time':  # Skip time column if present
                continue
                
            signal = data[channel].values
            
            if len(signal) < 32:  # Minimum samples needed for minimal PSD
                continue
            
            # Use Welch's method for stable PSD estimate
            # Ensure nperseg is reasonably small for short signals but large enough for frequency resolution
            nperseg_val = min(1024, len(signal))
            if nperseg_val > 256:
                nperseg_val = min(1024, len(signal)//2)
                
            frequencies, psd = welch(signal, fs=self.sampling_rate, 
                                    nperseg=nperseg_val)
            
            # Find power in the specified frequency band
            band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            if band_mask.any():
                if self.calculation_method == 'sum':
                    band_power = np.sum(psd[band_mask])
                else: # mean
                    band_power = np.mean(psd[band_mask])
                power_values_raw[channel] = band_power
        
        # Fill in missing electrodes by interpolation ONLY if enabled
        if self.interpolation_enabled:
            power_values = self._fill_missing_electrodes(power_values_raw)
        else:
            power_values = power_values_raw
        
        return power_values
    
    def _fill_missing_electrodes(self, power_values_raw):
        """
        Fill in missing electrode channels by interpolating from available data.
        
        Parameters:
        -----------
        power_values_raw : dict
            Dictionary with power values for available channels
            
        Returns:
        --------
        power_values : dict
            Dictionary with all electrode channels, including interpolated values
        """
        power_values = dict(power_values_raw)
        
        # Get all electrode positions
        electrode_positions = {}
        for elec_num, coords in self.electrode_coords.items():
            electrode_positions[coords['channel']] = (coords['x'], coords['y'])
        
        # Find missing channels
        available_channels = set(power_values_raw.keys())
        all_channels = set(electrode_positions.keys())
        missing_channels = all_channels - available_channels
        
        if missing_channels and available_channels:
            # Convert available data to numpy arrays
            available_pos = np.array([electrode_positions[ch] for ch in available_channels])
            available_powers = np.array([power_values_raw[ch] for ch in available_channels])
            
            max_power = available_powers.max()
            min_power = available_powers.min()
            
            # Interpolate missing values using inverse distance weighting
            for missing_channel in missing_channels:
                if missing_channel not in electrode_positions:
                    continue
                    
                missing_pos = np.array(electrode_positions[missing_channel])
                
                # Calculate distances
                distances = np.sqrt(np.sum((available_pos - missing_pos) ** 2, axis=1))
                
                # Inverse distance weighting with decay
                decay_factor = 2.0
                weights = np.exp(-decay_factor * distances)
                weights = weights / np.sum(weights)
                
                # Weighted average
                interpolated_power = np.sum(weights * available_powers)
                interpolated_power = np.clip(interpolated_power, min_power, max_power * 0.95)
                
                power_values[missing_channel] = float(interpolated_power)
        
        return power_values
    
    def _get_boundary_electrodes(self):
        """Get the outer boundary electrode channels automatically using convex hull."""
        # Return cached boundary if already computed
        if self._cached_boundary is not None:
            return self._cached_boundary
        
        # Automatically detect boundary using convex hull
        if not self.electrode_coords:
            return []
        
        # Get all electrode positions and channels
        positions = []
        channels = []
        for coords in self.electrode_coords.values():
            positions.append([coords['x'], coords['y']])
            channels.append(coords['channel'])
        
        if len(positions) < 3:
            return channels  # Too few electrodes
        
        positions = np.array(positions)
        
        try:
            # Compute convex hull to find boundary electrodes
            hull = ConvexHull(positions)
            
            # Get vertices in counter-clockwise order
            hull_vertices = hull.vertices
            
            # Convert to clockwise order (reverse)
            hull_vertices = hull_vertices[::-1]
            
            # Get boundary channels in order
            boundary = [channels[i] for i in hull_vertices]
            
            # Cache the result
            self._cached_boundary = boundary
            print(f"Auto-detected boundary electrodes: {boundary}")
            return boundary
            
        except Exception as e:
            print(f"Error computing convex hull: {e}")
            # Fallback: sort by angle from center
            center = positions.mean(axis=0)
            angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            boundary = [channels[i] for i in sorted_indices]
            # Cache the result
            self._cached_boundary = boundary
            print(f"Fallback boundary (sorted by angle): {boundary}")
            return boundary
    
    def _create_boundary_mask(self, grid_x, grid_y, active_channels=None):
        """
        Create a mask for the region inside the electrode boundary.
        
        Parameters:
        -----------
        grid_x, grid_y : np.ndarray
            Meshgrid coordinates
        active_channels : list, optional
            List of active channel names to compute boundary from.
            If None, uses all electrodes (cached boundary).
        """
        if active_channels is not None:
            # Dynamic boundary based on active channels only
            boundary_channels = self._compute_hull_from_channels(active_channels)
        else:
            # Static boundary based on all electrodes
            boundary_channels = self._get_boundary_electrodes()
        
        if len(boundary_channels) < 3:
            return np.ones(grid_x.shape, dtype=bool)
        
        # Get boundary coordinates
        boundary_coords = []
        for ch in boundary_channels:
            for coords in self.electrode_coords.values():
                if coords['channel'] == ch:
                    boundary_coords.append([coords['x'], coords['y']])
                    break
        
        if len(boundary_coords) < 3:
            return np.ones(grid_x.shape, dtype=bool)
        
        boundary_coords = np.array(boundary_coords)
        
        # Create polygon mask
        from matplotlib.path import Path as MPLPath
        boundary_path = MPLPath(boundary_coords)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        mask_flat = boundary_path.contains_points(grid_points)
        mask = mask_flat.reshape(grid_x.shape)
        
        return mask
        
    def _compute_hull_from_channels(self, channel_names):
        """Compute convex hull for a specific set of channels."""
        positions = []
        valid_channels = []
        
        for ch in channel_names:
            for coords in self.electrode_coords.values():
                if coords['channel'] == ch:
                    positions.append([coords['x'], coords['y']])
                    valid_channels.append(ch)
                    break
        
        if len(positions) < 3:
            return valid_channels

        positions = np.array(positions)
        try:
            hull = ConvexHull(positions)
            hull_vertices = hull.vertices[::-1] # Clockwise
            return [valid_channels[i] for i in hull_vertices]
        except:
            return valid_channels
    
    def update_topography(self):
        """Update topography with current epoch data and frequency band."""
        if self.epoch_data is None or self.epoch_data.empty:
            self.setup_topography_plot()
            return
        
        # Compute power values
        power_values = self.compute_psd_power(self.epoch_data, self.current_freq_band)
        
        if not power_values:
            self.setup_topography_plot()
            return
        
        # Plot the power map
        self.plot_power_on_head(power_values)
    
    def plot_power_on_head(self, power_values, ax=None, colorbar_ax=None, theme_colors=None, show_labels=True):
        """
        Plot power values on electrode map with interpolation.
        
        Parameters:
        -----------
        power_values : dict
            Dictionary with channel names and power values
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses self.ax
        colorbar_ax : matplotlib.axes.Axes, optional
            Axes to plot colorbar on. If None, uses self.colorbar_ax
        theme_colors : dict, optional
            Theme colors to use. If None, uses self.theme_colors
        show_labels : bool, optional
            Whether to show electrode labels. Default is True.
        """
        # Use provided axes or default to widget axes
        target_ax = ax if ax is not None else self.ax
        target_cbar_ax = colorbar_ax if colorbar_ax is not None else self.colorbar_ax
        colors = theme_colors if theme_colors is not None else self.theme_colors
        
        # Clear the figures
        target_ax.clear()
        target_ax.set_facecolor(colors['bg_primary'])
        target_ax.set_aspect('equal')
        target_ax.axis('off')
        
        target_cbar_ax.clear()
        target_cbar_ax.set_facecolor(colors['bg_primary'])
        
        # Get electrode coordinates and powers
        electrode_positions = []
        electrode_powers = []
        electrode_channels = []
        
        for elec_num, coords in sorted(self.electrode_coords.items()):
            channel = coords['channel']
            if channel in power_values:
                electrode_positions.append([coords['x'], coords['y']])
                electrode_powers.append(power_values[channel])
                electrode_channels.append(channel)
        
        if not electrode_positions:
            target_ax.text(0.5, 0.5, 'No data available', 
                        transform=target_ax.transAxes,
                        ha='center', va='center', 
                        fontsize=12, color=colors['fg_primary'])
            if ax is None: # Only draw if using internal canvas
                self.canvas.draw()
            return

        electrode_positions = np.array(electrode_positions)
        electrode_powers = np.array(electrode_powers)
        
        # Normalize power for color mapping
        if self.fixed_scale_enabled and self.fixed_scale_min is not None and self.fixed_scale_max is not None:
            # Use fixed scale
            power_min, power_max = self.fixed_scale_min, self.fixed_scale_max
            norm = Normalize(vmin=power_min, vmax=power_max)
        else:
            # Auto scale based on data
            power_min, power_max = electrode_powers.min(), electrode_powers.max()
            if power_max > power_min:
                norm = Normalize(vmin=power_min, vmax=power_max)
            else:
                norm = Normalize(vmin=0, vmax=1)
        
        colormap = cm.get_cmap('RdYlBu_r')
        
        # Create interpolation grid
        grid_resolution = 150
        x_min, x_max = electrode_positions[:, 0].min() - 1, electrode_positions[:, 0].max() + 1
        y_min, y_max = electrode_positions[:, 1].min() - 1, electrode_positions[:, 1].max() + 1
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution)
        )
        
        # Interpolate power values
        grid_power = griddata(
            electrode_positions, 
            electrode_powers, 
            (grid_x, grid_y), 
            method='cubic'
        )
        
        # Fill NaN values
        grid_power_filled = griddata(
            electrode_positions, 
            electrode_powers, 
            (grid_x, grid_y), 
            method='nearest'
        )
        grid_power = np.where(np.isnan(grid_power), grid_power_filled, grid_power)
        
        # Apply boundary mask
        # If interpolation is ENABLED: use the full boundary of all electrodes (extrapolate to edges)
        # If interpolation is DISABLED: use only the boundary of ACTIVE electrodes (constrained view)
        active_channels_for_mask = None if self.interpolation_enabled else electrode_channels
        boundary_mask = self._create_boundary_mask(grid_x, grid_y, active_channels=active_channels_for_mask)
        
        grid_power = np.where(boundary_mask, grid_power, np.nan)
        
        # Always plot interpolated surface, but the mask controls visibility
        # Plot interpolated surface
        im = target_ax.pcolormesh(
            grid_x, grid_y, grid_power,
            cmap=colormap,
            norm=norm,
            shading='auto',
            zorder=1,
            alpha=0.9
        )
        
        # Add contour fill
        levels = np.linspace(power_min, power_max, 25)
        target_ax.contourf(
            grid_x, grid_y, grid_power, 
            levels=levels, 
            cmap=colormap, 
            norm=norm,
            extend='both',
            zorder=2,
            alpha=0.7
        )
        
        # Add contour lines
        target_ax.contour(
            grid_x, grid_y, grid_power,
            levels=8,
            colors='gray',
            alpha=0.4,
            linewidths=0.8,
            zorder=3
        )
        
        # Draw boundary line connecting outer electrodes
        # Same logic as mask: if interp enabled, use full boundary; else use active hull
        if self.interpolation_enabled:
            boundary_channels = self._get_boundary_electrodes()
        else:
             boundary_channels = self._compute_hull_from_channels(electrode_channels)

        if len(boundary_channels) >= 3:
            boundary_coords = []
            for ch in boundary_channels:
                for coords in self.electrode_coords.values():
                    if coords['channel'] == ch:
                        boundary_coords.append([coords['x'], coords['y']])
                        break
            
            if len(boundary_coords) >= 3:
                boundary_coords = np.array(boundary_coords)
                # Close the boundary by adding first point at the end
                boundary_coords_closed = np.vstack([boundary_coords, boundary_coords[0]])
                target_ax.plot(
                    boundary_coords_closed[:, 0],
                    boundary_coords_closed[:, 1],
                    'k-',
                    linewidth=2.5,
                    zorder=9,
                    alpha=0.8
                )
        
        # Plot electrodes
        for pos, power, channel in zip(electrode_positions, electrode_powers, electrode_channels):
            x, y = pos
            color = colormap(norm(power))
            
            # Plot electrode marker
            target_ax.scatter(x, y, s=200, c=[color], 
                          edgecolors='black', linewidth=2, zorder=10, alpha=1.0)
            
            # Add label
            if show_labels:
                target_ax.text(x, y, channel, 
                           ha='center', va='center', fontsize=9, fontweight='bold', 
                           color='white', zorder=11, 
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors['bg_primary'], alpha=0.7,
                                   edgecolor='white', linewidth=1))
        
        # Set limits and title
        margin = 1.0
        target_ax.set_xlim(x_min - margin, x_max + margin)
        target_ax.set_ylim(y_min - margin, y_max + margin)
        
        # Title
        title = f"{self.current_freq_band[0]}-{self.current_freq_band[1]} Hz Band Power"
        target_ax.set_title(title, fontsize=10, fontweight='bold', 
                         color=colors['fg_primary'], pad=5)
        
        if ax is None:
            self.figure.tight_layout()
            self.canvas.draw()
        
        # Create separate colorbar in colorbar figure
        cbar = self.colorbar_figure.colorbar(im, cax=target_cbar_ax)
        
        # Dynamic label based on calculation method
        unit_label = 'Power (μV²)' if self.calculation_method == 'sum' else 'Power (μV²/Hz)'
        cbar.set_label(unit_label, color=colors['fg_primary'], fontsize=9)
        
        cbar.ax.tick_params(colors=colors['fg_primary'], labelsize=8)
        cbar.outline.set_edgecolor(colors['fg_primary'])
        
        if ax is None:
            self.colorbar_figure.tight_layout()
            self.colorbar_canvas.draw()
    
    def clear_display(self):
        """Clear the topography display (used when disabled for performance)."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Show disabled message
        self.ax.text(0.5, 0.5, 'Topography Disabled\n(for faster labeling)', 
                    transform=self.ax.transAxes,
                    ha='center', va='center', 
                    fontsize=12, color=self.theme_colors['fg_primary'],
                    style='italic')
        
        # Clear colorbar
        self.colorbar_ax.clear()
        self.colorbar_ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()
        self.colorbar_figure.tight_layout()
        self.colorbar_canvas.draw()
    
    def set_epoch(self, epoch_idx):
        """Set the current epoch to display."""
        self.current_epoch = epoch_idx
        # Note: Actual data update should come from parent via set_data()
    
    def set_epoch_length(self, epoch_length):
        """Set the epoch length."""
        self.epoch_length = epoch_length