"""
Data visualization window for EEG Analysis GUI application.

This module contains the plotting window component with time navigation
and efficient data loading for large EEG datasets.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QGroupBox, QSplitter, QTextEdit,
                            QMessageBox, QFormLayout, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from pathlib import Path

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

try:
    import sys
    import os
    # Add utils directory to path for imports
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
    from utils import read_intan
    from utils import widget
    import pandas as pd
except ImportError as e:
    print(f"Import error in plotting_window: {e}")
    # Fallback imports
    import pandas as pd


class PlottingWindow(QWidget):
    """Window for visualizing EEG data with time navigation and efficient data loading."""
    
    def __init__(self, data_results):
        super().__init__()
        self.data_results = data_results
        self.current_start_time = 0.0
        self.current_duration = 10.0
        self.sample_rate = None
        self.total_duration = 0.0
        self.file_metadata = []  # Store metadata for each file
        self.cached_segments = {}  # Cache for loaded data segments
        self.global_sigma = None  # Global sigma calculated once for entire dataset
        self.sigma_calculated = False  # Flag to track if sigma has been calculated
        
        self.setup_ui()
        self.initialize_data_metadata()
        
    def setup_ui(self):
        """Set up the plotting window UI."""
        self.setWindowTitle("EEG Data Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for plotting
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Navigation controls
        nav_group = QGroupBox("Time Navigation")
        nav_layout = QFormLayout(nav_group)
        
        # Time input controls
        time_controls = QHBoxLayout()
        
        self.start_time_input = QLineEdit("0.0")
        self.start_time_input.setValidator(QDoubleValidator(0.0, 999999.0, 2))
        self.start_time_input.setMaximumWidth(80)
        time_controls.addWidget(QLabel("Start (s):"))
        time_controls.addWidget(self.start_time_input)
        
        self.duration_input = QLineEdit("10.0")
        self.duration_input.setValidator(QDoubleValidator(0.1, 1000.0, 2))
        self.duration_input.setMaximumWidth(80)
        time_controls.addWidget(QLabel("Duration (s):"))
        time_controls.addWidget(self.duration_input)
        
        nav_layout.addRow("Time Range:", time_controls)
        
        # Plotting style controls
        style_layout = QHBoxLayout()
        
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItems([
            "Standard Plot", 
            "Enhanced EEG Plot", 
            "Differential Signals",
            "Clinical Scale (7µV/mm)"
        ])
        self.plot_style_combo.currentTextChanged.connect(self.on_plot_style_changed)
        style_layout.addWidget(QLabel("Plot Style:"))
        style_layout.addWidget(self.plot_style_combo)
        
        nav_layout.addRow("Visualization:", style_layout)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        self.filter_enabled = QCheckBox("Enable Filter")
        self.filter_enabled.stateChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_enabled)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "Broadband (0.5-50Hz)",
            "Clinical (0.5-70Hz)", 
            "Alpha (8-13Hz)",
            "Beta (13-30Hz)",
            "Custom"
        ])
        self.filter_combo.setEnabled(False)
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        
        nav_layout.addRow("Filtering:", filter_layout)
        
        # EEG Display Parameters
        params_layout = QHBoxLayout()
        
        # Sigma multiplier control
        self.sigma_multiplier_input = QLineEdit("5.0")
        self.sigma_multiplier_input.setValidator(QDoubleValidator(0.1, 20.0, 1))
        self.sigma_multiplier_input.setMaximumWidth(60)
        self.sigma_multiplier_input.textChanged.connect(self.on_display_params_changed)
        self.sigma_multiplier_input.setToolTip("Channel spacing as multiple of signal standard deviation\n(1.0-3.0: tight spacing, 3.0-7.0: medium, 7.0+: wide spacing)")
        params_layout.addWidget(QLabel("Spacing (σ×):"))
        params_layout.addWidget(self.sigma_multiplier_input)
        
        # Y-range multiplier control
        self.y_range_multiplier_input = QLineEdit("5.0")
        self.y_range_multiplier_input.setValidator(QDoubleValidator(0.5, 10.0, 1))
        self.y_range_multiplier_input.setMaximumWidth(60)
        self.y_range_multiplier_input.textChanged.connect(self.on_display_params_changed)
        self.y_range_multiplier_input.setToolTip("Y-axis range as multiple of signal standard deviation\n(2.0-4.0: tight range, 4.0-6.0: medium, 6.0+: wide range)")
        params_layout.addWidget(QLabel("Y-Range (σ×):"))
        params_layout.addWidget(self.y_range_multiplier_input)
        
        # Reset button for display parameters
        reset_params_btn = QPushButton("Reset")
        reset_params_btn.setMaximumWidth(100)
        reset_params_btn.clicked.connect(self.reset_display_params)
        reset_params_btn.setToolTip("Reset to default values (5.0, 5.0)")
        params_layout.addWidget(reset_params_btn)
        
        # Recalculate sigma button
        recalc_sigma_btn = QPushButton("Recalc σ")
        recalc_sigma_btn.setMaximumWidth(100)
        recalc_sigma_btn.clicked.connect(self.recalculate_sigma)
        recalc_sigma_btn.setToolTip("Force recalculation of global sigma value")
        params_layout.addWidget(recalc_sigma_btn)
        
        nav_layout.addRow("Display:", params_layout)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.go_previous)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.go_next)
        button_layout.addWidget(self.next_button)
        
        self.update_button = QPushButton("🔄 Update View")
        self.update_button.clicked.connect(self.update_plot)
        button_layout.addWidget(self.update_button)
        
        nav_layout.addRow("Navigation:", button_layout)
        
        plot_layout.addWidget(nav_group)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        splitter.addWidget(plot_widget)
        
        # Right panel for information
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumWidth(300)
        info_layout.addWidget(self.info_text)
        
        splitter.addWidget(info_group)
        
        # Set splitter proportions (plot takes more space)
        splitter.setSizes([800, 300])
        main_layout.addWidget(splitter)
        
    def initialize_data_metadata(self):
        """Initialize metadata for all files without loading full data."""
        try:
            if not self.data_results:
                self.show_message("No data available", "No EEG data loaded.")
                return
                
            # Check if we have any files with data
            files_with_data = [(filename, result) for filename, result, data_present in self.data_results if data_present]
            
            if not files_with_data:
                self.show_message("No data", "No files contain EEG data.")
                return
            
            print(f"Initializing metadata for {len(files_with_data)} files...")
            
            # Process metadata for each file
            cumulative_time = 0.0
            for i, (filename, result) in enumerate(files_with_data):
                # Get sample rate (should be same for all files)
                if self.sample_rate is None:
                    self.sample_rate = read_intan.get_sample_rate(result)
                    if not self.sample_rate:
                        self.show_message("Error", "Could not determine sample rate.")
                        return
                
                # Get data dimensions without loading full data
                if 'amplifier_data' not in result:
                    continue
                    
                amplifier_data = result['amplifier_data']
                num_samples = amplifier_data.shape[1]
                num_channels = amplifier_data.shape[0]
                file_duration = num_samples / self.sample_rate
                
                # Store file metadata
                file_info = {
                    'filename': filename,
                    'result': result,  # Keep reference for later loading
                    'start_time': cumulative_time,
                    'end_time': cumulative_time + file_duration,
                    'duration': file_duration,
                    'num_samples': num_samples,
                    'num_channels': num_channels,
                    'sample_indices': (0, num_samples)
                }
                
                self.file_metadata.append(file_info)
                cumulative_time += file_duration
                
                print(f"File {i+1}: {filename}, Duration: {file_duration:.2f}s")
            
            self.total_duration = cumulative_time
            print(f"Total dataset duration: {self.total_duration:.2f}s")
            
            # Update UI with metadata
            self.update_info_display()
            
            # Load initial segment
            self.load_time_segment(self.current_start_time, self.current_duration)
            
        except Exception as e:
            print(f"Error initializing data metadata: {e}")
            self.show_message("Error", f"Error processing data: {str(e)}")
    
    def calculate_global_sigma_once(self):
        """Calculate global sigma once for the entire dataset using robust statistical methods."""
        if self.sigma_calculated and self.global_sigma is not None:
            return self.global_sigma
        
        print("Calculating global sigma for entire dataset...")
        channel_sigmas = []  # Collect sigma from each channel separately
        
        # Sample from multiple segments across the dataset for statistical accuracy
        sample_duration = 2.0  # Sample 2 seconds from each file (shorter for better performance)
        max_samples_per_channel = 5000  # Limit samples to avoid memory issues
        
        for file_info in self.file_metadata:
            try:
                # Sample from the middle of each file
                mid_time = (file_info['start_time'] + file_info['end_time']) / 2
                sample_start = max(file_info['start_time'], mid_time - sample_duration / 2)
                sample_end = min(file_info['end_time'], sample_start + sample_duration)
                
                # Convert to file-relative time and sample indices
                relative_start = sample_start - file_info['start_time']
                relative_end = sample_end - file_info['start_time']
                start_sample = int(relative_start * self.sample_rate)
                end_sample = int(relative_end * self.sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(file_info['num_samples'], end_sample)
                
                if start_sample >= end_sample:
                    continue
                
                # Load sample data
                result = file_info['result']
                amplifier_data = result['amplifier_data'][:, start_sample:end_sample]
                
                # Calculate sigma for each channel separately using robust statistics
                for channel_idx in range(amplifier_data.shape[0]):
                    channel_data = amplifier_data[channel_idx, :].flatten()
                    
                    # Remove DC offset (subtract mean)
                    channel_data = channel_data - np.mean(channel_data)
                    
                    # Remove outliers using IQR method (more robust than simple std)
                    q75, q25 = np.percentile(channel_data, [75, 25])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    # Filter out outliers
                    filtered_data = channel_data[(channel_data >= lower_bound) & (channel_data <= upper_bound)]
                    
                    # Subsample if too many points
                    if len(filtered_data) > max_samples_per_channel:
                        step = len(filtered_data) // max_samples_per_channel
                        filtered_data = filtered_data[::step]
                    
                    if len(filtered_data) > 10:  # Need minimum samples for reliable sigma
                        channel_sigma = np.std(filtered_data)
                        if 1.0 <= channel_sigma <= 1000.0:  # Reasonable range for EEG (1-1000 μV)
                            channel_sigmas.append(channel_sigma)
                    
            except Exception as e:
                print(f"Warning: Could not sample from {file_info['filename']}: {e}")
                continue
        
        # Calculate global sigma using median of channel sigmas (more robust than mean)
        if channel_sigmas:
            self.global_sigma = np.median(channel_sigmas)
            self.sigma_calculated = True
            print(f"Global sigma calculated: {self.global_sigma:.2f} μV (from {len(channel_sigmas)} channel samples)")
            print(f"Channel sigma range: {np.min(channel_sigmas):.1f} - {np.max(channel_sigmas):.1f} μV")
        else:
            self.global_sigma = 50.0  # More reasonable fallback value for EEG
            self.sigma_calculated = True
            print("Warning: Could not calculate global sigma, using fallback value: 50.0 μV")
        
        return self.global_sigma
        
        return self.global_sigma
    
    def load_time_segment(self, start_time, duration):
        """Load only the requested time segment for efficient visualization."""
        try:
            end_time = start_time + duration
            
            # Check cache first
            cache_key = f"{start_time:.3f}_{end_time:.3f}"
            if cache_key in self.cached_segments:
                print(f"Loading from cache: {start_time:.2f}s - {end_time:.2f}s")
                channel_data = self.cached_segments[cache_key]
                # Ensure global sigma is calculated before plotting
                self.calculate_global_sigma_once()
                self.plot_data(channel_data)
                return
            
            print(f"Loading time segment: {start_time:.2f}s - {end_time:.2f}s")
            
            # Find which files contain the requested time range
            relevant_files = []
            for file_info in self.file_metadata:
                if (file_info['start_time'] < end_time and 
                    file_info['end_time'] > start_time):
                    relevant_files.append(file_info)
            
            if not relevant_files:
                self.show_message("No data", f"No data available for time range {start_time:.2f}s - {end_time:.2f}s")
                return
            
            # Load and concatenate data from relevant files
            all_channel_data = {}
            concatenated_time = []
            
            for file_info in relevant_files:
                # Calculate which part of this file we need
                file_start = max(start_time, file_info['start_time'])
                file_end = min(end_time, file_info['end_time'])
                
                # Convert to sample indices within this file
                relative_start = file_start - file_info['start_time']
                relative_end = file_end - file_info['start_time']
                
                start_sample = int(relative_start * self.sample_rate)
                end_sample = int(relative_end * self.sample_rate)
                
                # Ensure we don't go beyond file boundaries
                start_sample = max(0, start_sample)
                end_sample = min(file_info['num_samples'], end_sample)
                
                if start_sample >= end_sample:
                    continue
                
                print(f"  Loading from {file_info['filename']}: samples {start_sample}-{end_sample}")
                
                # Load only the required data segment
                result = file_info['result']
                amplifier_data = result['amplifier_data'][:, start_sample:end_sample]
                amplifier_channels = result.get('amplifier_channels', [])
                
                # Create time array for this segment
                num_samples = end_sample - start_sample
                segment_time = np.arange(num_samples, dtype=np.float64) / self.sample_rate + file_start
                concatenated_time.extend(segment_time)
                
                # Extract channel data
                for i, channel_info in enumerate(amplifier_channels):
                    if i < amplifier_data.shape[0]:
                        channel_name = (channel_info.get('custom_channel_name') or 
                                      channel_info.get('native_channel_name', f'Channel_{i}'))
                        
                        if channel_name not in all_channel_data:
                            all_channel_data[channel_name] = []
                        
                        all_channel_data[channel_name].extend(amplifier_data[i, :])
            
            # Combine all channel data
            channel_data = {'time': np.array(concatenated_time)}
            for channel_name, data_list in all_channel_data.items():
                channel_data[channel_name] = np.array(data_list)
            
            # Cache the loaded segment (limit cache size)
            if len(self.cached_segments) > 10:  # Keep only 10 segments cached
                # Remove oldest cache entry
                oldest_key = min(self.cached_segments.keys())
                del self.cached_segments[oldest_key]
            
            self.cached_segments[cache_key] = channel_data
            
            # Ensure global sigma is calculated before plotting
            self.calculate_global_sigma_once()
            
            # Plot the data
            self.plot_data(channel_data)
            
        except Exception as e:
            print(f"Error loading time segment: {e}")
            self.show_message("Error", f"Error loading data segment: {str(e)}")
    
    def plot_data(self, channel_data):
        """Plot the loaded channel data using enhanced plotting functions."""
        try:
            if not channel_data or 'time' not in channel_data:
                return
            
            # Convert to DataFrame for compatibility with widget functions
            df_data = pd.DataFrame(channel_data)
            
            # Apply filtering if enabled
            if hasattr(self, 'filter_enabled') and self.filter_enabled.isChecked():
                df_data = self.apply_current_filter(df_data)
            
            # Get user-selected display parameters
            try:
                sigma_multiplier = float(self.sigma_multiplier_input.text()) if hasattr(self, 'sigma_multiplier_input') else 5.0
                y_range_multiplier = float(self.y_range_multiplier_input.text()) if hasattr(self, 'y_range_multiplier_input') else 5.0
            except ValueError:
                sigma_multiplier = 5.0
                y_range_multiplier = 5.0
            
            # Get plot style
            plot_style = self.plot_style_combo.currentText() if hasattr(self, 'plot_style_combo') else "Standard Plot"
            
            # Use appropriate plotting function from widget module
            if plot_style == "Enhanced EEG Plot":
                widget.plot_enhanced_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
            elif plot_style == "Differential Signals":
                widget.plot_differential_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
            elif plot_style == "Clinical Scale (7µV/mm)":
                widget.plot_clinical_scale_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
            else:
                # Standard plot with user-configurable parameters
                widget.plot_standard_eeg_data(self.figure, df_data, self.cached_segments,
                                            sigma_multiplier=sigma_multiplier,
                                            y_range_multiplier=y_range_multiplier,
                                            global_sigma=self.global_sigma)
            
            # Adjust layout and refresh
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update navigation controls
            self.update_navigation_controls()
            
        except Exception as e:
            print(f"Error plotting data: {e}")
            self.show_message("Error", f"Error plotting data: {str(e)}")
    
    def plot_standard_eeg(self, df_data):
        """Plot standard EEG view - simplified wrapper."""
        try:
            sigma_multiplier = float(self.sigma_multiplier_input.text()) if hasattr(self, 'sigma_multiplier_input') else 5.0
            y_range_multiplier = float(self.y_range_multiplier_input.text()) if hasattr(self, 'y_range_multiplier_input') else 5.0
        except ValueError:
            sigma_multiplier = 5.0
            y_range_multiplier = 5.0
        
        widget.plot_standard_eeg_data(self.figure, df_data, self.cached_segments,
                                    sigma_multiplier=sigma_multiplier,
                                    y_range_multiplier=y_range_multiplier,
                                    global_sigma=self.global_sigma)
    
    def plot_enhanced_eeg(self, df_data):
        """Plot enhanced EEG view - simplified wrapper."""
        widget.plot_enhanced_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
    
    def plot_differential_signals(self, df_data):
        """Plot differential signals - simplified wrapper."""
        widget.plot_differential_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
    
    def plot_clinical_scale(self, df_data):
        """Plot with clinical scale - simplified wrapper."""
        widget.plot_clinical_scale_eeg_data(self.figure, df_data, self.cached_segments, global_sigma=self.global_sigma)
    
    def on_time_changed(self):
        """Handle changes in time input fields."""
        try:
            self.current_start_time = float(self.start_time_input.text() or "0.0")
            self.current_duration = float(self.duration_input.text() or "10.0")
            self.update_info_display()
        except ValueError:
            pass  # Invalid input, ignore
            
    def go_previous(self):
        """Navigate to previous time segment."""
        new_start = self.current_start_time - self.current_duration
        if new_start < 0:
            new_start = 0
        self.current_start_time = new_start
        self.start_time_input.setText(f"{new_start:.2f}")
        self.load_time_segment(self.current_start_time, self.current_duration)
        
    def go_next(self):
        """Navigate to next time segment."""
        new_start = self.current_start_time + self.current_duration
        
        # Check against total duration instead of df_data
        if new_start + self.current_duration > self.total_duration:
            new_start = max(0, self.total_duration - self.current_duration)
            
        self.current_start_time = new_start
        self.start_time_input.setText(f"{new_start:.2f}")
        self.load_time_segment(self.current_start_time, self.current_duration)
        
    def update_plot(self):
        """Update the plot with current time range."""
        try:
            # Get current time parameters
            self.on_time_changed()
            
            # Load and plot the new time segment
            self.load_time_segment(self.current_start_time, self.current_duration)
            
        except Exception as e:
            print(f"Error updating plot: {e}")
            self.show_message("Error", f"Error updating plot: {str(e)}")
    
    def update_info_display(self):
        """Update the information display panel."""
        if not self.file_metadata:
            self.info_text.setPlainText("No data loaded.")
            return
        
        total_files = len(self.file_metadata)
        total_channels = self.file_metadata[0]['num_channels'] if self.file_metadata else 0
        
        # Get sigma info
        sigma_info = f"σ={self.global_sigma:.1f}μV" if self.global_sigma else "σ=calculating..."
        
        info_lines = [
            f"Dataset Information:",
            f"═══════════════════",
            f"Total Duration: {self.total_duration:.2f} seconds",
            f"Files: {total_files}",
            f"Channels: {total_channels}",
            f"Sample Rate: {self.sample_rate} Hz",
            f"Global Sigma: {sigma_info}",
            "",
            f"Current View:",
            f"Start Time: {self.current_start_time:.2f}s",
            f"Duration: {self.current_duration:.2f}s", 
            f"End Time: {self.current_start_time + self.current_duration:.2f}s",
            "",
            f"Navigation:",
            f"Use ← → buttons or enter custom times",
            f"Cache: {len(self.cached_segments)} segments loaded"
        ]
        
        self.info_text.setPlainText("\n".join(info_lines))
    
    def update_navigation_controls(self):
        """Update the navigation control inputs."""
        self.start_time_input.setText(f"{self.current_start_time:.2f}")
        self.duration_input.setText(f"{self.current_duration:.2f}")
            
    def show_message(self, title, message):
        """Show a message box."""
        QMessageBox.information(self, title, message)
    
    def apply_current_filter(self, df_data):
        """Apply the currently selected filter to the data."""
        if not hasattr(self, 'filter_combo'):
            return df_data
        
        filter_type = self.filter_combo.currentText()
        
        if filter_type == "No Filter":
            return df_data
        
        # Get sampling rate (estimate from time data)
        time_diff = df_data['time'].diff().dropna()
        sampling_rate = 1.0 / time_diff.mean() if len(time_diff) > 0 else 1000.0
        
        # Apply filter based on type
        filtered_data = df_data.copy()
        
        try:
            from scipy import signal
            
            # Define filter parameters
            if filter_type == "Alpha (8-13 Hz)":
                low_freq, high_freq = 8, 13
            elif filter_type == "Beta (13-30 Hz)":
                low_freq, high_freq = 13, 30
            elif filter_type == "Gamma (30-100 Hz)":
                low_freq, high_freq = 30, 100
            elif filter_type == "Delta (0.5-4 Hz)":
                low_freq, high_freq = 0.5, 4
            elif filter_type == "Theta (4-8 Hz)":
                low_freq, high_freq = 4, 8
            else:
                return df_data
            
            # Design bandpass filter
            nyquist = sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if high >= 1.0:  # Avoid filter design issues
                high = 0.99
            
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter to each channel (except time)
            channels = [ch for ch in df_data.columns if ch != 'time']
            for channel in channels:
                filtered_data[channel] = signal.filtfilt(b, a, df_data[channel])
            
        except ImportError:
            print("SciPy not available for filtering")
        except Exception as e:
            print(f"Error applying filter: {e}")
        
        return filtered_data
    
    def on_plot_style_changed(self):
        """Handle plot style change."""
        if hasattr(self, 'data_metadata') and self.data_metadata:
            # Reload current time segment with new plot style
            self.load_time_segment(self.current_start_time, self.current_duration)
    
    def on_display_params_changed(self):
        """Handle display parameter changes (sigma multiplier, y-range multiplier)."""
        if hasattr(self, 'data_metadata') and self.data_metadata:
            # Reload current time segment with new display parameters
            self.load_time_segment(self.current_start_time, self.current_duration)
    
    def reset_display_params(self):
        """Reset display parameters to default values."""
        self.sigma_multiplier_input.setText("5.0")
        self.y_range_multiplier_input.setText("5.0")
        # The textChanged signal will automatically trigger a plot refresh
    
    def recalculate_sigma(self):
        """Force recalculation of global sigma."""
        self.global_sigma = None
        self.sigma_calculated = False
        print("Forcing sigma recalculation...")
        self.calculate_global_sigma_once()
        self.update_info_display()
        # Refresh current plot with new sigma
        if hasattr(self, 'file_metadata') and self.file_metadata:
            self.load_time_segment(self.current_start_time, self.current_duration)
    
    def on_filter_changed(self):
        """Handle filter change."""
        if hasattr(self, 'data_metadata') and self.data_metadata:
            # Reload current time segment with new filter
            self.load_time_segment(self.current_start_time, self.current_duration)
