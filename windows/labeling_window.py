"""
Labeling window for MEEG Analysis GUI application.

This module contains the labeling window that allows users to
visualize and label mosaic data (differential signals between electrode pairs).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QGroupBox, QSplitter, QTextEdit,
                            QMessageBox, QFormLayout, QLineEdit, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QFont, QShortcut, QKeySequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

try:
    from utils import read_intan
    from utils import widget
except ImportError as e:
    print(f"Import error in labeling_window: {e}")


class MosaicVisualizationWidget(QWidget):
    """Widget that displays electrode figure and standard EEG plot."""
    
    def __init__(self, mosaic_relationships, electrode_positions, current_data=None, channel_mapping=None):
        super().__init__()
        self.mosaic_relationships = mosaic_relationships or []
        self.electrode_positions = electrode_positions or []
        self.current_data = current_data
        self.channel_mapping = channel_mapping or {}
        self.image_path = str(Path(__file__).parent.parent / "img" / "Mouse_head_2.png")
        
        # Coordinate system parameters
        self.y_min = -8
        self.y_max = 5
        self.x_min = -5
        self.x_max = 5
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the configuration display with electrode positions and mosaic relationships."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Electrode & Mosaic Configuration")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Create matplotlib figure for electrode positions
        self.electrode_figure = Figure(figsize=(4, 4))
        self.electrode_canvas = FigureCanvas(self.electrode_figure)
        self.electrode_canvas.setMaximumHeight(300)
        layout.addWidget(self.electrode_canvas)
        
        self.update_electrode_display()
        
    def update_electrode_display(self):
        """Update the electrode position display with mosaic relationships."""
        self.electrode_figure.clear()
        ax = self.electrode_figure.add_subplot(111)
        
        try:
            # Load and display the mouse head image
            import matplotlib.pyplot as plt
            img = plt.imread(self.image_path)
            ax.imshow(img, extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                     aspect='equal', origin='upper')
        except Exception as e:
            print(f"Warning: Could not load image {self.image_path}: {e}")
        
        # Set labels and limits
        ax.set_xlabel('X (ML)', fontsize=8)
        ax.set_ylabel('Y (AP)', fontsize=8)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.tick_params(labelsize=7)
        
        # Plot electrodes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, electrode in enumerate(self.electrode_positions):
            x, y = electrode['x'], electrode['y']
            number = electrode['number']
            color = colors[i % len(colors)]
            
            # Draw electrode circle
            import matplotlib.patches as patches
            circle = patches.Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            
            # Add electrode number
            ax.text(x, y, str(number), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # Draw mosaic relationships as lines
        for rel in self.mosaic_relationships:
            electrode_a = rel.get('electrode_a')
            electrode_b = rel.get('electrode_b')
            
            # Find electrode positions
            pos_a = next((e for e in self.electrode_positions if e['number'] == electrode_a), None)
            pos_b = next((e for e in self.electrode_positions if e['number'] == electrode_b), None)
            
            if pos_a and pos_b:
                x1, y1 = pos_a['x'], pos_a['y']
                x2, y2 = pos_b['x'], pos_b['y']
                
                ax.plot([x1, x2], [y1, y2], color='blue', linewidth=1.5, alpha=0.7)
                
                # Add relationship name at midpoint
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, rel.get('name', ''), ha='center', va='center', 
                       fontsize=6, bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        
        ax.set_title('Electrode Layout', fontsize=10)
        self.electrode_figure.tight_layout()
        self.electrode_canvas.draw()
        
    def update_data(self, current_data, channel_mapping):
        """Update the data and channel mapping."""
        self.current_data = current_data
        self.channel_mapping = channel_mapping
        self.update_electrode_display()


class LabelingWindow(QWidget):
    """Window for labeling mosaic data analysis."""
    
    def __init__(self, mosaic_relationships, electrode_positions, current_data, channel_mapping, parent=None):
        super().__init__(parent)
        self.mosaic_relationships = mosaic_relationships
        self.electrode_positions = electrode_positions
        self.current_data = current_data
        self.channel_mapping = channel_mapping
        
        # Time navigation parameters
        self.current_start_time = 0.0
        self.current_epoch_length = 10.0  # Default epoch length
        self.sample_rate = None
        self.total_duration = 0.0
        self.file_metadata = []
        self.cached_segments = {}
        self.mosaic_data_cache = {}
        self.global_sigma = None  # Fixed global sigma for consistent scaling
        
        self.setup_ui()
        self.initialize_data()
        
    def setup_ui(self):
        """Set up the labeling window UI."""
        self.setWindowTitle("Mosaic Data Labeling")
        self.setGeometry(150, 150, 1400, 900)
        
        # Set window flags to ensure it's movable and resizable
        self.setWindowFlags(Qt.WindowType.Window | 
                           Qt.WindowType.WindowMinimizeButtonHint | 
                           Qt.WindowType.WindowMaximizeButtonHint | 
                           Qt.WindowType.WindowCloseButtonHint)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        # Mosaic configuration widget
        self.mosaic_config = MosaicVisualizationWidget(
            self.mosaic_relationships, 
            self.electrode_positions,
            self.current_data,
            self.channel_mapping
        )
        left_layout.addWidget(self.mosaic_config)
        
        # Time navigation controls
        nav_group = QGroupBox("Time Navigation")
        nav_layout = QFormLayout(nav_group)
        
        # Starting time input
        time_controls = QHBoxLayout()
        self.start_time_input = QLineEdit("0.0")
        self.start_time_input.setValidator(QDoubleValidator(0.0, 999999.0, 3))
        self.start_time_input.setMaximumWidth(100)
        time_controls.addWidget(QLabel("Starting Time (s):"))
        time_controls.addWidget(self.start_time_input)
        nav_layout.addRow("Start Time:", time_controls)
        
        # Epoch length input
        epoch_controls = QHBoxLayout()
        self.epoch_length_input = QLineEdit("10.0")
        self.epoch_length_input.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.epoch_length_input.setMaximumWidth(100)
        epoch_controls.addWidget(QLabel("Epoch Length (s):"))
        epoch_controls.addWidget(self.epoch_length_input)
        nav_layout.addRow("Epoch Length:", epoch_controls)
        
        # Mosaic selection
        mosaic_controls = QHBoxLayout()
        self.mosaic_combo = QComboBox()
        self.populate_mosaic_combo()
        mosaic_controls.addWidget(QLabel("Mosaic Pair:"))
        mosaic_controls.addWidget(self.mosaic_combo)
        nav_layout.addRow("Active Mosaic:", mosaic_controls)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.go_previous_epoch)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.go_next_epoch)
        button_layout.addWidget(self.next_button)
        
        self.update_button = QPushButton("🔄 Update")
        self.update_button.clicked.connect(self.update_mosaic_plot)
        button_layout.addWidget(self.update_button)
        
        self.recalc_sigma_button = QPushButton("📏 Recalc Sigma")
        self.recalc_sigma_button.clicked.connect(self.recalculate_global_sigma)
        self.recalc_sigma_button.setToolTip("Recalculate global sigma for consistent scaling")
        button_layout.addWidget(self.recalc_sigma_button)
        
        nav_layout.addRow("Navigation:", button_layout)
        
        left_layout.addWidget(nav_group)
        
        # Plot scaling controls
        scaling_group = QGroupBox("Plot Scaling (σ multipliers)")
        scaling_layout = QFormLayout(scaling_group)
        
        # Channel spacing control
        spacing_controls = QHBoxLayout()
        self.spacing_multiplier_input = QLineEdit("5")
        self.spacing_multiplier_input.setValidator(QDoubleValidator(0.1, 50.0, 1))
        self.spacing_multiplier_input.setMaximumWidth(80)
        self.spacing_multiplier_input.setToolTip("Vertical spacing between channels (in sigma units)")
        spacing_controls.addWidget(self.spacing_multiplier_input)
        spacing_controls.addWidget(QLabel("σ"))
        scaling_layout.addRow("Channel Spacing:", spacing_controls)
        
        # Y-axis range control
        ylim_controls = QHBoxLayout()
        self.ylim_multiplier_input = QLineEdit("5")
        self.ylim_multiplier_input.setValidator(QDoubleValidator(0.1, 50.0, 1))
        self.ylim_multiplier_input.setMaximumWidth(80)
        self.ylim_multiplier_input.setToolTip("Total height of the plot (in sigma units)")
        ylim_controls.addWidget(self.ylim_multiplier_input)
        ylim_controls.addWidget(QLabel("σ"))
        scaling_layout.addRow("Y-axis Range:", ylim_controls)
        
        # Apply scaling button
        apply_scaling_button = QPushButton("Apply Scaling")
        apply_scaling_button.clicked.connect(self.update_mosaic_plot)
        apply_scaling_button.setToolTip("Apply the new scaling parameters to the plot")
        scaling_layout.addRow("", apply_scaling_button)
        
        left_layout.addWidget(scaling_group)
        
        # Data information
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        left_layout.addWidget(info_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel for plotting
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Plot title
        plot_title = QLabel("Mosaic Data Visualization")
        plot_title_font = QFont()
        plot_title_font.setBold(True)
        plot_title_font.setPointSize(14)
        plot_title.setFont(plot_title_font)
        plot_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_layout.addWidget(plot_title)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        splitter.addWidget(plot_widget)
        
        # Set splitter proportions
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for navigation and scaling."""
        # Left Arrow for Previous
        self.prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.prev_shortcut.activated.connect(self.go_previous_epoch)
        
        # Right Arrow for Next
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.next_shortcut.activated.connect(self.go_next_epoch)
        
        # Plus key to increase spacing
        self.increase_spacing_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Plus), self)
        self.increase_spacing_shortcut.activated.connect(self.increase_spacing)
        
        # Minus key to decrease spacing
        self.decrease_spacing_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Minus), self)
        self.decrease_spacing_shortcut.activated.connect(self.decrease_spacing)
        
    def increase_spacing(self):
        """Increase channel spacing by 0.5 sigma."""
        try:
            current_value = float(self.spacing_multiplier_input.text() or "5")
            new_value = min(current_value + 0.5, 50.0)  # Cap at 50
            self.spacing_multiplier_input.setText(f"{new_value:.1f}")
            self.update_mosaic_plot()
        except ValueError:
            pass
    
    def decrease_spacing(self):
        """Decrease channel spacing by 0.5 sigma."""
        try:
            current_value = float(self.spacing_multiplier_input.text() or "5")
            new_value = max(current_value - 0.5, 0.1)  # Minimum 0.1
            self.spacing_multiplier_input.setText(f"{new_value:.1f}")
            self.update_mosaic_plot()
        except ValueError:
            pass
        
    def populate_mosaic_combo(self):
        """Populate the mosaic selection combo box."""
        self.mosaic_combo.clear()
        for rel in self.mosaic_relationships:
            display_text = f"{rel['name']} (E{rel['electrode_a']} - E{rel['electrode_b']})"
            self.mosaic_combo.addItem(display_text)
            
    def initialize_data(self):
        """Initialize data parameters from cached data."""
        try:
            if not self.current_data:
                self.show_message("No Data", "No data available for analysis.")
                return
                
            # Calculate total duration and set sample rate from current_data
            self.total_duration = 0.0
            
            for item in self.current_data:
                if len(item) >= 2:
                    filename, result = item[0], item[1]
                    
                    if 'amplifier_data' not in result:
                        continue
                        
                    # Get sample rate from first file
                    if self.sample_rate is None:
                        freq_params = result.get('frequency_parameters', {})
                        self.sample_rate = freq_params.get('amplifier_sample_rate', 30000)
                    
                    # Calculate file duration
                    amplifier_data = result['amplifier_data']
                    num_samples = amplifier_data.shape[1]
                    file_duration = num_samples / self.sample_rate
                    
                    self.total_duration += file_duration
            
            if self.total_duration == 0:
                self.show_message("No Data", "No valid data found for analysis.")
                return
            
            print(f"Initialized data: {len(self.current_data)} files, {self.total_duration:.2f}s total, {self.sample_rate}Hz")
            
            # Update displays
            self.update_info_display()
            self.calculate_global_sigma()  # Calculate fixed global sigma
            self.update_mosaic_plot()
            
        except Exception as e:
            print(f"Error initializing labeling window data: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Error", f"Error processing data: {str(e)}")
    
    def calculate_global_sigma(self):
        """Calculate global sigma from all available mosaic data for consistent scaling."""
        try:
            print("Calculating global sigma from all available data...")
            
            # Use a larger time window to sample more data for statistics
            sample_duration = min(10.0, self.total_duration)  # Sample up to 10 seconds
            sample_start = 0.0
            sample_end = sample_duration
            
            all_mosaic_values = []
            
            # Calculate mosaic data for all relationships over the sample period
            for relationship in self.mosaic_relationships:
                time_array, mosaic_data = self.calculate_mosaic_data(relationship, sample_start, sample_end)
                if time_array is not None and mosaic_data is not None:
                    all_mosaic_values.extend(mosaic_data)
            
            if all_mosaic_values:
                self.global_sigma = np.std(all_mosaic_values)
                print(f"Global sigma calculated: {self.global_sigma:.2f} μV")
            else:
                self.global_sigma = 100.0  # Default fallback value
                print("No mosaic data available for global sigma calculation, using default: 100.0 μV")
                
        except Exception as e:
            print(f"Error calculating global sigma: {e}")
            self.global_sigma = 100.0  # Default fallback value
    
    def recalculate_global_sigma(self):
        """Recalculate global sigma and update the plot."""
        self.calculate_global_sigma()
        self.update_mosaic_plot()
        self.update_info_display()
        print(f"Global sigma recalculated: {self.global_sigma:.2f} μV")
    
    def get_channel_data_for_electrode(self, electrode_number, start_time, end_time):
        """Get channel data for a specific electrode within time range from cached data."""
        try:
            print(f"Debug: Getting data for electrode {electrode_number}, time {start_time}-{end_time}")
            
            # Find the channel mapping for this electrode
            if electrode_number not in self.channel_mapping:
                print(f"Warning: Electrode {electrode_number} not mapped to any channel")
                print(f"Available mappings: {self.channel_mapping}")
                return None, None
                
            # Get the channel name from the mapping
            channel_name = self.channel_mapping[electrode_number]
            print(f"Debug: Electrode {electrode_number} maps to channel {channel_name}")
            
            if not self.current_data:
                print("Warning: No current data available")
                return None, None
            
            print(f"Debug: Processing {len(self.current_data)} data files")
            
            # Use sample rate from initialization or default
            sample_rate = self.sample_rate or 30000
            
            # Convert time to sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Collect data from all files
            all_data = []
            current_time_offset = 0
            
            for i, item in enumerate(self.current_data):
                if len(item) >= 2:
                    filename, result = item[0], item[1]
                    
                    if 'amplifier_data' not in result:
                        print(f"Debug: File {i} ({filename}) has no amplifier_data")
                        continue
                        
                    amplifier_data = result['amplifier_data']
                    file_sample_rate = result.get('frequency_parameters', {}).get('amplifier_sample_rate', 30000)
                    amplifier_channels = result.get('amplifier_channels', [])
                    
                    print(f"Debug: File {i} - shape: {amplifier_data.shape}, sample_rate: {file_sample_rate}")
                    
                    # Find the channel index by name using amplifier_channels metadata
                    channel_index = None
                    for ch_idx, ch_info in enumerate(amplifier_channels):
                        ch_name = (ch_info.get('custom_channel_name') or 
                                  ch_info.get('native_channel_name', f'Channel_{ch_idx}'))
                        if ch_name == channel_name:
                            channel_index = ch_idx
                            break
                    
                    if channel_index is None:
                        print(f"Warning: Channel {channel_name} not found in file {filename}")
                        print(f"Available channels: {[ch.get('custom_channel_name') or ch.get('native_channel_name', f'Channel_{i}') for i, ch in enumerate(amplifier_channels)]}")
                        continue
                        
                    if channel_index >= amplifier_data.shape[0]:
                        print(f"Warning: Channel index {channel_index} exceeds data shape in file {filename}")
                        continue
                    
                    # Get the number of samples in this file
                    file_num_samples = amplifier_data.shape[1]
                    file_duration = file_num_samples / file_sample_rate
                    
                    print(f"Debug: File {i} duration: {file_duration:.3f}s, time offset: {current_time_offset:.3f}s")
                    
                    # Calculate which part of this file overlaps with our time range
                    file_start_time = current_time_offset
                    file_end_time = current_time_offset + file_duration
                    
                    # Check if this file overlaps with our requested time range
                    if file_end_time > start_time and file_start_time < end_time:
                        # Calculate the sample range we need from this file
                        relative_start_time = max(0, start_time - file_start_time)
                        relative_end_time = min(file_duration, end_time - file_start_time)
                        
                        file_start_sample = int(relative_start_time * file_sample_rate)
                        file_end_sample = int(relative_end_time * file_sample_rate)
                        
                        # Clamp to file bounds
                        file_start_sample = max(0, file_start_sample)
                        file_end_sample = min(file_num_samples, file_end_sample)
                        
                        print(f"Debug: Extracting samples {file_start_sample}-{file_end_sample} from file {i}")
                        
                        if file_start_sample < file_end_sample:
                            # Extract the channel data for this segment
                            channel_data = amplifier_data[channel_index, file_start_sample:file_end_sample]
                            all_data.extend(channel_data)
                            print(f"Debug: Added {len(channel_data)} samples from file {i}")
                    
                    # Update time offset for next file
                    current_time_offset += file_duration
            
            print(f"Debug: Total extracted data length: {len(all_data)}")
            
            if not all_data:
                print(f"Warning: No data found for electrode {electrode_number} in time range {start_time}-{end_time}")
                return None, None
            
            # Create time array
            time_array = np.linspace(start_time, end_time, len(all_data))
            
            return time_array, np.array(all_data)
            
        except Exception as e:
            print(f"Error getting channel data for electrode {electrode_number}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def calculate_mosaic_data(self, relationship, start_time, end_time):
        """Calculate mosaic data (working electrode - reference electrode)."""
        try:
            electrode_a = relationship['electrode_a']
            electrode_b = relationship['electrode_b']
            
            print(f"Debug: Calculating mosaic for E{electrode_a} - E{electrode_b}")
            
            # Get data for both electrodes
            time_a, data_a = self.get_channel_data_for_electrode(electrode_a, start_time, end_time)
            time_b, data_b = self.get_channel_data_for_electrode(electrode_b, start_time, end_time)
            
            print(f"Debug: Electrode {electrode_a} data: {type(data_a)}, length: {len(data_a) if data_a is not None else 'None'}")
            print(f"Debug: Electrode {electrode_b} data: {type(data_b)}, length: {len(data_b) if data_b is not None else 'None'}")
            
            if time_a is None or time_b is None:
                print(f"Debug: Missing data - time_a: {time_a is not None}, time_b: {time_b is not None}")
                return None, None
            
            # Ensure both arrays have the same length (take minimum)
            min_length = min(len(data_a), len(data_b))
            if min_length == 0:
                print(f"Debug: No data available - min_length: {min_length}")
                return None, None
                
            time_array = time_a[:min_length]
            mosaic_data = data_a[:min_length] - data_b[:min_length]
            
            print(f"Debug: Successfully calculated mosaic data, length: {len(mosaic_data)}")
            return time_array, mosaic_data
            
        except Exception as e:
            print(f"Error calculating mosaic data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def update_mosaic_plot(self):
        """Update the mosaic data plot showing all relationships."""
        try:
            # Get current parameters
            start_time = float(self.start_time_input.text() or "0.0")
            epoch_length = float(self.epoch_length_input.text() or "10.0")
            
            # Calculate time range with boundary checks
            plot_start = max(0, start_time - epoch_length)  # Don't go below 0
            plot_end = start_time + epoch_length
            
            # Clamp plot_end to available data duration if we know it
            if self.total_duration > 0:
                plot_end = min(plot_end, self.total_duration)
            
            # Ensure valid time range
            if plot_start >= plot_end:
                # Adjust if start time is near the end of data
                if start_time >= self.total_duration - epoch_length:
                    plot_end = self.total_duration
                    plot_start = max(0, plot_end - 2 * epoch_length)
                else:
                    plot_start = 0
                    plot_end = 2 * epoch_length
            
            if not self.mosaic_relationships:
                self.show_message("No Relationships", "No mosaic relationships available.")
                return
            
            print(f"Debug: Plotting {len(self.mosaic_relationships)} relationships")
            print(f"Debug: Time range: {plot_start:.3f}s to {plot_end:.3f}s")
            print(f"Debug: Channel mapping: {self.channel_mapping}")
            
            # Clear previous plots
            self.figure.clear()
            
            # Create single plot with all mosaic relationships (matching widget.py style)
            ax = self.figure.add_subplot(1, 1, 1)
            
            # Collect valid data for plotting
            valid_relationships = []
            valid_data = []
            successful_plots = 0
            
            # Collect data for plotting (but don't recalculate global sigma)
            for relationship in self.mosaic_relationships:
                time_array, mosaic_data = self.calculate_mosaic_data(relationship, plot_start, plot_end)
                if time_array is not None and mosaic_data is not None:
                    valid_relationships.append(relationship)
                    valid_data.append((time_array, mosaic_data))
            
            if not valid_relationships:
                self.show_message("No Data", "Could not calculate mosaic data for any relationships in the selected time range.")
                return
            
            # Use the fixed global sigma for consistent scaling
            global_sigma = self.global_sigma if self.global_sigma is not None else 100.0
            
            # Get user-defined sigma multipliers
            try:
                sigma_multiplier = float(self.spacing_multiplier_input.text() or "5")
                y_range_multiplier = float(self.ylim_multiplier_input.text() or "5")
            except ValueError:
                # Fall back to defaults if invalid input
                sigma_multiplier = 5
                y_range_multiplier = 5
                self.spacing_multiplier_input.setText("5")
                self.ylim_multiplier_input.setText("5")
            
            # Channel spacing and Y-axis range (user-configurable)
            channel_spacing = sigma_multiplier * global_sigma
            y_range = y_range_multiplier * global_sigma
            
            # Calculate total height for scale bars
            total_height = (len(valid_relationships) - 1) * channel_spacing + 2 * y_range
            y_center = 0
            
            # Plot each mosaic relationship on the same axes
            for i, (relationship, (time_array, mosaic_data)) in enumerate(zip(valid_relationships, valid_data)):
                # Calculate vertical offset for this channel
                offset = (i - (len(valid_relationships) - 1) / 2) * channel_spacing
                offset_signal = mosaic_data + offset
                
                # Plot mosaic signal with standard EEG styling
                relationship_name = relationship.get('name', f'Mosaic {i+1}')
                electrode_a = relationship.get('electrode_a', 'Unknown')
                electrode_b = relationship.get('electrode_b', 'Unknown')
                
                # Plot signal (matching widget.py line style)
                ax.plot(time_array, offset_signal, color='black', 
                       linewidth=0.8, alpha=0.8)
                
                # Add channel label on the right side (widget.py style)
                ax.text(time_array[-1], offset, f'  {relationship_name}', 
                       va='center', ha='left', fontsize=8, fontweight='bold')
                
                successful_plots += 1
            
            # Add vertical line at the center time (start_time) - widget.py style
            ax.axvline(start_time, color='red', linestyle='--', alpha=1.0, linewidth=1)
            
            # Set Y-axis limits matching widget.py scaling
            ax.set_ylim(y_center - total_height/2, y_center + total_height/2)
            
            # Set X-axis limits to fit the data exactly
            if valid_data:
                time_array = valid_data[0][0]  # Use first valid time array
                ax.set_xlim(time_array[0], time_array[-1])
            
            # Clean axes (remove spines and ticks) - widget.py style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add scale bars
            if valid_data:
                time_array = valid_data[0][0]  # Use first valid time array
                self.add_scale_bars(ax, time_array, global_sigma, y_center, total_height)
            
            # Add overall title (matching widget.py style)
            title = f'Mosaic Data: {plot_start:.2f}s - {plot_end:.2f}s (σ={global_sigma:.1f}μV)'
            self.figure.suptitle(title, fontsize=12, fontweight='bold')
            
            # Update current parameters
            self.current_start_time = start_time
            self.current_epoch_length = epoch_length
            
            # Refresh canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update info display
            self.update_info_display()
            
        except Exception as e:
            print(f"Error updating mosaic plot: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Error", f"Error updating plot: {str(e)}")
    
    def add_scale_bars(self, ax, time_data, global_sigma, y_center, total_height):
        """
        Add scale bars for amplitude and time to the mosaic plot.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The plot axes
        time_data : np.array
            Time data array
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
        scale_x = time_data[0] + 0.02 * (time_data[-1] - time_data[0])
        scale_y_bottom = y_center + total_height/2 - scale_bar_size - 20
        scale_y_top = scale_y_bottom + scale_bar_size
        
        ax.plot([scale_x, scale_x], [scale_y_bottom, scale_y_top], 'k-', linewidth=2)
        ax.text(scale_x - 0.01 * (time_data[-1] - time_data[0]), 
               (scale_y_bottom + scale_y_top) / 2, 
               f'{int(scale_bar_size)}μV', 
               ha='right', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # X-scale bar (time)
        time_duration = time_data[-1] - time_data[0]
        
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
        bar_end_x = time_data[-1] - 0.05 * time_duration
        bar_start_x = bar_end_x - time_scale
        time_bar_y = y_center - total_height/2 + 20
        
        # Only draw the bar if it fits within the visible time range
        if bar_start_x >= time_data[0]:
            ax.plot([bar_start_x, bar_end_x], [time_bar_y, time_bar_y], 'k-', linewidth=2)
            ax.text((bar_start_x + bar_end_x) / 2, 
                   time_bar_y - 15, 
                   f'{time_scale}s', 
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def go_previous_epoch(self):
        """Navigate to previous epoch."""
        new_start = self.current_start_time - self.current_epoch_length
        if new_start < 0:
            new_start = 0
        self.start_time_input.setText(f"{new_start:.3f}")
        self.update_mosaic_plot()
        
    def go_next_epoch(self):
        """Navigate to next epoch."""
        new_start = self.current_start_time + self.current_epoch_length
        
        # Check against total duration
        max_end = new_start + self.current_epoch_length
        if max_end > self.total_duration:
            new_start = max(0, self.total_duration - 2 * self.current_epoch_length)
            
        self.start_time_input.setText(f"{new_start:.3f}")
        self.update_mosaic_plot()
    
    def update_info_display(self):
        """Update the information display."""
        try:
            start_time = float(self.start_time_input.text() or "0.0")
            epoch_length = float(self.epoch_length_input.text() or "10.0")
            
            # Calculate actual plot range with boundary checks
            plot_start = max(0, start_time - epoch_length)
            plot_end = start_time + epoch_length
            if self.total_duration > 0:
                plot_end = min(plot_end, self.total_duration)
            
            # Get current scaling parameters
            try:
                spacing_multiplier = float(self.spacing_multiplier_input.text() or "5")
                ylim_multiplier = float(self.ylim_multiplier_input.text() or "5")
            except ValueError:
                spacing_multiplier = 5
                ylim_multiplier = 5
            
            info_lines = [
                f"📊 Mosaic Data Analysis",
                f"═══════════════════════",
                f"⏱️ Total Duration: {self.total_duration:.2f}s",
                f"📡 Sample Rate: {self.sample_rate} Hz" if self.sample_rate else "📡 Sample Rate: Unknown",
                f"📁 Files: {len(self.file_metadata)}",
                f"📏 Global Sigma: {self.global_sigma:.2f} μV" if self.global_sigma else "📏 Global Sigma: Not calculated",
                "",
                f"📐 Plot Scaling:",
                f"  • Channel Spacing: {spacing_multiplier}σ",
                f"  • Y-axis Range: {ylim_multiplier}σ",
                "",
                f"🎯 Current Analysis:",
                f"  • Center Time: {start_time:.3f}s",
                f"  • Epoch Length: {epoch_length:.2f}s",
                f"  • Plot Range: {plot_start:.3f}s to {plot_end:.3f}s",
                "",
                f"🔗 All Mosaic Relationships ({len(self.mosaic_relationships)}):",
            ]
            
            for i, rel in enumerate(self.mosaic_relationships):
                name = rel.get('name', f'Mosaic {i+1}')
                electrode_a = rel.get('electrode_a', 'Unknown')
                electrode_b = rel.get('electrode_b', 'Unknown')
                
                info_lines.extend([
                    f"  {i+1}. {name}",
                    f"     Working: E{electrode_a} → Reference: E{electrode_b}",
                ])
                
                # Get channel mapping info if available
                working_channel = self.channel_mapping.get(electrode_a)
                reference_channel = self.channel_mapping.get(electrode_b)
                
                if working_channel is not None and reference_channel is not None:
                    info_lines.append(f"     Channels: {working_channel} - {reference_channel}")
                else:
                    info_lines.append(f"     ⚠️ Channels: Not mapped")
                info_lines.append("")
            
            info_lines.extend([
                "⌨️ Navigation Controls:",
                "  • Left Arrow: Previous epoch",
                "  • Right Arrow: Next epoch", 
                "  • Update Button: Refresh plot",
                "",
                "🎛️ Scaling Controls:",
                "  • Adjust Channel Spacing: σ multiplier for vertical separation",
                "  • Adjust Y-axis Range: σ multiplier for plot height",
                "  • Apply Scaling: Update plot with new parameters",
                "  • + Key: Increase channel spacing by 0.5σ",
                "  • - Key: Decrease channel spacing by 0.5σ"
            ])
            
        except Exception as e:
            info_lines = [f"❌ Error updating info display: {str(e)}"]
            
        self.info_text.setPlainText("\n".join(info_lines))
    
    def show_message(self, title, message):
        """Show a message box."""
        QMessageBox.information(self, title, message)
