"""
Epilepsy/Seizure Label Main Window for MEEG Analysis.

This module contains the main window for seizure detection and labeling.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QMessageBox, QTextEdit, QSpinBox, QPushButton,
                             QGroupBox, QFormLayout, QSplitter, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtGui import QColor

# Import theme system
from theme import preferences_manager

# Import custom widgets
from .topowindow import TopographyWidget
from .mosaicwindow import MosaicPlotterWidget
from .spectrogram_label_widgets import SpectrogramWidget, LabelWidget


class EpilepsyLabelWindow(QWidget):
    """Main window for seizure/epilepsy labeling."""
    
    def __init__(self, electrode_positions=None, current_data=None, 
                 channel_mapping=None, mosaic_relationships=None, parent=None):
        super().__init__(parent)
        self.electrode_positions = electrode_positions or []
        self.current_data = current_data
        self.channel_mapping = channel_mapping or {}
        self.mosaic_relationships = mosaic_relationships or []
        
        # Load theme colors
        self.load_theme()
        
        # Extract EEG data from cache
        self.df, self.mosaic, self.sampling_rate = self.extract_eeg_data()
        
        # Initialize epoch parameters
        self.epoch_length = 1  # seconds
        self.current_epoch = 0
        self.epochs_to_show = 15  # Display 5 epochs at once
        self.label_file = None  # CSV file for labels
        
        self.init_ui()
        
    def extract_eeg_data(self) -> Tuple[pd.DataFrame, list, float]:
        """
        Extract EEG data from the current_data loaded from cache.
        Creates a DataFrame with electrode names as columns.
        
        Returns:
        --------
        tuple : (df, mosaic, sampling_rate)
            - df: pandas DataFrame where df['electrode_name'] contains amplifier data
            - mosaic: list of tuples ('mosaic_name', 'electrode_a_name', 'electrode_b_name')
            - sampling_rate: float, sampling rate in Hz
        """
        df = pd.DataFrame()
        mosaic = []
        sampling_rate = None
        
        if not self.current_data:
            print("No data available to extract")
            return df, mosaic, sampling_rate
        
        try:
            # Step 1: Extract raw data and metadata
            amplifier_data_list = []
            amplifier_channels = []
            frequency_parameters = None
            
            for filename, result, data_present in self.current_data:
                if data_present and result:
                    if 'amplifier_data' in result:
                        amplifier_data_list.append(result['amplifier_data'])
                    
                    if 'frequency_parameters' in result and not frequency_parameters:
                        frequency_parameters = result['frequency_parameters']
                    
                    if 'amplifier_channels' in result and not amplifier_channels:
                        amplifier_channels = result['amplifier_channels']
            
            # Step 2: Concatenate data from multiple files
            if amplifier_data_list:
                if len(amplifier_data_list) > 1:
                    amplifier_data = np.concatenate(amplifier_data_list, axis=1)
                else:
                    amplifier_data = amplifier_data_list[0]
            else:
                print("No amplifier data found")
                return df, mosaic, sampling_rate
            
            # Step 3: Extract sampling rate
            if frequency_parameters:
                sampling_rate = frequency_parameters.get('amplifier_sample_rate', None)
            
            # Step 4: Create mapping from electrode number -> name
            electrode_num_to_name = {}
            for electrode in self.electrode_positions:
                num = electrode['number']
                name = electrode.get('name', f"E{num}")
                electrode_num_to_name[num] = name
            
            # Step 5: Create mapping from amplifier channel name -> electrode name
            # channel_mapping maps electrode_number -> amplifier_channel_name
            # We need: amplifier_channel_name -> electrode_name
            channel_name_to_electrode_name = {}
            duplicate_channels = {}
            
            for electrode_num, amp_channel_name in self.channel_mapping.items():
                # electrode_num might be int or str
                electrode_num_int = int(electrode_num) if isinstance(electrode_num, str) else electrode_num
                electrode_name = electrode_num_to_name.get(electrode_num_int, f"E{electrode_num_int}")
                
                # Check for duplicate mappings (multiple electrodes mapped to same amplifier channel)
                if amp_channel_name in channel_name_to_electrode_name:
                    if amp_channel_name not in duplicate_channels:
                        duplicate_channels[amp_channel_name] = [channel_name_to_electrode_name[amp_channel_name]]
                    duplicate_channels[amp_channel_name].append(electrode_name)
                    print(f"WARNING: Amplifier channel '{amp_channel_name}' is mapped to multiple electrodes:")
                    print(f"  - Electrodes: {duplicate_channels[amp_channel_name]}")
                
                channel_name_to_electrode_name[amp_channel_name] = electrode_name
            
            print(f"\n=== Channel Mapping Debug ===")
            print(f"Total mapped electrodes: {len(self.channel_mapping)}")
            print(f"Unique amplifier channels: {len(channel_name_to_electrode_name)}")
            print(f"Electrode -> Amplifier channel:")
            for electrode_num, amp_channel_name in self.channel_mapping.items():
                electrode_num_int = int(electrode_num) if isinstance(electrode_num, str) else electrode_num
                electrode_name = electrode_num_to_name.get(electrode_num_int, f"E{electrode_num_int}")
                print(f"  {electrode_name} (#{electrode_num}) -> {amp_channel_name}")
            
            if duplicate_channels:
                print(f"\nWARNING: Found {len(duplicate_channels)} amplifier channel(s) mapped to multiple electrodes:")
                for amp_ch, elec_list in duplicate_channels.items():
                    print(f"  {amp_ch}: {elec_list}")
                print(f"  This reduces the number of unique DataFrame columns!")
            print(f"===========================\n")
            
            # Step 6: Build DataFrame with electrode names as columns
            # Only include channels that have a mapping defined
            df_dict = {}
            matched_channels = []
            unmatched_channels = []
            
            for i, channel_info in enumerate(amplifier_channels):
                amp_channel_name = channel_info.get('native_channel_name', f'Channel_{i}')
                
                # Only add this channel if it's mapped to an electrode
                if amp_channel_name in channel_name_to_electrode_name:
                    electrode_name = channel_name_to_electrode_name[amp_channel_name]
                    
                    # Add this channel's data to the DataFrame dictionary
                    if i < amplifier_data.shape[0]:
                        df_dict[electrode_name] = amplifier_data[i, :]
                        matched_channels.append((amp_channel_name, electrode_name))
                else:
                    unmatched_channels.append(amp_channel_name)
            
            print(f"\n=== DataFrame Construction ===")
            print(f"Matched channels (in amplifier data): {len(matched_channels)}")
            for amp_ch, elec_name in matched_channels:
                print(f"  {amp_ch} -> {elec_name}")
            
            if unmatched_channels:
                print(f"\nAmplifier channels without mapping (first 10): {unmatched_channels[:10]}")
            print(f"================================\n")
            
            df = pd.DataFrame(df_dict)
            
            # Assert that mapped channels equals DataFrame columns
            expected_channels = len(self.channel_mapping)
            actual_channels = len(df.columns)
            unique_amp_channels = len(channel_name_to_electrode_name)
            
            print(f"\n=== Channel Count Verification ===")
            print(f"Mapped electrodes in config: {expected_channels}")
            print(f"Unique amplifier channels: {unique_amp_channels}")
            print(f"DataFrame columns: {actual_channels}")
            
            if expected_channels != actual_channels:
                print(f"⚠ Channel count mismatch detected!")
                
                if duplicate_channels:
                    print(f"\nCause: {len(duplicate_channels)} amplifier channel(s) are mapped to multiple electrodes:")
                    for amp_ch, elec_list in duplicate_channels.items():
                        print(f"  {amp_ch}: {elec_list}")
                    print(f"\nExpected {expected_channels} but got {actual_channels} because duplicates reduce unique columns.")
                    print(f"Solution: Each amplifier channel should only be mapped to one electrode.")
                else:
                    # Find which mappings didn't make it into the DataFrame
                    mapped_electrodes = set()
                    for electrode_num in self.channel_mapping.keys():
                        electrode_num_int = int(electrode_num) if isinstance(electrode_num, str) else electrode_num
                        electrode_name = electrode_num_to_name.get(electrode_num_int, f"E{electrode_num_int}")
                        mapped_electrodes.add(electrode_name)
                    
                    df_electrodes = set(df.columns)
                    missing = mapped_electrodes - df_electrodes
                    if missing:
                        print(f"Mapped electrodes not in DataFrame: {missing}")
            else:
                print(f"✓ Channel count matches!")
            print(f"=================================\n")
            
            # Assertion with informative error message
            if duplicate_channels:
                error_msg = (
                    f"Channel count mismatch: Expected {expected_channels} mapped electrodes, "
                    f"but DataFrame has {actual_channels} columns. "
                    f"Reason: {len(duplicate_channels)} amplifier channel(s) are mapped to multiple electrodes. "
                    f"Duplicate channels: {list(duplicate_channels.keys())}"
                )
            else:
                error_msg = (
                    f"Channel count mismatch: Expected {expected_channels} mapped channels, "
                    f"but DataFrame has {actual_channels} columns"
                )
            
            assert expected_channels == actual_channels, error_msg
            
            # Step 7: Process mosaic relationships to use electrode names
            for rel in self.mosaic_relationships:
                if isinstance(rel, dict):
                    mosaic_name = rel.get('name', 'mosaic')
                    electrode_a_num = rel.get('electrode_a')
                    electrode_b_num = rel.get('electrode_b')
                    
                    # Convert numbers to names
                    electrode_a_name = electrode_num_to_name.get(electrode_a_num, f"E{electrode_a_num}")
                    electrode_b_name = electrode_num_to_name.get(electrode_b_num, f"E{electrode_b_num}")
                    
                    mosaic.append((mosaic_name, electrode_a_name, electrode_b_name))
            
            # Print extraction summary
            print(f"\n=== EEG Data Extraction Summary ===")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns (electrodes): {list(df.columns)}")
            print(f"Number of samples: {len(df)}")
            print(f"Sampling rate: {sampling_rate} Hz")
            print(f"Mosaic relationships: {len(mosaic)}")
            for m_name, e_a, e_b in mosaic:
                print(f"  - {m_name}: {e_a} ↔ {e_b}")
            print(f"===================================\n")
            
        except Exception as e:
            print(f"Error extracting EEG data: {e}")
            import traceback
            traceback.print_exc()
        
        return df, mosaic, sampling_rate
    
    def load_theme(self):
        """Load theme colors based on current theme setting."""
        current_theme = preferences_manager.get_theme()
        
        if current_theme == "tokyo_night":
            from theme import TOKYO_NIGHT_COLORS as THEME_COLORS
        elif current_theme == "dark":
            from theme import DARK_COLORS as THEME_COLORS
        else:  # normal
            from theme import NORMAL_COLORS as THEME_COLORS
        
        self.theme_colors = THEME_COLORS
        
    def init_ui(self):
        """Initialize the user interface with 5-panel layout."""
        self.showFullScreen()
        
        # Set window background color to match theme
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(self.theme_colors['bg_primary']))
        self.setPalette(palette)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main splitter (left 0.3 | right 0.7)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === LEFT PANEL (0.3 width) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Left splitter (upper 0.5 | lower 0.5)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Left Upper: Topography (0.5 height)
        self.topo_widget = TopographyWidget(self.electrode_positions, self.theme_colors)
        left_splitter.addWidget(self.topo_widget)
        
        # Left Lower: Control Panel (0.5 height)
        self.control_panel = self.create_control_panel()
        left_splitter.addWidget(self.control_panel)
        
        # Set left splitter sizes (50-50)
        left_splitter.setSizes([500, 500])
        left_layout.addWidget(left_splitter)
        
        main_splitter.addWidget(left_widget)
        
        # === RIGHT PANEL (0.7 width) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # Right Upper: Mosaic Plotter (0.6 height)
        self.mosaic_plotter = MosaicPlotterWidget(self.df, self.mosaic, self.sampling_rate, self.theme_colors)
        self.mosaic_plotter.epoch_clicked.connect(self.on_epoch_selected)
        right_layout.addWidget(self.mosaic_plotter, 60)  # 60% stretch
        
        # Right Middle: Spectrogram (0.2 height)
        self.spectrogram_widget = SpectrogramWidget(self.theme_colors)
        # Get first channel for spectrogram if available
        if not self.df.empty:
            first_channel_data = self.df.iloc[:, 0].values
            self.spectrogram_widget.set_data(first_channel_data, self.sampling_rate)
        right_layout.addWidget(self.spectrogram_widget, 20)  # 20% stretch
        
        # Right Lower: Label Window (0.2 height)
        self.label_widget = LabelWidget(self.theme_colors)
        self.label_widget.label_changed.connect(self.on_label_changed)
        # Initialize labels based on number of epochs
        n_epochs = self.get_n_epochs()
        self.label_widget.initialize_labels(n_epochs)
        right_layout.addWidget(self.label_widget, 20)  # 20% stretch
        
        main_splitter.addWidget(right_widget)
        
        # Set main splitter sizes (30-70)
        main_splitter.setSizes([300, 700])
        
        main_layout.addWidget(main_splitter)
        
        # Initialize epoch display
        self.update_epoch_displays()
        
    def create_control_panel(self):
        """Create the control panel widget."""
        panel = QGroupBox("Control Panel")
        
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Seizure Analysis Controls")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Data summary
        num_channels = len(self.df.columns) if not self.df.empty else 0
        num_samples = len(self.df) if not self.df.empty else 0
        n_epochs = self.get_n_epochs()
        duration_sec = num_samples / self.sampling_rate if self.sampling_rate and self.sampling_rate > 0 else 0
        
        summary_group = QGroupBox("Data Summary")
        summary_layout = QFormLayout(summary_group)
        summary_layout.addRow("Electrodes:", QLabel(str(len(self.electrode_positions))))
        summary_layout.addRow("Mapped Channels:", QLabel(str(len(self.channel_mapping))))
        summary_layout.addRow("Mosaic Pairs:", QLabel(str(len(self.mosaic))))
        summary_layout.addRow("Sampling Rate:", QLabel(f"{self.sampling_rate:.0f} Hz"))
        summary_layout.addRow("Duration:", QLabel(f"{duration_sec:.1f} sec"))
        summary_layout.addRow("Total Epochs:", QLabel(str(n_epochs)))
        layout.addWidget(summary_group)
        
        # Epoch controls
        epoch_group = QGroupBox("Epoch Controls")
        epoch_layout = QFormLayout(epoch_group)
        
        # Epoch length
        self.epoch_length_spin = QSpinBox()
        self.epoch_length_spin.setRange(1, 60)
        self.epoch_length_spin.setValue(self.epoch_length)
        self.epoch_length_spin.setSuffix(" sec")
        self.epoch_length_spin.valueChanged.connect(self.on_epoch_length_changed)
        epoch_layout.addRow("Epoch Length:", self.epoch_length_spin)
        
        # Epochs to show
        self.epochs_to_show_spin = QSpinBox()
        self.epochs_to_show_spin.setRange(1, 20)
        self.epochs_to_show_spin.setValue(self.epochs_to_show)
        self.epochs_to_show_spin.valueChanged.connect(self.on_epochs_to_show_changed)
        epoch_layout.addRow("Epochs to Show:", self.epochs_to_show_spin)
        
        # Current epoch
        self.current_epoch_spin = QSpinBox()
        self.current_epoch_spin.setRange(0, max(0, n_epochs - 1))
        self.current_epoch_spin.setValue(self.current_epoch)
        self.current_epoch_spin.valueChanged.connect(self.on_current_epoch_changed)
        epoch_layout.addRow("Current Epoch:", self.current_epoch_spin)
        
        layout.addWidget(epoch_group)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self.previous_epoch)
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self.next_epoch)
        nav_layout.addWidget(next_btn)
        
        layout.addWidget(nav_group)
        
        # Label file buttons
        action_group = QGroupBox("Labels")
        action_layout = QVBoxLayout(action_group)
        
        create_btn = QPushButton("Create Label File")
        create_btn.clicked.connect(self.create_label_file)
        action_layout.addWidget(create_btn)
        
        import_btn = QPushButton("Import Label File")
        import_btn.clicked.connect(self.import_label_file)
        action_layout.addWidget(import_btn)
        
        save_btn = QPushButton("Save Labels")
        save_btn.clicked.connect(self.save_labels)
        action_layout.addWidget(save_btn)
        
        layout.addWidget(action_group)
        
        layout.addStretch()
        
        return panel
        
    def get_n_epochs(self):
        """Get total number of epochs in the data."""
        if self.df.empty or self.sampling_rate == 0:
            return 0
        total_samples = len(self.df)
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        return max(1, total_samples // samples_per_epoch)
    
    def get_epoch_data(self, epoch_idx):
        """
        Extract data for a specific epoch.
        
        Parameters:
        -----------
        epoch_idx : int
            Epoch index
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing the epoch data with channels as columns
        """
        if self.df.empty or self.sampling_rate == 0:
            return None
        
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        start_idx = epoch_idx * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        
        # Check bounds
        if start_idx >= len(self.df):
            return None
        
        # Clip end_idx to data length
        end_idx = min(end_idx, len(self.df))
        
        # Extract epoch data
        epoch_data = self.df.iloc[start_idx:end_idx].copy()
        
        return epoch_data
        
    def update_epoch_displays(self):
        """Update all widgets to show current epoch."""
        self.mosaic_plotter.set_epoch(self.current_epoch)
        self.spectrogram_widget.set_epoch(self.current_epoch)
        self.label_widget.set_epoch(self.current_epoch)
        self.topo_widget.set_epoch(self.current_epoch)
        
        # Extract epoch data and pass to topography widget
        epoch_data = self.get_epoch_data(self.current_epoch)
        if epoch_data is not None:
            self.topo_widget.set_data(epoch_data, self.sampling_rate, self.epoch_length)
        
    def on_epoch_selected(self, epoch_idx):
        """Handle epoch selection from mosaic plotter."""
        self.current_epoch = epoch_idx
        self.current_epoch_spin.setValue(epoch_idx)
        self.update_epoch_displays()
        
    def on_label_changed(self, epoch_idx, score):
        """Handle label change from label widget."""
        print(f"Epoch {epoch_idx} labeled as Racine score {score}")
        # Sync current epoch and update all displays
        self.current_epoch = epoch_idx
        self.current_epoch_spin.setValue(epoch_idx)
        self.update_epoch_displays()
        
    def on_epoch_length_changed(self, value):
        """Handle epoch length change."""
        self.epoch_length = value
        # Update widgets with new epoch length
        self.mosaic_plotter.set_epoch_length(value)
        self.spectrogram_widget.set_epoch_length(value)
        self.topo_widget.set_epoch_length(value)
        # Recalculate epochs
        n_epochs = self.get_n_epochs()
        self.current_epoch_spin.setRange(0, max(0, n_epochs - 1))
        self.label_widget.initialize_labels(n_epochs)
        self.update_epoch_displays()
        
    def on_epochs_to_show_changed(self, value):
        """Handle epochs to show change."""
        self.epochs_to_show = value
        self.mosaic_plotter.set_epochs_to_show(value)
        self.spectrogram_widget.set_epochs_to_show(value)
        self.label_widget.set_epochs_to_show(value)
        
    def on_current_epoch_changed(self, value):
        """Handle current epoch change from spin box."""
        self.current_epoch = value
        self.update_epoch_displays()
        
    def previous_epoch(self):
        """Navigate to previous epoch."""
        if self.current_epoch > 0:
            self.current_epoch -= 1
            self.current_epoch_spin.setValue(self.current_epoch)
            self.update_epoch_displays()
            
    def next_epoch(self):
        """Navigate to next epoch."""
        n_epochs = self.get_n_epochs()
        if self.current_epoch < n_epochs - 1:
            self.current_epoch += 1
            self.current_epoch_spin.setValue(self.current_epoch)
            self.update_epoch_displays()
    
    def create_label_file(self):
        """Create a new CSV file for labeling."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Create Label File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Ensure .csv extension
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        
        try:
            n_epochs = self.get_n_epochs()
            
            # Create a new CSV with empty labels
            df = pd.DataFrame({
                'epoch': np.arange(n_epochs),
                'label': [0] * n_epochs  # Initialize with 0 (no seizure)
            })
            
            df.to_csv(file_path, index=False)
            
            # Set as current label file
            self.label_file = file_path
            
            # Initialize label widget with empty labels
            self.label_widget.initialize_labels(n_epochs)
            
            QMessageBox.information(self, "Create Success", 
                                   f"Created new label file with {n_epochs} epochs:\n{file_path}\n\n"
                                   f"All labels initialized to 0 (no seizure).")
            print(f"Created label file: {self.label_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Create Error", f"Failed to create label file:\n{str(e)}")
            self.label_file = None
            
    def import_label_file(self):
        """Import a CSV file for labels (can be empty or contain existing labels)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Label File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.label_file = file_path
        
        try:
            # Try to read the CSV
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                
                # Check if 'label' column exists
                if 'label' in df.columns:
                    # Load existing labels
                    labels = df['label'].values
                    n_epochs = self.get_n_epochs()
                    
                    if len(labels) == n_epochs:
                        self.label_widget.set_labels(labels)
                        QMessageBox.information(self, "Import Success", 
                                               f"Loaded {len(labels)} existing labels from file.")
                    else:
                        # Initialize with current labels
                        self.label_widget.initialize_labels(n_epochs)
                        QMessageBox.warning(self, "Label Count Mismatch", 
                                           f"File has {len(labels)} labels but data has {n_epochs} epochs.\n"
                                           f"Initialized with empty labels instead.")
                else:
                    # CSV exists but no label column - initialize empty labels
                    n_epochs = self.get_n_epochs()
                    self.label_widget.initialize_labels(n_epochs)
                    QMessageBox.information(self, "Import Success", 
                                           f"Imported file (no existing labels).\nReady to label {n_epochs} epochs.")
            
            print(f"Label file set to: {self.label_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import file:\n{str(e)}")
            self.label_file = None
    
    def save_labels(self):
        """Save current labels to the imported CSV file."""
        if not self.label_file:
            QMessageBox.warning(self, "No Label File", 
                               "Please import a label file first using 'Import Label File' button.")
            return
        
        try:
            labels = self.label_widget.get_labels()
            
            # Create or update CSV with labels
            n_epochs = len(labels)
            df = pd.DataFrame({
                'epoch': np.arange(n_epochs),
                'label': labels
            })
            
            df.to_csv(self.label_file, index=False)
            
            QMessageBox.information(self, "Save Success", 
                                   f"Saved {n_epochs} labels to:\n{self.label_file}")
            print(f"Labels saved to: {self.label_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save labels:\n{str(e)}")

