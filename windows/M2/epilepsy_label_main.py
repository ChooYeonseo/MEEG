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
                             QGroupBox, QFormLayout, QSplitter, QFileDialog,
                             QDialog, QDoubleSpinBox, QCheckBox, QComboBox,
                             QDialogButtonBox, QMenuBar, QTabWidget,
                             QInputDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QAction, QKeySequence
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import theme system
from theme import preferences_manager

# Import custom widgets
from .topowindow import TopographyWidget
from .mosaicwindow import MosaicPlotterWidget
from .spectrogram_label_widgets import SpectrogramWidget, LabelWidget
from .video_sync_widget import VideoSyncWidget


class InitialConfigDialog(QDialog):
    """Dialog to configure epoch time interval and sampling rate before main window."""
    
    def __init__(self, detected_sampling_rate=None, theme_colors=None, parent=None):
        super().__init__(parent)
        self.detected_sampling_rate = detected_sampling_rate
        self.epoch_length = 1  # default
        self.sampling_rate = detected_sampling_rate
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the configuration dialog UI."""
        self.setWindowTitle("EEG Analysis Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # Set background color to match theme
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(self.theme_colors['bg_primary']))
        self.setPalette(palette)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Configure Analysis Parameters")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Epoch length input
        epoch_group = QGroupBox("Epoch Time Interval")
        epoch_layout = QFormLayout(epoch_group)
        
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 60)
        self.epoch_spin.setValue(1)
        self.epoch_spin.setSingleStep(1)
        self.epoch_spin.setSuffix(" seconds")
        epoch_layout.addRow("Epoch Length:", self.epoch_spin)
        
        layout.addWidget(epoch_group)
        
        # Sampling rate section
        sampling_group = QGroupBox("Sampling Rate")
        sampling_layout = QVBoxLayout(sampling_group)
        
        # Detected sampling rate display
        if self.detected_sampling_rate:
            detected_label = QLabel(f"Detected: {self.detected_sampling_rate:.0f} Hz")
            detected_font = QFont()
            detected_font.setBold(True)
            detected_label.setFont(detected_font)
            sampling_layout.addWidget(detected_label)
            
            # Checkbox to use detected rate
            self.use_detected_check = QCheckBox("Use detected sampling rate")
            self.use_detected_check.setChecked(True)
            self.use_detected_check.toggled.connect(self.on_use_detected_toggled)
            sampling_layout.addWidget(self.use_detected_check)
        else:
            self.use_detected_check = None
            no_detect_label = QLabel("No sampling rate detected")
            sampling_layout.addWidget(no_detect_label)
        
        # Manual sampling rate input
        manual_layout = QFormLayout()
        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(1.0, 100000.0)
        self.sampling_spin.setValue(self.detected_sampling_rate if self.detected_sampling_rate else 2000.0)
        self.sampling_spin.setSingleStep(100.0)
        self.sampling_spin.setSuffix(" Hz")
        self.sampling_spin.setDecimals(1)
        self.sampling_spin.setEnabled(False if self.detected_sampling_rate else True)
        manual_layout.addRow("Manual Override:", self.sampling_spin)
        
        sampling_layout.addLayout(manual_layout)
        layout.addWidget(sampling_group)
        
        layout.addSpacing(20)
        
        # Button box
        button_box = QDialogButtonBox()
        self.go_button = QPushButton("Go")
        self.go_button.setDefault(True)
        self.go_button.clicked.connect(self.accept)
        button_box.addButton(self.go_button, QDialogButtonBox.ButtonRole.AcceptRole)
        
        cancel_button = button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.clicked.connect(self.reject)
        
        layout.addWidget(button_box)
        
    def on_use_detected_toggled(self, checked):
        """Handle use detected checkbox toggle."""
        self.sampling_spin.setEnabled(not checked)
        
    def accept(self):
        """Validate and accept configuration."""
        self.epoch_length = self.epoch_spin.value()
        
        if self.use_detected_check and self.use_detected_check.isChecked():
            self.sampling_rate = self.detected_sampling_rate
        else:
            self.sampling_rate = self.sampling_spin.value()
        
        super().accept()
        
    def get_config(self):
        """Return the configured parameters."""
        return self.epoch_length, self.sampling_rate


class EpilepsyLabelWindow(QWidget):
    """Main window for seizure/epilepsy labeling."""
    
    def __init__(self, electrode_positions=None, current_data=None, 
                 channel_mapping=None, mosaic_relationships=None, cache_key=None, parent=None):
        super().__init__(parent)
        self.electrode_positions = electrode_positions or []
        self.current_data = current_data
        self.channel_mapping = channel_mapping or {}
        self.mosaic_relationships = mosaic_relationships or []
        self.cache_key = cache_key
        
        # Load theme colors
        self.load_theme()
        
        # Extract EEG data from cache
        self.df, self.mosaic, self.detected_sampling_rate = self.extract_eeg_data()
        
        # Show configuration dialog
        config_dialog = InitialConfigDialog(self.detected_sampling_rate, self.theme_colors, parent)
        if config_dialog.exec() != QDialog.DialogCode.Accepted:
            # User cancelled - close window
            self.close()
            return
        
        # Get configuration from dialog
        self.epoch_length, self.sampling_rate = config_dialog.get_config()
        
        # Initialize epoch parameters
        self.current_epoch = 0
        self.mosaic_epochs_to_show = 15  # For mosaic plot
        self.label_epochs_to_show = 5   # For label widget
        # Set spectrogram to full view initially - will be updated after data is loaded
        self.spectrogram_epochs_to_show = None  # Will be set to total epochs
        self.label_file = None  # CSV file for labels
        
        # Spectrogram brightness controls
        self.spectrogram_vmin = None  # Auto
        self.spectrogram_vmax = None  # Auto
        self.zoom_acceleration = 1  # Tracks consecutive zoom operations
        self.last_zoom_time = 0  # Track last zoom button click time
        self.zoom_threshold = 0.3  # Time threshold for continuous clicks (seconds)
        
        # Topography toggle state
        self.topo_enabled = True  # Topography updates enabled by default
        
        self.init_ui()
    
    def show_message_box(self, icon, title, text):
        """Show a styled message box with visible buttons.
        
        Args:
            icon: QMessageBox.Icon (Information, Warning, Critical)
            title: Dialog title
            text: Message text
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        
        # Get theme colors for button styling - match secondary button style
        bg_color = self.theme_colors.get('bg_tertiary', '#414868')
        fg_color = self.theme_colors.get('fg_primary', '#c0caf5')
        border_color = self.theme_colors.get('border', '#414868')
        hover_bg = self.theme_colors.get('hover', '#565f89')
        hover_border = self.theme_colors.get('accent_blue', '#7aa2f7')  # Accent on hover only
        
        # Style the message box buttons to match theme secondary button style
        msg_box.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: {fg_color};
                border: 1px solid {border_color};
                padding: 8px 16px;
                border-radius: 6px;
                min-width: 80px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                border-color: {hover_border};
            }}
            QPushButton:pressed {{
                background-color: {self.theme_colors.get('active', '#565f89')};
            }}
        """)
        
        msg_box.exec()
        
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
        
        # Create menu bar
        menu_bar = QMenuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        create_label_action = QAction("Create Label File", self)
        create_label_action.triggered.connect(self.create_label_file)
        file_menu.addAction(create_label_action)
        
        import_label_action = QAction("Import Label File", self)
        import_label_action.triggered.connect(self.import_label_file)
        file_menu.addAction(import_label_action)
        
        save_labels_action = QAction("Save Labels", self)
        save_labels_action.triggered.connect(self.save_labels)
        file_menu.addAction(save_labels_action)
        
        # Save menu
        save_menu = menu_bar.addMenu("Save")
        
        # Save Topography Image (Ctrl+S)
        save_topo_action = QAction("Save Topography Image", self)
        save_topo_action.setShortcut(QKeySequence("Ctrl+S"))
        save_topo_action.triggered.connect(self.save_topography_image)
        save_menu.addAction(save_topo_action)
        
        # Save Mosaic Plot (Ctrl+Shift+S)
        save_mosaic_action = QAction("Save Mosaic Plot", self)
        save_mosaic_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_mosaic_action.triggered.connect(self.save_mosaic_plot)
        save_menu.addAction(save_mosaic_action)
        
        # Save Power Analysis (Ctrl+E)
        save_power_action = QAction("Save Power Analysis", self)
        save_power_action.setShortcut(QKeySequence("Ctrl+E"))
        save_power_action.triggered.connect(self.save_power_analysis)
        save_menu.addAction(save_power_action)

        # File to DB (Ctrl+D)
        save_db_action = QAction("File to DB", self)
        save_db_action.setShortcut(QKeySequence("Ctrl+D"))
        save_db_action.triggered.connect(self.save_file_to_db)
        save_menu.addAction(save_db_action)
        
        main_layout.setMenuBar(menu_bar)
        
        # === CREATE TAB WIDGET ===
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # === LABEL TAB (original content) ===
        label_tab = QWidget()
        label_tab_layout = QVBoxLayout(label_tab)
        label_tab_layout.setContentsMargins(0, 0, 0, 0)
        label_tab_layout.setSpacing(5)
        
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
        self.mosaic_plotter.epoch_double_clicked.connect(self.on_epoch_double_clicked)
        self.mosaic_plotter.set_epochs_to_show(self.mosaic_epochs_to_show)
        self.mosaic_plotter.set_epoch_length(self.epoch_length)
        right_layout.addWidget(self.mosaic_plotter, 60)  # 60% stretch
        
        # Right Middle: Spectrogram (0.2 height)
        self.spectrogram_widget = SpectrogramWidget(self.theme_colors)
        self.spectrogram_widget.epoch_selected.connect(self.on_epoch_selected)
        # Get first channel for spectrogram if available
        if not self.df.empty:
            first_channel_data = self.df.iloc[:, 0].values
            print(f"Setting spectrogram data with {len(first_channel_data)} samples at {self.sampling_rate} Hz...")
            # This will trigger pre-computation of spectrogram
            self.spectrogram_widget.set_data(first_channel_data, self.sampling_rate)
        
        # Set initial spectrogram view to full length (all epochs)
        if self.spectrogram_epochs_to_show is None:
            self.spectrogram_epochs_to_show = self.get_n_epochs()
            print(f"Initial spectrogram view set to full length: {self.spectrogram_epochs_to_show} epochs")
        
        self.spectrogram_widget.set_epochs_to_show(self.spectrogram_epochs_to_show)
        self.spectrogram_widget.set_epoch_length(self.epoch_length)
        right_layout.addWidget(self.spectrogram_widget, 20)  # 20% stretch
        
        # Right Lower: Label Window (0.2 height)
        self.label_widget = LabelWidget(self.theme_colors)
        self.label_widget.label_changed.connect(self.on_label_changed)
        self.label_widget.spacebar_pressed.connect(self.on_spacebar_pressed)
        # Initialize labels based on number of epochs
        n_epochs = self.get_n_epochs()
        self.label_widget.initialize_labels(n_epochs)
        self.label_widget.set_epochs_to_show(self.label_epochs_to_show)
        self.label_widget.set_mosaic_epochs_to_show(self.mosaic_epochs_to_show)
        right_layout.addWidget(self.label_widget, 20)  # 20% stretch
        
        main_splitter.addWidget(right_widget)
        
        # Set main splitter sizes (33-67, i.e., 1/3 - 2/3)
        main_splitter.setSizes([333, 667])
        
        label_tab_layout.addWidget(main_splitter)
        self.tab_widget.addTab(label_tab, "📊 Label")
        
        # === VIDEO TAB ===
        self.video_widget = VideoSyncWidget(self.theme_colors)
        self.video_widget.set_epoch_length(self.epoch_length)
        self.video_widget.set_sampling_rate(self.sampling_rate)
        self.video_widget.set_eeg_data(self.df, self.mosaic)
        self.video_widget.set_electrode_positions(self.electrode_positions)
        self.video_widget.return_to_label.connect(self.on_return_to_label)
        self.tab_widget.addTab(self.video_widget, "🎬 Video")
        
        main_layout.addWidget(self.tab_widget)
        
        # Initialize epoch display
        self.update_epoch_displays()

        
    def create_control_panel(self):
        """Create the control panel widget with 3 sub-panels."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Analysis Controls")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Data summary
        num_samples = len(self.df) if not self.df.empty else 0
        n_epochs = self.get_n_epochs()
        duration_sec = num_samples / self.sampling_rate if self.sampling_rate and self.sampling_rate > 0 else 0
        
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        # First row: Electrodes, Sampling Rate, Duration
        summary_text_1 = (f"Electrodes: {len(self.electrode_positions)} | "
                       f"Sampling Rate: {self.sampling_rate:.0f} Hz | "
                       f"Duration: {duration_sec:.1f} sec")
        summary_label_1 = QLabel(summary_text_1)
        summary_label_1.setWordWrap(True)
        summary_layout.addWidget(summary_label_1)
        
        # Second row: Epoch Length, Total Epochs
        summary_text_2 = (f"Epoch Length: {self.epoch_length:.1f} sec | "
                       f"Total Epochs: {n_epochs}")
        summary_label_2 = QLabel(summary_text_2)
        summary_label_2.setWordWrap(True)
        summary_layout.addWidget(summary_label_2)
        
        layout.addWidget(summary_group)
        
        # Topography toggle
        topo_group = QGroupBox("Topography Display")
        topo_layout = QVBoxLayout(topo_group)
        
        self.topo_toggle_btn = QPushButton("Disable Topography")
        self.topo_toggle_btn.setCheckable(True)
        self.topo_toggle_btn.setChecked(False)  # Not disabled by default
        self.topo_toggle_btn.clicked.connect(self.on_topo_toggle_clicked)
        topo_layout.addWidget(self.topo_toggle_btn)
        
        topo_info = QLabel("Disable to speed up labeling")
        topo_info.setWordWrap(True)
        topo_info.setStyleSheet("font-size: 9px; color: gray;")
        topo_layout.addWidget(topo_info)
        
        layout.addWidget(topo_group)
        
        # === Horizontal container for Panel 1 and Panel 3 ===
        horizontal_panels = QWidget()
        horizontal_layout = QHBoxLayout(horizontal_panels)
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(5)
        
        # === Panel 1: EEG Display Control ===
        eeg_group = QGroupBox("EEG Display Control")
        eeg_layout = QFormLayout(eeg_group)
        
        self.mosaic_epochs_combo = QComboBox()
        self.mosaic_epochs_combo.addItems(["1", "3", "5", "7", "9", "11", "13", "15"])
        self.mosaic_epochs_combo.setCurrentText(str(self.mosaic_epochs_to_show))
        self.mosaic_epochs_combo.currentTextChanged.connect(self.on_mosaic_epochs_changed)
        eeg_layout.addRow("Epochs to Show:", self.mosaic_epochs_combo)
        
        horizontal_layout.addWidget(eeg_group)
        
        # === Panel 3: Label Control Panel ===
        label_group = QGroupBox("Label Control Panel")
        label_layout = QFormLayout(label_group)
        
        # Epochs to show
        self.label_epochs_combo = QComboBox()
        self.label_epochs_combo.addItems(["1", "3", "5", "7", "9", "11", "13", "15"])
        self.label_epochs_combo.setCurrentText(str(self.label_epochs_to_show))
        self.label_epochs_combo.currentTextChanged.connect(self.on_label_epochs_changed)
        label_layout.addRow("Epochs to Show:", self.label_epochs_combo)
        
        # Status indicator - create inline widget
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(5)
        
        self.label_status_indicator = QLabel("Empty")
        self.label_status_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_status_indicator.setStyleSheet("""
            QLabel {
                background-color: #d32f2f;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        status_layout.addWidget(self.label_status_indicator)
        status_layout.addStretch()
        
        label_layout.addRow("Status:", status_widget)
        
        horizontal_layout.addWidget(label_group)
        
        layout.addWidget(horizontal_panels)
        
        # === Panel 2: Spectrogram Control Panel ===
        spec_group = QGroupBox("Spectrogram Control Panel")
        spec_layout = QVBoxLayout(spec_group)
        
        # Brightness controls
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Brightness:")
        brightness_layout.addWidget(brightness_label)
        
        self.dimmer_btn = QPushButton("Dimmer")
        self.dimmer_btn.clicked.connect(self.on_dimmer_clicked)
        brightness_layout.addWidget(self.dimmer_btn)
        
        self.brighter_btn = QPushButton("Brighter")
        self.brighter_btn.clicked.connect(self.on_brighter_clicked)
        brightness_layout.addWidget(self.brighter_btn)
        
        spec_layout.addLayout(brightness_layout)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        zoom_layout.addWidget(zoom_label)
        
        self.zoom_out_btn = QPushButton("- (Show More)")
        self.zoom_out_btn.clicked.connect(self.on_zoom_out_clicked)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_in_btn = QPushButton("+ (Show Less)")
        self.zoom_in_btn.clicked.connect(self.on_zoom_in_clicked)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        spec_layout.addLayout(zoom_layout)
        
        # Current epochs shown
        self.spec_epochs_label = QLabel(f"Showing {self.spectrogram_epochs_to_show} epochs")
        self.spec_epochs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spec_layout.addWidget(self.spec_epochs_label)
        
        layout.addWidget(spec_group)
        
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
        
    def update_epoch_displays(self, update_label_view=True):
        """Update all widgets to show current epoch.
        
        Args:
            update_label_view: If False, skip updating label widget view
                             (used when label change originates from label widget)
        """
        # Update all widgets with the same current epoch
        self.mosaic_plotter.set_epoch(self.current_epoch)
        self.spectrogram_widget.set_epoch(self.current_epoch)
        
        # Only update label view if requested (skip when change comes from label widget)
        if update_label_view:
            self.label_widget.set_epoch(self.current_epoch)
        
        # Only update topography if enabled
        if self.topo_enabled:
            self.topo_widget.set_epoch(self.current_epoch)
            
            # Extract epoch data and pass to topography widget
            epoch_data = self.get_epoch_data(self.current_epoch)
            if epoch_data is not None:
                self.topo_widget.set_data(epoch_data, self.sampling_rate, self.epoch_length)
        
    def on_epoch_selected(self, epoch_idx):
        """Handle epoch selection from mosaic plotter."""
        self.current_epoch = epoch_idx
        self.update_epoch_displays()
        
    def on_label_changed(self, epoch_idx, score):
        """Handle label change from label widget.
        
        This is called whenever:
        - User labels an epoch (0-8 key press)
        - User navigates with arrow keys
        - User clicks on an epoch in the label widget
        
        It ensures all widgets (mosaic, spectrogram, topography) stay synchronized
        with the current epoch shown in the label widget.
        """
        print(f"Epoch {epoch_idx} - Racine score {score}")
        # Sync current epoch and update displays (but skip label view update to prevent auto-shift)
        self.current_epoch = epoch_idx
        self.update_epoch_displays(update_label_view=False)
    
    def on_epoch_double_clicked(self, epoch_idx):
        """Handle double-click on epoch in mosaic plotter - switch to video tab."""
        self.current_epoch = epoch_idx
        self.update_epoch_displays()
        # Switch to video tab and seek to epoch
        self.switch_to_video_tab(epoch_idx)
        
    def on_spacebar_pressed(self, epoch_idx):
        """Handle spacebar press in label widget - switch to video tab."""
        # Switch to video tab and seek to current epoch
        self.switch_to_video_tab(epoch_idx)
        
    def on_return_to_label(self):
        """Handle return to label button in video widget."""
        self.tab_widget.setCurrentIndex(0)  # Switch to Label tab
        # Set focus back to label widget for keyboard navigation
        self.label_widget.setFocus()
        
    def switch_to_video_tab(self, epoch_idx):
        """Switch to video tab and seek to the given epoch."""
        # Update video widget with current epoch
        self.video_widget.seek_to_epoch(epoch_idx)
        # Switch to Video tab (index 1)
        self.tab_widget.setCurrentIndex(1)
        print(f"Switched to Video tab at epoch {epoch_idx}")
    
    def on_mosaic_epochs_changed(self, value):
        """Handle mosaic epochs to show change."""
        self.mosaic_epochs_to_show = int(value)
        self.mosaic_plotter.set_epochs_to_show(self.mosaic_epochs_to_show)
        # Also update label widget for keyboard navigation
        self.label_widget.set_mosaic_epochs_to_show(self.mosaic_epochs_to_show)
        
    def on_label_epochs_changed(self, value):
        """Handle label epochs to show change."""
        self.label_epochs_to_show = int(value)
        self.label_widget.set_epochs_to_show(self.label_epochs_to_show)
    
    def on_topo_toggle_clicked(self, checked):
        """Handle topography enable/disable toggle."""
        if checked:
            # Disable topography
            self.topo_enabled = False
            self.topo_toggle_btn.setText("Enable Topography")
            # Clear the topography display
            self.topo_widget.clear_display()
            print("Topography updates disabled for faster labeling")
        else:
            # Enable topography
            self.topo_enabled = True
            self.topo_toggle_btn.setText("Disable Topography")
            # Update with current epoch
            self.update_epoch_displays()
            print("Topography updates enabled")
        
    def on_brighter_clicked(self):
        """Make spectrogram brighter by adjusting color limits."""
        # Reset acceleration counter
        self.zoom_acceleration = 1
        
        # Adjust brightness (reduce dynamic range)
        if self.spectrogram_vmax is None:
            # Get current auto values from spectrogram
            self.spectrogram_vmin, self.spectrogram_vmax = self.spectrogram_widget.get_color_limits()
        
        if self.spectrogram_vmin is not None and self.spectrogram_vmax is not None:
            range_val = self.spectrogram_vmax - self.spectrogram_vmin
            # Reduce max, making bright areas dimmer (overall brighter look)
            self.spectrogram_vmax -= range_val * 0.1
            self.spectrogram_widget.set_color_limits(self.spectrogram_vmin, self.spectrogram_vmax)
    
    def on_dimmer_clicked(self):
        """Make spectrogram dimmer by adjusting color limits."""
        # Reset acceleration counter
        self.zoom_acceleration = 1
        
        # Adjust brightness (increase dynamic range)
        if self.spectrogram_vmax is None:
            # Get current auto values from spectrogram
            self.spectrogram_vmin, self.spectrogram_vmax = self.spectrogram_widget.get_color_limits()
        
        if self.spectrogram_vmin is not None and self.spectrogram_vmax is not None:
            range_val = self.spectrogram_vmax - self.spectrogram_vmin
            # Increase max, making bright areas brighter (overall dimmer look)
            self.spectrogram_vmax += range_val * 0.1
            self.spectrogram_widget.set_color_limits(self.spectrogram_vmin, self.spectrogram_vmax)
    
    def on_zoom_in_clicked(self):
        """Zoom in - show fewer epochs (more detail)."""
        import time
        current_time = time.time()
        
        # Check if this is a continuous click (within threshold)
        if (current_time - self.last_zoom_time) <= self.zoom_threshold:
            # Continuous click - use exponential acceleration
            # Formula designed so ~10 clicks goes from full view to 1 epoch
            n_epochs = self.get_n_epochs()
            if n_epochs > 1:
                # Calculate step size based on current position and acceleration
                # Use exponential growth: step = max(1, current_epochs * factor)
                factor = 0.25 * self.zoom_acceleration  # Acceleration factor
                step = max(1, int(self.spectrogram_epochs_to_show * factor))
            else:
                step = 1
        else:
            # Non-continuous click - linear step
            step = 1
            self.zoom_acceleration = 1  # Reset acceleration
        
        self.last_zoom_time = current_time
        
        new_epochs = max(1, self.spectrogram_epochs_to_show - step)
        if new_epochs != self.spectrogram_epochs_to_show:
            self.spectrogram_epochs_to_show = new_epochs
            self.spectrogram_widget.set_epochs_to_show(self.spectrogram_epochs_to_show)
            self.spec_epochs_label.setText(f"Showing {self.spectrogram_epochs_to_show} epochs")
            # Increase acceleration for consecutive clicks
            self.zoom_acceleration = min(3.0, self.zoom_acceleration + 0.3)
        else:
            # Can't zoom more, reset acceleration
            self.zoom_acceleration = 1
    
    def on_zoom_out_clicked(self):
        """Zoom out - show more epochs (wider view)."""
        import time
        current_time = time.time()
        
        # Check if this is a continuous click (within threshold)
        if (current_time - self.last_zoom_time) <= self.zoom_threshold:
            # Continuous click - use exponential acceleration
            # Formula designed so ~10 clicks goes from 1 epoch to full view
            n_epochs = self.get_n_epochs()
            if n_epochs > 1:
                # Calculate step size based on remaining distance and acceleration
                remaining = n_epochs - self.spectrogram_epochs_to_show
                factor = 0.25 * self.zoom_acceleration  # Acceleration factor
                step = max(1, int(self.spectrogram_epochs_to_show * factor))
            else:
                step = 1
        else:
            # Non-continuous click - linear step
            step = 1
            self.zoom_acceleration = 1  # Reset acceleration
        
        self.last_zoom_time = current_time
        
        n_epochs = self.get_n_epochs()
        new_epochs = min(n_epochs, self.spectrogram_epochs_to_show + step)
        if new_epochs != self.spectrogram_epochs_to_show:
            self.spectrogram_epochs_to_show = new_epochs
            self.spectrogram_widget.set_epochs_to_show(self.spectrogram_epochs_to_show)
            self.spec_epochs_label.setText(f"Showing {self.spectrogram_epochs_to_show} epochs")
            # Increase acceleration for consecutive clicks
            self.zoom_acceleration = min(3.0, self.zoom_acceleration + 0.3)
        else:
            # Can't zoom more, reset acceleration
            self.zoom_acceleration = 1
    
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
            
            # Update status indicator
            self.update_label_status(True)
            
            self.show_message_box(QMessageBox.Icon.Information, "Create Success", 
                                   f"Created new label file with {n_epochs} epochs:\n{file_path}\n\n"
                                   f"All labels initialized to 0 (no seizure).")
            print(f"Created label file: {self.label_file}")
            
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, "Create Error", f"Failed to create label file:\n{str(e)}")
            self.label_file = None
            self.update_label_status(False)
            
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
                        self.show_message_box(QMessageBox.Icon.Information, "Import Success", 
                                               f"Loaded {len(labels)} existing labels from file.")
                    else:
                        # Initialize with current labels
                        self.label_widget.initialize_labels(n_epochs)
                        self.show_message_box(QMessageBox.Icon.Warning, "Label Count Mismatch", 
                                           f"File has {len(labels)} labels but data has {n_epochs} epochs.\n"
                                           f"Initialized with empty labels instead.")
                else:
                    # CSV exists but no label column - initialize empty labels
                    n_epochs = self.get_n_epochs()
                    self.label_widget.initialize_labels(n_epochs)
                    self.show_message_box(QMessageBox.Icon.Information, "Import Success", 
                                           f"Imported file (no existing labels).\nReady to label {n_epochs} epochs.")
            
            # Update status indicator
            self.update_label_status(True)
            print(f"Label file set to: {self.label_file}")
            
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, "Import Error", f"Failed to import file:\n{str(e)}")
            self.label_file = None
            self.update_label_status(False)
    
    def save_labels(self):
        """Save current labels to the imported CSV file."""
        if not self.label_file:
            self.show_message_box(QMessageBox.Icon.Warning, "No Label File", 
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
            
            self.show_message_box(QMessageBox.Icon.Information, "Save Success", 
                                   f"Saved {n_epochs} labels to:\n{self.label_file}")
            print(f"Labels saved to: {self.label_file}")
            
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, "Save Error", f"Failed to save labels:\n{str(e)}")
    
    def update_label_status(self, is_loaded):
        """Update the label status indicator.
        
        Parameters:
        -----------
        is_loaded : bool
            True if label file is loaded/imported, False if empty
        """
        if is_loaded:
            self.label_status_indicator.setText("Imported")
            self.label_status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #388e3c;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 10px;
                }
            """)
        else:
            self.label_status_indicator.setText("Empty")
            self.label_status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #d32f2f;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 10px;
                }
            """)

    # ----------------------------------------------------------------------
    # Save Functionality Handlers
    # ----------------------------------------------------------------------
    
    def save_topography_image(self):
        """Save the current topography as a white-background image."""
        try:
            # 1. Ask for file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Topography Image", "", 
                "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Ask if user wants to include channel labels
            reply = QMessageBox.question(
                self, "Include Labels?", 
                "Do you want to include channel name labels in the image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            show_labels = (reply == QMessageBox.StandardButton.Yes)
            
            # 2. Create temporary figure with white background
            # 5x5 inches, decent resolution
            fig = Figure(figsize=(6, 5), facecolor='white')
            
            # Create subplots grid: Main plot (left) and Colorbar (right)
            # Adjust widths to look good
            gs = fig.add_gridspec(1, 2, width_ratios=[15, 1], wspace=0.1, 
                                left=0.05, right=0.9, top=0.9, bottom=0.1)
            
            ax = fig.add_subplot(gs[0])
            cbar_ax = fig.add_subplot(gs[1])
            
            # 3. Plot data
            # Use white/black theme for export
            export_theme = {'bg_primary': '#ffffff', 'fg_primary': '#000000'}
            
            # Get current power values
            self.topo_widget.plot_power_on_head(
                self.topo_widget.compute_psd_power(self.topo_widget.epoch_data, self.topo_widget.current_freq_band),
                ax=ax,
                colorbar_ax=cbar_ax,
                theme_colors=export_theme,
                show_labels=show_labels
            )
            
            # 4. Save
            canvas = FigureCanvas(fig)
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.show_message_box(QMessageBox.Icon.Information, 
                                "Success", f"Topography saved to:\n{file_path}")
            
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, 
                                "Error", f"Failed to save topography:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_mosaic_plot(self):
        """Save the mosaic plot for a specific range."""
        try:
            n_epochs = self.get_n_epochs()
            if n_epochs == 0:
                print("No epochs to save.")
                return
                
            # 1. Ask for Start Epoch and Duration
            dialog = QDialog(self)
            dialog.setWindowTitle("Save Mosaic Plot")
            layout = QFormLayout(dialog)
            
            start_spin = QSpinBox()
            start_spin.setRange(0, n_epochs - 1)
            start_spin.setValue(self.current_epoch)
            layout.addRow("Start Epoch:", start_spin)
            
            duration_spin = QSpinBox()
            duration_spin.setRange(1, n_epochs)
            duration_spin.setValue(15)  # Default duration
            layout.addRow("Duration (epochs):", duration_spin)
            
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                     QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addRow(buttons)
            
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            start_epoch = start_spin.value()
            duration = duration_spin.value()
            
            # 2. Ask for file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Mosaic Plot", "", 
                "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf)"
            )
            
            if not file_path:
                return
                
            # 3. Create temporary figure
            # Width depends on duration, height on vertical size
            # Approx 1 second per inch width might be good, or fixed size
            width = max(10, min(duration, 30)) # dynamic width
            fig = Figure(figsize=(width, 8), facecolor='white')
            ax = fig.add_subplot(111)
            
            # 4. Plot data
            self.mosaic_plotter.plot_data(
                ax=ax,
                start_epoch=start_epoch,
                n_epochs_to_show=duration,
                theme_colors={'bg_primary': '#ffffff', 'fg_primary': '#000000'}
            )
            
            # 5. Save
            canvas = FigureCanvas(fig)
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            self.show_message_box(QMessageBox.Icon.Information, 
                                "Success", f"Mosaic plot saved to:\n{file_path}")
                                
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, 
                                "Error", f"Failed to save mosaic plot:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_power_analysis(self):
        """Export power analysis for all epochs."""
        try:
            if self.df.empty:
                return
                
            # 1. Ask for Frequency Band
            bands = list(self.topo_widget.FREQ_BANDS.keys()) + ["Custom"]
            band_name, ok = QInputDialog.getItem(
                self, "Select Band", 
                "Frequency Band:", bands, 0, False
            )
            
            if not ok or not band_name:
                return
                
            # Determine frequency range
            if band_name == "Custom":
                # We need a custom dialog for this, reusing topowindow's dialog?
                # For simplicity, let's just ask for Low and High HZ
                low, ok1 = QInputDialog.getDouble(self, "Custom Band", "Low Frequency (Hz):", 5, 0.1, 200, 1)
                if not ok1: return
                high, ok2 = QInputDialog.getDouble(self, "Custom Band", "High Frequency (Hz):", 15, 0.1, 200, 1)
                if not ok2: return
                freq_band = (low, high)
            else:
                info = self.topo_widget.FREQ_BANDS[band_name]
                freq_band = (info[0], info[1])
            
            # 2. Ask for file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Power Analysis", "", 
                "CSV File (*.csv)"
            )
            
            if not file_path:
                return
                
            # 3. Compute power for all epochs
            n_epochs = self.get_n_epochs()
            results = []
            
            # Simple progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Computing power analysis...", "Cancel", 0, n_epochs, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            for i in range(n_epochs):
                if progress.wasCanceled():
                    return
                progress.setValue(i)
                
                # Get epoch data
                epoch_data = self.get_epoch_data(i)
                if epoch_data is None:
                    continue
                
                powers = self.topo_widget.compute_psd_power(epoch_data, freq_band)
                
                # Add epoch index
                powers['epoch'] = i
                results.append(powers)
            
            progress.setValue(n_epochs)
            
            # 4. Create DataFrame and Save
            if not results:
                self.show_message_box(QMessageBox.Icon.Warning, "No Data", "No power data computed.")
                return

            df_results = pd.DataFrame(results)
            
            # Reorder columns: epoch first, then electrodes sorted
            cols = ['epoch'] + sorted([c for c in df_results.columns if c != 'epoch'])
            df_results = df_results[cols]
            
            df_results.to_csv(file_path, index=False)
            
            self.show_message_box(QMessageBox.Icon.Information, 
                                "Success", f"Power analysis saved to:\n{file_path}\n\n"
                                          f"Band: {band_name} {freq_band} Hz\n"
                                          f"Epochs: {len(df_results)}")
                                          
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, 
                                "Error", f"Failed to save analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def save_file_to_db(self):
        """Save ALL epochs data to CSV files for DB import."""
        try:
            if self.df.empty:
                self.show_message_box(QMessageBox.Icon.Warning, "No Data", "No data available to save.")
                return

            # 1. Ask for Data ID (e.g. M25001)
            data_id, ok = QInputDialog.getText(
                self, "File to DB", 
                "Data ID (ex. M1AP01):"
            )
            if not ok or not data_id:
                return

            # 2. Check if ExtraData exists (2-digit portion at position 4-5)
            # and show memo dialog if it does
            memo_text = ""
            if len(data_id) >= 6:
                # ExtraData exists, show memo dialog
                memo_text, ok = QInputDialog.getMultiLineText(
                    self, "Add Memo",
                    "Enter notes/memo for this recording:\n(ExtraData detected)",
                    ""
                )
                if not ok:
                    return
            
            # 3. Ask for the recording date
            date_str, ok = QInputDialog.getText(
                self, "Recording Date",
                "Enter recording date (ex. 2026-01-20):",
                text=""
            )
            if not ok:
                return

            # 4. Ask for float approximation digits (precision)
            precision, ok = QInputDialog.getInt(
                self, "File to DB",
                "Decimal Precision (digits under .):",
                4, 0, 20, 1
            )
            if not ok:
                return

            # 5. Ask for output directory
            directory = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for CSV Files"
            )
            if not directory:
                return

            # Get total epochs
            n_epochs = self.get_n_epochs()
            if n_epochs == 0:
                 self.show_message_box(QMessageBox.Icon.Warning, "No Data", "No epochs found.")
                 return

            # Progress Dialog
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Saving all epochs to CSV...", "Cancel", 0, n_epochs, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            saved_count = 0
            from pathlib import Path
            save_dir = Path(directory)
            
            # Loop through ALL epochs
            for epoch_idx in range(n_epochs):
                if progress.wasCanceled():
                    break
                
                progress.setValue(epoch_idx)
                
                # Get epoch data
                epoch_data = self.get_epoch_data(epoch_idx)
                if epoch_data is None:
                    continue
                
                # Format epoch number string
                epoch_str = f"EP{epoch_idx:06d}"
                
                for channel_name in epoch_data.columns:
                    # Construct filename: DataID + ChannelName + EP + number
                    file_name = f"{data_id}_{channel_name}_{epoch_str}.csv"
                    file_path = save_dir / file_name
                    
                    # Get data for this channel
                    channel_data = epoch_data[channel_name]
                    
                    # Round
                    channel_data_rounded = channel_data.round(precision)
                    
                    # Save
                    channel_data_rounded.to_csv(file_path, index=False, header=False)
                    saved_count += 1
            
            progress.setValue(n_epochs)

            # Save Labels with custom header
            label_file = save_dir / f"{data_id}_labels.csv"
            
            try:
                # Get current labels from widget
                labels = self.label_widget.get_labels()
                if len(labels) != n_epochs:
                     # Fallback if label widget not sync (shouldn't happen)
                     labels = [0] * n_epochs
                
                # Create DataFrame with 'epoch' and custom label column name
                df_labels = pd.DataFrame({
                    'epoch': np.arange(n_epochs),
                    data_id: labels
                })
                
                # Save label file
                df_labels.to_csv(label_file, index=False)
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving label file: {e}")
            
            # Save Metadata JSON
            try:
                import json
                from .create_meta import create_meta
                
                # Get activated channel names (sorted alphabetically)
                activated_channels = list(self.df.columns)
                
                # Create metadata dictionary
                meta_data = create_meta(
                    name=data_id,
                    memo=memo_text,
                    date=date_str,
                    activated_channels=activated_channels
                )
                
                # Save as JSON
                meta_file = save_dir / f"{data_id}_metadata.json"
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
                saved_count += 1
                print(f"Metadata saved to: {meta_file}")
                
            except Exception as e:
                print(f"Error saving metadata file: {e}")
                
            if saved_count > 0:
                self.show_message_box(QMessageBox.Icon.Information, 
                                    "Success", 
                                    f"Saved {saved_count} files to:\n{directory}\n\n"
                                    f"Covered {n_epochs} epochs + Labels + Metadata.")
                
                # Cache Deletion Option
                if self.cache_key:
                    reply = QMessageBox.question(
                        self, "Delete Cache?", 
                        "Do you want to delete the original cache to save space?\n"
                        "Since data is now saved to DB format, the cache might be redundant.\n\n"
                        "WARNING: This action cannot be undone!",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                        QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        try:
                            from utils.cache_manager import cache_manager
                            cache_manager.remove_cached_project(self.cache_key)
                            self.show_message_box(QMessageBox.Icon.Information, "Cache Deleted", 
                                                "Original cache has been successfully removed.")
                        except Exception as e:
                            self.show_message_box(QMessageBox.Icon.Critical, "Error", 
                                                f"Failed to delete cache:\n{str(e)}")
            else:
                 self.show_message_box(QMessageBox.Icon.Warning, "Cancelled", "Operation cancelled or no files saved.")
            
        except Exception as e:
            self.show_message_box(QMessageBox.Icon.Critical, 
                                "Error", f"Failed to save to DB format:\n{str(e)}")
            import traceback
            traceback.print_exc()

