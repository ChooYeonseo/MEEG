"""
Main window for EEG Analysis GUI application.

This module contains the main window class and all GUI components
for the EEG data analysis tool.
"""

import os
import sys
import traceback
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QTextEdit, QLabel, QProgressBar,
                            QGroupBox, QFileDialog, QMessageBox, QSplitter,
                            QStatusBar, QMenuBar, QMenu)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QAction

from .data_reader_thread import DataReaderThread, CacheLoaderThread
from .output_capture import OutputCapture
from .cached_project_dialog import CachedProjectDialog
from .preprocessing_pipeline import PreprocessingPipelineDialog

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

# Import preferences system
sys.path.insert(0, str(current_dir))
from preferences import preferences_manager

try:
    import read_intan
except ImportError as e:
    print(f"Error importing read_intan utilities: {e}")
    traceback.print_exc()


class EEGMainWindow(QMainWindow):
    """Main GUI application window for EEG analysis."""
    
    def __init__(self, app=None):
        super().__init__()
        self.app = app  # Store reference to the application for theme switching
        self.current_data = None
        self.current_directory = None
        self.current_cache_key = None  # Store cache key for current data
        self.reader_thread = None
        self.electrode_positions = None  # Track electrode positions
        
        # Set up output capture
        self.output_capture = OutputCapture()
        self.output_capture.output_received.connect(self.append_to_terminal)
        
        self.init_ui()
        self.setup_styles()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("EEG Data Analysis Tool")
        self.setGeometry(100, 100, 1000, 700)
        
        # Set up menu bar
        self.setup_menu_bar()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("MEEG - Mouse EEG Analysis Tool")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Control panel
        control_group = QGroupBox("Data Loading")
        control_layout = QVBoxLayout(control_group)
        
        # Button and directory selection
        button_layout = QHBoxLayout()
        
        self.read_button = QPushButton("📁 Select Folder & Read Data")
        self.read_button.setMinimumHeight(40)
        self.read_button.clicked.connect(self.select_and_read_data)
        button_layout.addWidget(self.read_button)
        
        self.load_cached_button = QPushButton("💾 Load Existing Project")
        self.load_cached_button.setMinimumHeight(40)
        self.load_cached_button.clicked.connect(self.load_cached_project)
        button_layout.addWidget(self.load_cached_button)
        
        self.visualize_button = QPushButton("📊 Visualize Data")
        self.visualize_button.setMinimumHeight(40)
        self.visualize_button.clicked.connect(self.open_plotting_window)
        self.visualize_button.setEnabled(False)
        button_layout.addWidget(self.visualize_button)
        
        self.electrode_plot_button = QPushButton("🧠 Electrode Plotting")
        self.electrode_plot_button.setMinimumHeight(40)
        self.electrode_plot_button.clicked.connect(self.open_electrode_plotting_window)
        button_layout.addWidget(self.electrode_plot_button)
        
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        # Directory label
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("color: gray; font-style: italic;")
        control_layout.addWidget(self.dir_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(control_group)
        
        # Splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Data information section
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        info_layout.addWidget(self.info_text)
        
        splitter.addWidget(info_group)
        
        # Terminal output section
        terminal_group = QGroupBox("Terminal Output")
        terminal_layout = QVBoxLayout(terminal_group)
        
        self.terminal_text = QTextEdit()
        self.terminal_text.setReadOnly(True)
        self.terminal_text.setMinimumHeight(200)
        terminal_layout.addWidget(self.terminal_text)
        
        # Terminal controls
        terminal_controls_layout = QHBoxLayout()
        
        # Clear terminal button
        clear_button = QPushButton("Clear Terminal")
        clear_button.clicked.connect(self.clear_terminal)
        terminal_controls_layout.addWidget(clear_button)
        
        # MEEG Analysis button
        self.meeg_analysis_button = QPushButton("🧠 MEEG Analysis")
        self.meeg_analysis_button.clicked.connect(self.open_meeg_analysis_window)
        self.meeg_analysis_button.setEnabled(False)  # Initially disabled
        terminal_controls_layout.addWidget(self.meeg_analysis_button)
        
        terminal_layout.addLayout(terminal_controls_layout)
        
        splitter.addWidget(terminal_group)
        
        # Set splitter proportions
        splitter.setSizes([300, 300])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)
        
    def setup_menu_bar(self):
        """Set up the application menu bar."""
        menubar = self.menuBar()
        
        # Preferences menu
        preferences_menu = menubar.addMenu('Preferences')
        
        # Theme submenu
        theme_menu = preferences_menu.addMenu('Theme')
        
        # Normal theme action
        normal_theme_action = QAction('Normal Theme', self)
        normal_theme_action.setCheckable(True)
        normal_theme_action.triggered.connect(lambda: self.change_theme('normal'))
        theme_menu.addAction(normal_theme_action)
        
        # Tokyo Night theme action
        tokyo_night_action = QAction('Tokyo Night', self)
        tokyo_night_action.setCheckable(True)
        tokyo_night_action.triggered.connect(lambda: self.change_theme('tokyo_night'))
        theme_menu.addAction(tokyo_night_action)
        
        # Store actions for later reference
        self.theme_actions = {
            'normal': normal_theme_action,
            'tokyo_night': tokyo_night_action
        }
        
        # Set initial theme state
        current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        self.update_theme_menu_state(current_theme)
        
        # Connect to preferences changes
        preferences_manager.theme_changed.connect(self.on_theme_changed)
        
    def change_theme(self, theme_name):
        """Change the application theme."""
        preferences_manager.set_setting('theme', theme_name)
        
    def update_theme_menu_state(self, current_theme):
        """Update the menu state to reflect the current theme."""
        for theme, action in self.theme_actions.items():
            action.setChecked(theme == current_theme)
            
    def on_theme_changed(self, theme_name):
        """Handle theme changes from preferences."""
        self.update_theme_menu_state(theme_name)
        
        # Apply the theme
        if self.app:  # Only apply if we have app reference
            if theme_name == 'tokyo_night':
                from tokyo_night_theme import apply_tokyo_night_theme
                apply_tokyo_night_theme(self.app)
            else:  # normal theme
                from normal_theme import apply_normal_theme
                apply_normal_theme(self.app)
        
        # Reapply component-specific styles
        self.setup_styles()
        
    def setup_styles(self):
        """Set up custom styles for the application."""
        # Import Tokyo Night theme
        from tokyo_night_theme import TOKYO_NIGHT_STYLES, TOKYO_NIGHT_COLORS
        
        # Apply specific component styles
        # Note: Main theme is applied globally, these are component-specific overrides
        
        # Terminal styling with Tokyo Night colors
        self.terminal_text.setStyleSheet(TOKYO_NIGHT_STYLES['terminal'])
        
        # Button styling
        self.read_button.setStyleSheet(TOKYO_NIGHT_STYLES['button_primary'])
        self.load_cached_button.setStyleSheet(TOKYO_NIGHT_STYLES['button_primary'])
        self.visualize_button.setStyleSheet(TOKYO_NIGHT_STYLES['button_primary'])
        
    def select_and_read_data(self):
        """Select directory and start data reading process."""
        if self.reader_thread and self.reader_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Data reading is already in progress!")
            return
            
        # Open directory selection dialog
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing RHD files",
            self.current_directory or os.path.expanduser("~")
        )
        
        if not directory:
            self.status_bar.showMessage("No directory selected")
            return
            
        self.current_directory = directory
        self.dir_label.setText(f"Selected: {directory}")
        self.dir_label.setStyleSheet("color: blue;")
        
        # Start reading data in separate thread
        self.start_data_reading(directory)
        
    def load_cached_project(self):
        """Load a cached project."""
        if self.reader_thread and self.reader_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Data reading is already in progress!")
            return
            
        # Show cached project selection dialog
        dialog = CachedProjectDialog(self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            cache_key = dialog.get_selected_cache_key()
            if cache_key:
                self.start_cache_loading(cache_key)
        
    def start_cache_loading(self, cache_key):
        """Start loading cached data in a separate thread."""
        self.current_cache_key = cache_key  # Store cache key
        self.read_button.setEnabled(False)
        self.load_cached_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage("Loading cached project...")
        
        # Create and start cache loader thread
        self.reader_thread = CacheLoaderThread(cache_key)
        self.reader_thread.set_output_capture(self.output_capture)
        
        # Connect signals
        self.reader_thread.progress_update.connect(self.update_progress)
        self.reader_thread.data_loaded.connect(self.on_data_loaded)
        self.reader_thread.error_occurred.connect(self.on_error_occurred)
        self.reader_thread.finished_loading.connect(self.on_loading_finished)
        
        self.reader_thread.start()
        
    def start_data_reading(self, directory):
        """Start the data reading process in a separate thread."""
        self.read_button.setEnabled(False)
        self.load_cached_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage("Selecting directory...")
        
        # Create and start reader thread (without saving to cache yet)
        self.reader_thread = DataReaderThread(directory, use_cache=True, save_cache=False)
        self.reader_thread.set_output_capture(self.output_capture)
        
        # Connect signals
        self.reader_thread.progress_update.connect(self.update_progress)
        self.reader_thread.data_loaded.connect(self.on_raw_data_loaded)
        self.reader_thread.error_occurred.connect(self.on_error_occurred)
        self.reader_thread.finished_loading.connect(self.on_loading_finished)
        
        self.reader_thread.start()
        
    def on_raw_data_loaded(self, results):
        """Handle raw data loading before preprocessing."""
        if not results:
            self.update_info_display("No RHD data found.")
            self.status_bar.showMessage("No data found")
            self.visualize_button.setEnabled(False)
            return
        
        # Show preprocessing pipeline dialog
        preprocessing_dialog = PreprocessingPipelineDialog(
            results, 
            self.current_directory, 
            self
        )
        
        if preprocessing_dialog.exec() == preprocessing_dialog.DialogCode.Accepted:
            # Get processed data
            processed_data = preprocessing_dialog.get_processed_data()
            if processed_data:
                self.on_data_loaded(processed_data)
            else:
                # If no processed data, use original data and cache it
                self.cache_original_data(results)
        else:
            # User cancelled preprocessing, ask if they want to cache original data
            reply = QMessageBox.question(
                self,
                "Cache Original Data?",
                "Would you like to cache the original (unprocessed) data for faster future access?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.cache_original_data(results)
            
            self.on_data_loaded(results)
    
    def cache_original_data(self, results):
        """Cache the original data without preprocessing."""
        try:
            # Add utils directory to path for cache_manager
            current_dir = Path(__file__).parent.parent
            utils_dir = current_dir / "utils"
            import sys
            sys.path.insert(0, str(utils_dir))
            from cache_manager import cache_manager
            
            self.status_bar.showMessage("Caching original data...")
            cache_manager.save_to_cache(self.current_directory, results)
            self.status_bar.showMessage("Original data cached successfully")
        except Exception as e:
            QMessageBox.warning(self, "Cache Error", f"Failed to cache data: {str(e)}")
        
    def update_progress(self, message):
        """Update progress status."""
        self.status_bar.showMessage(message)
        
    def on_data_loaded(self, results):
        """Handle successful data loading."""
        self.current_data = results
        
        if not results:
            self.update_info_display("No RHD data found.")
            self.status_bar.showMessage("No data found")
            self.visualize_button.setEnabled(False)
        else:
            info_text = self.generate_data_summary(results)
            self.update_info_display(info_text)
            self.status_bar.showMessage(f"Loaded {len(results)} files successfully")
            # Enable visualization button when data is loaded
            self.visualize_button.setEnabled(True)
            # Check if we can enable MEEG Analysis button
            self.update_meeg_analysis_button_state()
            
    def on_error_occurred(self, error_message):
        """Handle errors during data loading."""
        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage("Error occurred")
        
    def on_loading_finished(self):
        """Handle completion of data loading process."""
        self.read_button.setEnabled(True)
        self.load_cached_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def append_to_terminal(self, text):
        """Append text to the terminal window."""
        self.terminal_text.append(text)
        # Auto-scroll to bottom
        cursor = self.terminal_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.terminal_text.setTextCursor(cursor)
        
    def clear_terminal(self):
        """Clear the terminal window."""
        self.terminal_text.clear()
        
    def update_info_display(self, info_text):
        """Update the data information display."""
        self.info_text.setPlainText(info_text)
        
    def generate_data_summary(self, results):
        """Generate a summary of the loaded data."""
        if not results:
            return "No data loaded."
            
        summary_lines = [
            f"Data Summary ({len(results)} files loaded)",
            "=" * 40,
            ""
        ]
        
        total_files_with_data = sum(1 for _, _, data_present in results if data_present)
        
        # Calculate total duration across all files
        total_duration = 0.0
        total_samples = 0
        sample_rate = None
        
        files_with_data = [(filename, result) for filename, result, data_present in results if data_present]
        
        for filename, result in files_with_data:
            try:
                if sample_rate is None:
                    sample_rate = read_intan.get_sample_rate(result)
                
                if 'amplifier_data' in result and sample_rate:
                    num_samples = result['amplifier_data'].shape[1]
                    file_duration = num_samples / sample_rate
                    total_duration += file_duration
                    total_samples += num_samples
            except Exception:
                pass
        
        summary_lines.extend([
            f"Total files: {len(results)}",
            f"Files with data: {total_files_with_data}",
            f"Files without data: {len(results) - total_files_with_data}",
            ""
        ])
        
        if total_files_with_data > 1:
            summary_lines.extend([
                "Concatenated Dataset:",
                "-" * 20,
                f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)",
                f"Total samples: {total_samples:,}",
                f"Sample rate: {sample_rate} Hz" if sample_rate else "Sample rate: Unknown",
                ""
            ])
        
        summary_lines.extend([
            "Individual File Details:",
            "-" * 20
        ])
        
        for filename, result, data_present in results:
            status = "✓ Data" if data_present else "✗ No data"
            summary_lines.append(f"{filename}: {status}")
            
            if data_present:
                # Try to get additional info about the file
                try:
                    sample_rate = read_intan.get_sample_rate(result)
                    if sample_rate:
                        summary_lines.append(f"  Sample rate: {sample_rate} Hz")
                        
                    # Count channels
                    if 'amplifier_data' in result:
                        num_channels = result['amplifier_data'].shape[0]
                        num_samples = result['amplifier_data'].shape[1]
                        duration = num_samples / sample_rate if sample_rate else "Unknown"
                        summary_lines.append(f"  Channels: {num_channels}")
                        summary_lines.append(f"  Samples: {num_samples}")
                        summary_lines.append(f"  Duration: {duration:.2f}s" if isinstance(duration, float) else f"  Duration: {duration}")
                        
                except Exception as e:
                    summary_lines.append(f"  Error getting details: {str(e)}")
                    
            summary_lines.append("")
            
        return "\n".join(summary_lines)
        
    def closeEvent(self, event):
        """Handle application closing."""
        if self.reader_thread and self.reader_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                'Quit', 
                'Data reading is in progress. Are you sure you want to quit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.reader_thread.terminate()
                self.reader_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            
    def open_plotting_window(self):
        """Open the data visualization window."""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Please load EEG data first.")
            return
            
        try:
            # Import here to avoid circular imports
            from .plotting_window import PlottingWindow
            
            # Create and show plotting window
            self.plotting_window = PlottingWindow(self.current_data)
            self.plotting_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening plotting window: {str(e)}")
    
    def open_electrode_plotting_window(self):
        """Open the electrode positioning window."""
        try:
            # Import here to avoid circular imports
            from .electrode_plotting_window import ElectrodePlottingWindow
            
            # Create and show electrode plotting window
            self.electrode_plotting_window = ElectrodePlottingWindow(main_window=self)
            self.electrode_plotting_window.positions_updated.connect(self.set_electrode_positions)
            self.electrode_plotting_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening electrode plotting window: {str(e)}")
    
    def update_meeg_analysis_button_state(self):
        """Update the state of the MEEG Analysis button based on data and electrode positions."""
        # Enable if both data and electrode positions are available
        has_data = self.current_data is not None
        has_electrodes = self.electrode_positions is not None
        self.meeg_analysis_button.setEnabled(has_data and has_electrodes)
    
    def set_electrode_positions(self, positions):
        """Set electrode positions and update button state."""
        self.electrode_positions = positions
        self.update_meeg_analysis_button_state()
    
    def open_meeg_analysis_window(self):
        """Open the MEEG analysis window for electrode mapping."""
        try:
            # Import here to avoid circular imports
            from .electrode_mapping_window import ElectrodeMappingWindow
            
            # Create and show electrode mapping window
            self.mapping_window = ElectrodeMappingWindow(
                electrode_positions=self.electrode_positions,
                current_data=self.current_data,
                cache_directory=str(Path(__file__).parent.parent / "cache"),
                cache_key=self.current_cache_key,
                directory_path=self.current_directory
            )
            self.mapping_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening MEEG analysis window: {str(e)}")
