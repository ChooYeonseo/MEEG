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
                            QStatusBar, QMenuBar, QMenu, QApplication)
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
from theme import preferences_manager

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
        
        # Check for updates in background (silent on startup)
        self._setup_auto_updater()
    
    def _setup_auto_updater(self):
        """Initialize auto-update checking on startup."""
        try:
            from utils.auto_updater import check_for_updates_on_startup
            # Keep reference to prevent garbage collection
            self._updater = check_for_updates_on_startup(parent=self, silent=True)
        except ImportError:
            # Auto-updater not available (requests not installed)
            pass
        except Exception as e:
            print(f"Auto-update check failed: {e}")
        
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("MEEG")
        #self.setGeometry(100, 100, 1000, 700) # this is the inital window size
        self.showMaximized()
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
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use inline stylesheet to prevent theme from overriding
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Control panel
        control_group = QGroupBox("Data Loading")
        control_layout = QVBoxLayout(control_group)
        
        # Button and directory selection
        button_layout = QHBoxLayout()
        
        self.read_button = QPushButton("📁 Select Folder && Read Data")
        self.read_button.setMinimumHeight(40)
        self.read_button.clicked.connect(self.select_and_read_data)
        button_layout.addWidget(self.read_button)
        
        self.load_cached_button = QPushButton("💾 Load Existing Project")
        self.load_cached_button.setMinimumHeight(40)
        self.load_cached_button.clicked.connect(self.load_cached_project)
        button_layout.addWidget(self.load_cached_button)
        
        self.visualize_button = QPushButton("Data Preview")
        self.visualize_button.setMinimumHeight(40)
        self.visualize_button.clicked.connect(self.open_plotting_window)
        self.visualize_button.setEnabled(False)
        button_layout.addWidget(self.visualize_button)
        
        self.electrode_plot_button = QPushButton("Electrode Configuration")
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
        
        # MEEG menu (main menu - will show as application menu on macOS)
        meeg_menu = menubar.addMenu('MEEG')
        
        # About action
        about_action = QAction('About', self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)  # macOS will move this to app menu
        about_action.triggered.connect(self.show_about_dialog)
        meeg_menu.addAction(about_action)
        
        # Separator
        meeg_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setMenuRole(QAction.MenuRole.QuitRole)  # macOS will move this to app menu
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.exit_application)
        meeg_menu.addAction(exit_action)
        
        # Preferences menu (separate top-level menu for better visibility)
        preferences_menu = menubar.addMenu('Preferences')
        
        # Theme submenu under Preferences
        theme_menu = preferences_menu.addMenu('Theme')
        
        # Bright theme action (formerly Normal theme)
        bright_theme_action = QAction('Bright Theme', self)
        bright_theme_action.setCheckable(True)
        bright_theme_action.triggered.connect(lambda: self.change_theme('normal'))
        theme_menu.addAction(bright_theme_action)
        
        # Dark mode theme action
        dark_theme_action = QAction('Dark Mode', self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.triggered.connect(lambda: self.change_theme('dark'))
        theme_menu.addAction(dark_theme_action)
        
        # Tokyo Night theme action
        tokyo_night_action = QAction('Tokyo Night', self)
        tokyo_night_action.setCheckable(True)
        tokyo_night_action.triggered.connect(lambda: self.change_theme('tokyo_night'))
        theme_menu.addAction(tokyo_night_action)
        
        # Store actions for later reference
        self.theme_actions = {
            'normal': bright_theme_action,
            'dark': dark_theme_action,
            'tokyo_night': tokyo_night_action
        }
        
        # Set initial theme state
        current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        self.update_theme_menu_state(current_theme)
        
        # Connect to preferences changes
        preferences_manager.theme_changed.connect(self.on_theme_changed)
        
    def change_theme(self, theme_name):
        """Change the application theme."""
        # Save the preference
        preferences_manager.set_setting('theme', theme_name)
        
        # Apply the theme immediately
        self.apply_theme(theme_name)
        
    def apply_theme(self, theme_name):
        """Apply the specified theme to the application."""
        # Update menu state
        self.update_theme_menu_state(theme_name)
        
        # Apply the theme
        if self.app:  # Only apply if we have app reference
            if theme_name == 'tokyo_night':
                from theme import apply_tokyo_night_theme
                apply_tokyo_night_theme(self.app)
            elif theme_name == 'dark':
                from theme import apply_dark_theme
                apply_dark_theme(self.app)
            else:  # normal theme
                from theme import apply_normal_theme
                apply_normal_theme(self.app)
        
        # Reapply component-specific styles
        self.setup_styles()
        
    def update_theme_menu_state(self, current_theme):
        """Update the menu state to reflect the current theme."""
        for theme, action in self.theme_actions.items():
            action.setChecked(theme == current_theme)
            
    def on_theme_changed(self, theme_name):
        """Handle theme changes from preferences (when changed externally)."""
        self.apply_theme(theme_name)
    
    def show_about_dialog(self):
        """Display the About dialog with program information."""
        about_text = """
        <h2>MEEG</h2>
        <h3>Mouse EEG Analysis Tool</h3>
        <p><b>Version:</b> 1.0</p>
        <p><b>Build Date:</b> November 2025</p>
        <p><b>Developed by:</b> Yeonseo (Sean) Choo</p>
        <p><b>Affiliation:</b> Korea University, College of Medicine</p>
        <hr>
        <p>A comprehensive tool for analyzing mouse EEG data with advanced visualization and processing capabilities.</p>
        """
        
        QMessageBox.about(self, "About MEEG", about_text)
    
    def exit_application(self):
        """Exit the application."""
        # Clean up any running threads
        if self.reader_thread and self.reader_thread.isRunning():
            self.reader_thread.quit()
            self.reader_thread.wait()
        
        # Close the application
        QApplication.quit()
        
    def setup_styles(self):
        """Set up custom styles for the application."""
        # Get current theme
        current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        # Import appropriate theme styles
        if current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_STYLES as THEME_STYLES
        elif current_theme == 'dark':
            from theme import DARK_THEME_STYLES as THEME_STYLES
        else:  # normal
            from theme import NORMAL_THEME_STYLES as THEME_STYLES
        
        # Apply specific component styles
        # Note: Main theme is applied globally, these are component-specific overrides
        
        # Terminal styling
        self.terminal_text.setStyleSheet(THEME_STYLES['terminal'])
        
        # Button styling
        self.read_button.setStyleSheet(THEME_STYLES['button_primary'])
        self.load_cached_button.setStyleSheet(THEME_STYLES['button_primary'])
        self.visualize_button.setStyleSheet(THEME_STYLES['button_primary'])
        
        # Update electrode button style based on whether electrodes are loaded
        self.update_electrode_button_style()
        
    def select_and_read_data(self):
        """Select directory and start data reading process."""
        if self.reader_thread and self.reader_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Data reading is already in progress!")
            return
        
        # Create custom dialog for file type selection
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
        
        file_type_dialog = QDialog(self)
        file_type_dialog.setWindowTitle("Select File Type")
        file_type_dialog.setModal(True)
        
        # Main layout
        dialog_layout = QVBoxLayout(file_type_dialog)
        
        # Question label
        question_label = QLabel("Select the EEG data type.")
        question_label2 = QLabel("(🚧We are continuing to add more formats in future updates.)")
        dialog_layout.addWidget(question_label)
        dialog_layout.addWidget(question_label2)

        # Add spacing
        dialog_layout.addSpacing(20)
        
        # File type buttons (horizontal layout)
        file_buttons_layout = QHBoxLayout()
        rhd_button = QPushButton("RHD Files")
        rhd_button.setMinimumHeight(40)
        rhd_button.setMinimumWidth(120)
        csv_button = QPushButton("CSV Files")
        csv_button.setMinimumHeight(40)
        csv_button.setMinimumWidth(120)
        
        file_buttons_layout.addWidget(rhd_button)
        file_buttons_layout.addWidget(csv_button)
        dialog_layout.addLayout(file_buttons_layout)
        
        # Store the selected file type
        selected_file_type = [None]  # Use list to allow modification in nested function
        
        def on_rhd_clicked():
            selected_file_type[0] = 'rhd'
            file_type_dialog.accept()
        
        def on_csv_clicked():
            selected_file_type[0] = 'csv'
            file_type_dialog.accept()
        
        # Connect buttons
        rhd_button.clicked.connect(on_rhd_clicked)
        csv_button.clicked.connect(on_csv_clicked)
        
        # Execute dialog
        if file_type_dialog.exec() != QDialog.DialogCode.Accepted:
            self.status_bar.showMessage("File selection cancelled")
            return
        
        # Get the selected file type
        file_type = selected_file_type[0]
        
        if file_type == 'rhd':
            # For RHD files, select a folder
            file_description = "RHD files"
            directory = QFileDialog.getExistingDirectory(
                self,
                f"Select folder containing {file_description}",
                self.current_directory or os.path.expanduser("~")
            )
            
            if not directory:
                self.status_bar.showMessage("No directory selected")
                return
            
            self.current_directory = directory
            self.dir_label.setText(f"Selected: {directory} ({file_type.upper()} files)")
            self.dir_label.setStyleSheet("color: red;")
            
            # Start reading data in separate thread
            self.start_data_reading(directory, file_type)
            
        else:  # csv
            # For CSV files, select individual file(s)
            from PyQt6.QtWidgets import QInputDialog
            
            csv_files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select CSV file(s)",
                self.current_directory or os.path.expanduser("~"),
                "All Files (*)"
            )
            
            if not csv_files:
                self.status_bar.showMessage("No files selected")
                return
            
            # Ask for sampling frequency
            sampling_rate, ok = QInputDialog.getDouble(
                self,
                "Sampling Frequency",
                "Enter the sampling frequency (Hz):",
                20000,  # Default value
                1,      # Minimum value
                1000000,  # Maximum value
                0       # Decimals
            )
            
            if not ok:
                self.status_bar.showMessage("Sampling frequency not provided")
                return
            
            # Store the first file's directory
            self.current_directory = str(Path(csv_files[0]).parent)
            file_names = [Path(f).name for f in csv_files]
            self.dir_label.setText(f"Selected: {len(csv_files)} CSV file(s)")
            self.dir_label.setStyleSheet("color: red;")
            
            # Start reading data in separate thread with CSV files and sampling rate
            self.start_data_reading(csv_files, file_type, sampling_rate)
        
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
        
    def start_data_reading(self, path_or_files, file_type='rhd', sampling_rate=None):
        """Start the data reading process in a separate thread.
        
        Args:
            path_or_files: Directory path (str) for RHD files, or list of file paths for CSV
            file_type: 'rhd' or 'csv'
            sampling_rate: Sampling rate in Hz (required for CSV files)
        """
        self.read_button.setEnabled(False)
        self.load_cached_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage(f"Reading {file_type.upper()} files...")
        
        # Create and start reader thread (without saving to cache yet)
        # Pass file_type and sampling_rate to the DataReaderThread
        self.reader_thread = DataReaderThread(
            path_or_files, 
            use_cache=True, 
            save_cache=False, 
            file_type=file_type,
            sampling_rate=sampling_rate
        )
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
        
        # Clean up the thread properly
        if self.reader_thread:
            self.reader_thread.quit()
            self.reader_thread.wait()
            self.reader_thread.deleteLater()
            self.reader_thread = None
        
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
                self.reader_thread.wait(2000)  # Wait up to 2 seconds
                event.accept()
            else:
                event.ignore()
        else:
            # Clean up any finished threads
            if self.reader_thread:
                if self.reader_thread.isFinished():
                    self.reader_thread.wait()
                    self.reader_thread.deleteLater()
                    self.reader_thread = None
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
        # Update electrode button style to show it's activated
        self.update_electrode_button_style()
    
    def update_electrode_button_style(self):
        """Update the electrode configuration button style based on whether electrodes are loaded."""
        # Get current theme
        current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        # Import appropriate theme styles
        if current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_STYLES as THEME_STYLES
        elif current_theme == 'dark':
            from theme import DARK_THEME_STYLES as THEME_STYLES
        else:  # normal
            from theme import NORMAL_THEME_STYLES as THEME_STYLES
        
        # Apply primary button style if electrodes are loaded, otherwise secondary style
        if self.electrode_positions:
            self.electrode_plot_button.setStyleSheet(THEME_STYLES['button_primary'])
        else:
            self.electrode_plot_button.setStyleSheet(THEME_STYLES['button_secondary'])
    
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
