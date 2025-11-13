"""
Preprocessing pipeline window for EEG data.

This module provides a window where users can configure and chain
preprocessing methods before caching the data.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                            QListWidgetItem, QPushButton, QLabel, QGroupBox,
                            QComboBox, QSpinBox, QDoubleSpinBox, QScrollArea,
                            QWidget, QFormLayout, QCheckBox, QMessageBox,
                            QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QBrush

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
theme_dir = current_dir / "theme"
sys.path.insert(0, str(utils_dir))
sys.path.insert(0, str(theme_dir))

try:
    import signal_preprocessing
    from cache_manager import cache_manager
    from data_format_converter import (
        prepare_data_for_preprocessing, 
        finalize_processed_data, 
        convert_rhd_result_to_dataframe
    )
    from theme.preferences import preferences_manager
except ImportError as e:
    print(f"Error importing preprocessing utilities: {e}")


class PreprocessingStep:
    """Represents a single preprocessing step in the pipeline."""
    
    def __init__(self, name, function, parameters=None, description=""):
        self.name = name
        self.function = function
        self.parameters = parameters or {}
        self.description = description
        self.enabled = True
    
    def apply(self, data, fs=None):
        """Apply this preprocessing step to the data."""
        if not self.enabled:
            return data
        
        # Add sampling rate to parameters if function needs it
        params = self.parameters.copy()
        if 'fs' in self.function.__code__.co_varnames and fs is not None:
            params['fs'] = fs
        
        return self.function(data, **params)


class PreprocessingWorker(QThread):
    """Worker thread for applying preprocessing pipeline."""
    
    progress_update = pyqtSignal(str)
    step_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal(object)  # Processed data
    
    def __init__(self, data, pipeline, sampling_rate=None):
        super().__init__()
        self.data = data
        self.pipeline = pipeline
        self.sampling_rate = sampling_rate
    
    def run(self):
        """Run the preprocessing pipeline."""
        try:
            # First, prepare data for preprocessing (convert to DataFrame format)
            from data_format_converter import prepare_data_for_preprocessing, finalize_processed_data
            # Import signal_preprocessing for get_sampling_rate function
            import signal_preprocessing
            
            self.progress_update.emit("Preparing data for preprocessing...")
            prepared_data = prepare_data_for_preprocessing(self.data)
            
            processed_data = []
            
            for i, (filename, data_df, data_present, original_result) in enumerate(prepared_data):
                if not data_present or data_df is None:
                    processed_data.append((filename, data_df, data_present, original_result))
                    continue
                
                self.progress_update.emit(f"Processing {filename} ({i+1}/{len(prepared_data)})")
                
                # Get the initial sampling rate for this file
                fs = self.sampling_rate
                if original_result and 'frequency_parameters' in original_result:
                    fs = original_result['frequency_parameters'].get('amplifier_sample_rate', fs)
                
                # Apply pipeline to the DataFrame
                processed_df = data_df.copy()
                current_fs = fs  # Track current sampling rate through pipeline
                
                for step in self.pipeline:
                    if step.enabled:
                        self.step_completed.emit(f"Applying {step.name} to {filename}")
                        processed_df = step.apply(processed_df, current_fs)
                        
                        # If this was a resampling step, recalculate the sampling rate
                        if hasattr(step.function, '__name__') and step.function.__name__ == 'resample_dataframe':
                            # Get the new sampling rate from the step parameters
                            if 'target_rate' in step.parameters:
                                current_fs = step.parameters['target_rate']
                                self.step_completed.emit(f"Sampling rate updated to {current_fs} Hz after resampling")
                            else:
                                # If no target rate specified, recalculate from data
                                current_fs = signal_preprocessing.get_sampling_rate(processed_df)
                                self.step_completed.emit(f"Sampling rate recalculated to {current_fs} Hz after resampling")
                
                processed_data.append((filename, processed_df, data_present, original_result, current_fs))
            
            # Convert processed DataFrames back to RHD result format
            self.progress_update.emit("Converting processed data back to cache format...")
            final_data = finalize_processed_data(processed_data)
            
            self.finished_processing.emit(final_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Error during preprocessing: {str(e)}")


class PreprocessingPipelineDialog(QDialog):
    """Dialog for configuring preprocessing pipeline."""
    
    def __init__(self, data, directory_path, parent=None):
        super().__init__(parent)
        self.data = data
        self.directory_path = directory_path
        self.pipeline = []
        self.processed_data = None
        self.setup_ui()
        self.setup_available_methods()
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme to this dialog."""
        # Theme is already applied globally to the application
        # We don't need to set it again on this dialog
        # Individual widgets like list widgets have their own stylesheets set
        pass
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Preprocessing Pipeline Configuration")
        self.setMinimumSize(900, 700)
        self.resize(1000, 800)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel - Available methods
        left_panel = self.create_available_methods_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Pipeline and controls
        right_panel = self.create_pipeline_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_available_methods_panel(self):
        """Create the panel with available preprocessing methods."""
        panel = QGroupBox("Available Preprocessing Methods")
        layout = QVBoxLayout(panel)
        
        # Search/filter (future enhancement)
        # filter_layout = QHBoxLayout()
        # self.filter_edit = QLineEdit()
        # self.filter_edit.setPlaceholderText("Search methods...")
        # filter_layout.addWidget(QLabel("Filter:"))
        # filter_layout.addWidget(self.filter_edit)
        # layout.addLayout(filter_layout)
        
        # Available methods list
        self.available_methods = QListWidget()
        self.available_methods.setMaximumHeight(300)
        
        # Set list widget stylesheet that doesn't override item colors
        # This allows setForeground() to work on individual items
        current_theme = preferences_manager.get_theme()
        if current_theme == "tokyo_night":
            from theme.tokyo_night_theme import TOKYO_NIGHT_COLORS
            list_bg = TOKYO_NIGHT_COLORS['bg_secondary']
            list_border = TOKYO_NIGHT_COLORS['border']
            list_selection = TOKYO_NIGHT_COLORS['selection']
            list_text = TOKYO_NIGHT_COLORS['fg_primary']
        elif current_theme == "dark":
            from theme.dark_theme import DARK_COLORS
            list_bg = DARK_COLORS['bg_secondary']
            list_border = DARK_COLORS['border']
            list_selection = DARK_COLORS['accent_primary']
            list_text = DARK_COLORS['fg_primary']
        else:
            from theme.normal_theme import NORMAL_COLORS
            list_bg = NORMAL_COLORS['bg_secondary']
            list_border = NORMAL_COLORS['border']
            list_selection = NORMAL_COLORS['accent_primary']
            list_text = NORMAL_COLORS['fg_primary']
        
        # Stylesheet with default text color that can be overridden by setForeground
        self.available_methods.setStyleSheet(f"""
            QListWidget {{
                background-color: {list_bg};
                color: {list_text};
                border: 1px solid {list_border};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-radius: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {list_selection};
            }}
        """)
        
        layout.addWidget(self.available_methods)
        
        # Method description and details
        desc_group = QGroupBox("Method Information")
        desc_layout = QVBoxLayout(desc_group)
        
        self.method_description = QTextEdit()
        self.method_description.setReadOnly(True)
        self.method_description.setMaximumHeight(180)
        self.method_description.setPlainText("Select a preprocessing method to view detailed information...")
        desc_layout.addWidget(self.method_description)
        layout.addWidget(desc_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        # Refresh methods button
        self.refresh_methods_button = QPushButton("🔄 Refresh Methods")
        self.refresh_methods_button.clicked.connect(self.refresh_methods)
        self.refresh_methods_button.setToolTip("Reload preprocessing methods from signal_preprocessing.py")
        control_layout.addWidget(self.refresh_methods_button)
        
        control_layout.addStretch()
        
        # Add to pipeline button
        self.add_method_button = QPushButton("➕ Add to Pipeline")
        self.add_method_button.clicked.connect(self.add_method_to_pipeline)
        self.add_method_button.setEnabled(False)
        control_layout.addWidget(self.add_method_button)
        
        layout.addLayout(control_layout)
        
        # Connect selection change
        self.available_methods.itemSelectionChanged.connect(self.on_method_selection_changed)
        
        return panel
    
    def create_pipeline_panel(self):
        """Create the pipeline configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Pipeline list
        pipeline_group = QGroupBox("Preprocessing Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        
        self.pipeline_list = QListWidget()
        self.pipeline_list.setMinimumHeight(200)
        
        # Set pipeline list widget stylesheet to match theme
        current_theme = preferences_manager.get_theme()
        if current_theme == "tokyo_night":
            from theme.tokyo_night_theme import TOKYO_NIGHT_COLORS
            pipeline_bg = TOKYO_NIGHT_COLORS['bg_secondary']
            pipeline_border = TOKYO_NIGHT_COLORS['border']
            pipeline_selection = TOKYO_NIGHT_COLORS['selection']
            pipeline_text = TOKYO_NIGHT_COLORS['fg_primary']
        elif current_theme == "dark":
            from theme.dark_theme import DARK_COLORS
            pipeline_bg = DARK_COLORS['bg_secondary']
            pipeline_border = DARK_COLORS['border']
            pipeline_selection = DARK_COLORS['accent_primary']
            pipeline_text = DARK_COLORS['fg_primary']
        else:
            from theme.normal_theme import NORMAL_COLORS
            pipeline_bg = NORMAL_COLORS['bg_secondary']
            pipeline_border = NORMAL_COLORS['border']
            pipeline_selection = NORMAL_COLORS['accent_primary']
            pipeline_text = NORMAL_COLORS['fg_primary']
        
        # Apply stylesheet with proper text color
        self.pipeline_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {pipeline_bg};
                color: {pipeline_text};
                border: 1px solid {pipeline_border};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-radius: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {pipeline_selection};
            }}
        """)
        
        pipeline_layout.addWidget(self.pipeline_list)
        
        # Pipeline controls
        pipeline_controls = QHBoxLayout()
        self.remove_step_button = QPushButton("🗑️ Remove")
        self.move_up_button = QPushButton("⬆️ Move Up")
        self.move_down_button = QPushButton("⬇️ Move Down")
        self.configure_step_button = QPushButton("⚙️ Configure")
        
        self.remove_step_button.clicked.connect(self.remove_pipeline_step)
        self.move_up_button.clicked.connect(self.move_step_up)
        self.move_down_button.clicked.connect(self.move_step_down)
        self.configure_step_button.clicked.connect(self.configure_step)
        
        pipeline_controls.addWidget(self.remove_step_button)
        pipeline_controls.addWidget(self.move_up_button)
        pipeline_controls.addWidget(self.move_down_button)
        pipeline_controls.addWidget(self.configure_step_button)
        pipeline_controls.addStretch()
        
        pipeline_layout.addLayout(pipeline_controls)
        layout.addWidget(pipeline_group)
        
        # Progress section
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to process")
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("👁️ Preview Pipeline")
        self.process_button = QPushButton("⚡ Process & Cache")
        self.cancel_button = QPushButton("Cancel")
        
        self.preview_button.clicked.connect(self.preview_pipeline)
        self.process_button.clicked.connect(self.process_and_cache)
        self.cancel_button.clicked.connect(self.reject)
        
        self.process_button.setDefault(True)
        
        button_layout.addWidget(self.preview_button)
        button_layout.addStretch()
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        return panel
    
    def setup_available_methods(self):
        """Populate the available methods list by discovering them from signal_preprocessing module."""
        try:
            # Get all registered preprocessing methods
            methods_dict = signal_preprocessing.get_preprocessing_methods()
            
            if not methods_dict:
                # Fallback to manual method definitions if no methods are registered
                self.setup_fallback_methods()
                return
            
            self.method_definitions = {}
            
            # Group methods by category
            categories = {}
            for method_info in methods_dict.values():
                category = method_info.get('category', 'General')
                if category not in categories:
                    categories[category] = []
                categories[category].append(method_info)
            
            # Add methods to the list, grouped by category
            for category, methods in sorted(categories.items()):
                # Add category header
                if len(categories) > 1:
                    # Get theme-specific color for category headers
                    current_theme = preferences_manager.get_theme()
                    if current_theme == "tokyo_night":
                        from theme.tokyo_night_theme import TOKYO_NIGHT_COLORS
                        category_color = TOKYO_NIGHT_COLORS['accent_cyan']
                    elif current_theme == "dark":
                        from theme.dark_theme import DARK_COLORS
                        category_color = DARK_COLORS['fg_primary']
                    else:
                        from theme.normal_theme import NORMAL_COLORS
                        category_color = NORMAL_COLORS['accent_primary']
                    
                    category_item = QListWidgetItem(f"--- {category} ---")
                    category_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it unselectable
                    category_item.setData(Qt.ItemDataRole.UserRole, None)
                    font = category_item.font()
                    font.setBold(True)
                    category_item.setFont(font)
                    
                    # Set category header color using QBrush for better compatibility
                    from PyQt6.QtGui import QBrush
                    category_item.setForeground(QBrush(QColor(category_color)))
                    
                    self.available_methods.addItem(category_item)
                
                # Add methods in this category
                for method_info in sorted(methods, key=lambda x: x['name']):
                    item = QListWidgetItem(f"  {method_info['name']}")
                    item.setData(Qt.ItemDataRole.UserRole, method_info)
                    self.available_methods.addItem(item)
                    self.method_definitions[method_info['name']] = method_info
                    
        except Exception as e:
            print(f"Error discovering preprocessing methods: {e}")
            # Fallback to manual definitions
            self.setup_fallback_methods()
    
    def setup_fallback_methods(self):
        """Fallback method definitions if automatic discovery fails."""
        # Define basic preprocessing methods manually as fallback
        methods = [
            {
                'name': 'Remove DC Offset',
                'function': signal_preprocessing.remove_dc_offset,
                'description': 'Removes the DC (direct current) offset from the signal by subtracting the mean.',
                'category': 'Baseline Correction',
                'parameters': {}
            },
            {
                'name': 'Bandpass Filter',
                'function': signal_preprocessing.bandpass_filter,
                'description': 'Applies a bandpass filter to retain frequencies within a specified range.',
                'category': 'Filtering',
                'parameters': {
                    'lowcut': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 1000, 'label': 'Low cutoff (Hz)'},
                    'highcut': {'type': 'float', 'default': 100, 'min': 1, 'max': 1000, 'label': 'High cutoff (Hz)'},
                    'order': {'type': 'int', 'default': 4, 'min': 1, 'max': 10, 'label': 'Filter order'}
                }
            }
            # Add more fallback methods as needed
        ]
        
        self.method_definitions = {}
        
        for method in methods:
            item = QListWidgetItem(method['name'])
            item.setData(Qt.ItemDataRole.UserRole, method)
            self.available_methods.addItem(item)
            self.method_definitions[method['name']] = method
    
    def on_method_selection_changed(self):
        """Handle selection change in available methods."""
        current_item = self.available_methods.currentItem()
        if current_item:
            method_data = current_item.data(Qt.ItemDataRole.UserRole)
            if method_data:
                # Display detailed method information
                info_text = self.format_method_info(method_data)
                self.method_description.setPlainText(info_text)
                self.add_method_button.setEnabled(True)
            else:
                self.method_description.setPlainText("Select a preprocessing method to view detailed information...")
                self.add_method_button.setEnabled(False)
        else:
            self.method_description.setPlainText("Select a preprocessing method to view detailed information...")
            self.add_method_button.setEnabled(False)
    
    def format_method_info(self, method_data):
        """Format detailed information about a preprocessing method."""
        info_lines = []
        
        info_lines.append(f"Method: {method_data['name']}")
        info_lines.append(f"Category: {method_data.get('category', 'General')}")
        info_lines.append("")
        info_lines.append("Description:")
        info_lines.append(method_data.get('description', 'No description available.'))
        
        # Parameter information
        parameters = method_data.get('parameters', {})
        if parameters:
            info_lines.append("")
            info_lines.append("Parameters:")
            for param_name, param_config in parameters.items():
                param_line = f"• {param_config.get('label', param_name)}"
                if param_config.get('type'):
                    param_line += f" ({param_config['type']})"
                if param_config.get('default') is not None:
                    param_line += f" [default: {param_config['default']}]"
                info_lines.append(param_line)
                
                if param_config.get('description'):
                    info_lines.append(f"  {param_config['description']}")
                    
                if 'min' in param_config and 'max' in param_config:
                    info_lines.append(f"  Range: {param_config['min']} - {param_config['max']}")
        else:
            info_lines.append("")
            info_lines.append("Parameters: None")
        
        # Function information
        if hasattr(method_data.get('function'), '__doc__') and method_data['function'].__doc__:
            info_lines.append("")
            info_lines.append("Function Documentation:")
            # Extract first few lines of docstring
            docstring = method_data['function'].__doc__.strip()
            doc_lines = docstring.split('\n')[:5]  # First 5 lines
            for line in doc_lines:
                info_lines.append(f"  {line.strip()}")
            if len(docstring.split('\n')) > 5:
                info_lines.append("  ...")
        
        return '\n'.join(info_lines)
    
    def refresh_methods(self):
        """Refresh the list of available preprocessing methods."""
        try:
            # Clear current methods
            self.available_methods.clear()
            self.method_definitions.clear()
            
            # Reload the signal_preprocessing module
            import importlib
            importlib.reload(signal_preprocessing)
            
            # Repopulate methods
            self.setup_available_methods()
            
            # Clear description
            self.method_description.setPlainText("Methods refreshed! Select a method to view information...")
            
        except Exception as e:
            QMessageBox.warning(self, "Refresh Error", f"Error refreshing methods: {str(e)}")
    
    def add_method_to_pipeline(self):
        """Add selected method to the pipeline."""
        current_item = self.available_methods.currentItem()
        if not current_item:
            return
        
        method_data = current_item.data(Qt.ItemDataRole.UserRole)
        if not method_data:
            return
        
        # Create preprocessing step with default parameters
        default_params = {}
        for param_name, param_config in method_data.get('parameters', {}).items():
            if param_config['type'] == 'bool':
                default_params[param_name] = param_config['default']
            else:
                default_params[param_name] = param_config['default']
        
        step = PreprocessingStep(
            name=method_data['name'],
            function=method_data['function'],
            parameters=default_params,
            description=method_data['description']
        )
        
        self.pipeline.append(step)
        self.update_pipeline_list()
    
    def update_pipeline_list(self):
        """Update the pipeline list display."""
        self.pipeline_list.clear()
        
        for i, step in enumerate(self.pipeline):
            status = "✓" if step.enabled else "✗"
            display_text = f"{i+1}. {status} {step.name}"
            
            # Add parameter summary
            if step.parameters:
                param_summary = ", ".join([f"{k}={v}" for k, v in step.parameters.items()])
                display_text += f" ({param_summary})"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.pipeline_list.addItem(item)
    
    def remove_pipeline_step(self):
        """Remove selected step from pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            del self.pipeline[current_row]
            self.update_pipeline_list()
    
    def move_step_up(self):
        """Move selected step up in the pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row > 0:
            self.pipeline[current_row], self.pipeline[current_row-1] = \
                self.pipeline[current_row-1], self.pipeline[current_row]
            self.update_pipeline_list()
            self.pipeline_list.setCurrentRow(current_row-1)
    
    def move_step_down(self):
        """Move selected step down in the pipeline."""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0 and current_row < len(self.pipeline) - 1:
            self.pipeline[current_row], self.pipeline[current_row+1] = \
                self.pipeline[current_row+1], self.pipeline[current_row]
            self.update_pipeline_list()
            self.pipeline_list.setCurrentRow(current_row+1)
    
    def configure_step(self):
        """Configure parameters for selected step."""
        current_row = self.pipeline_list.currentRow()
        if current_row < 0:
            return
        
        step = self.pipeline[current_row]
        method_def = self.method_definitions.get(step.name)
        
        if not method_def or not method_def.get('parameters'):
            QMessageBox.information(self, "No Parameters", 
                                   f"The method '{step.name}' has no configurable parameters.")
            return
        
        # Create parameter configuration dialog
        config_dialog = ParameterConfigDialog(step, method_def, self)
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_pipeline_list()
    
    def preview_pipeline(self):
        """Show a preview of what the pipeline will do."""
        if not self.pipeline:
            QMessageBox.information(self, "Empty Pipeline", 
                                   "Please add some preprocessing steps to the pipeline first.")
            return
        
        preview_text = "Preprocessing Pipeline Preview:\n\n"
        for i, step in enumerate(self.pipeline):
            status = "ENABLED" if step.enabled else "DISABLED"
            preview_text += f"{i+1}. {step.name} ({status})\n"
            preview_text += f"   Description: {step.description}\n"
            if step.parameters:
                preview_text += f"   Parameters: {step.parameters}\n"
            preview_text += "\n"
        
        QMessageBox.information(self, "Pipeline Preview", preview_text)
    
    def process_and_cache(self):
        """Process the data through the pipeline and cache the results."""
        if not self.pipeline:
            QMessageBox.information(self, "Empty Pipeline", 
                                   "Please add some preprocessing steps to the pipeline first.")
            return
        
        reply = QMessageBox.question(
            self,
            "Process and Cache",
            "This will apply the preprocessing pipeline to your data and save it to cache.\n\n"
            "This process may take some time depending on the amount of data.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Disable UI elements
        self.process_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        
        # Start processing
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_label.setText("Starting preprocessing...")
        
        # Get sampling rate from first file if available
        sampling_rate = None
        for filename, result, data_present in self.data:
            if data_present and result and 'frequency_parameters' in result:
                sampling_rate = result['frequency_parameters'].get('amplifier_sample_rate')
                break
        
        # Start worker thread
        self.worker = PreprocessingWorker(self.data, self.pipeline, sampling_rate)
        self.worker.progress_update.connect(self.progress_label.setText)
        self.worker.step_completed.connect(lambda msg: print(msg))
        self.worker.error_occurred.connect(self.on_processing_error)
        self.worker.finished_processing.connect(self.on_processing_finished)
        self.worker.start()
    
    def on_processing_error(self, error_msg):
        """Handle processing error."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Error occurred")
        
        self.process_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        
        QMessageBox.critical(self, "Processing Error", error_msg)
    
    def on_processing_finished(self, processed_data):
        """Handle completion of preprocessing."""
        self.processed_data = processed_data
        
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Processing complete! Saving to cache...")
        
        try:
            # Save processed data to cache
            cache_manager.save_to_cache(self.directory_path, processed_data)
            
            self.progress_label.setText("Processing and caching complete!")
            
            QMessageBox.information(
                self,
                "Success",
                "Preprocessing pipeline completed successfully!\n\n"
                "The processed data has been saved to cache."
            )
            
            self.accept()  # Close dialog with success
            
        except Exception as e:
            self.on_processing_error(f"Error saving to cache: {str(e)}")
    
    def get_processed_data(self):
        """Get the processed data."""
        return self.processed_data


class ParameterConfigDialog(QDialog):
    """Dialog for configuring preprocessing step parameters."""
    
    def __init__(self, step, method_def, parent=None):
        super().__init__(parent)
        self.step = step
        self.method_def = method_def
        self.parameter_widgets = {}
        self.setup_ui()
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme to this dialog."""
        # Theme is already applied globally to the application
        # We don't need to set it again on this dialog
        pass
    
    def setup_ui(self):
        """Set up the parameter configuration UI."""
        self.setWindowTitle(f"Configure {self.step.name}")
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"Configure Parameters for {self.step.name}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(self.step.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(desc_label)
        
        # Enable/disable checkbox
        self.enabled_checkbox = QCheckBox("Enable this preprocessing step")
        self.enabled_checkbox.setChecked(self.step.enabled)
        layout.addWidget(self.enabled_checkbox)
        
        # Parameters form
        if self.method_def.get('parameters'):
            form_group = QGroupBox("Parameters")
            form_layout = QFormLayout(form_group)
            
            for param_name, param_config in self.method_def['parameters'].items():
                widget = self.create_parameter_widget(param_name, param_config)
                self.parameter_widgets[param_name] = widget
                form_layout.addRow(param_config['label'], widget)
            
            layout.addWidget(form_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        ok_button.setDefault(True)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def create_parameter_widget(self, param_name, param_config):
        """Create appropriate widget for parameter type."""
        param_type = param_config['type']
        current_value = self.step.parameters.get(param_name, param_config['default'])
        
        if param_type == 'int':
            widget = QSpinBox()
            widget.setRange(param_config.get('min', -1000000), 
                           param_config.get('max', 1000000))
            widget.setValue(current_value)
            
        elif param_type == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(param_config.get('min', -1000000.0), 
                           param_config.get('max', 1000000.0))
            widget.setDecimals(3)
            widget.setValue(current_value)
            
        elif param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(current_value)
            
        else:
            # Default to spin box for unknown types
            widget = QDoubleSpinBox()
            widget.setValue(float(current_value))
        
        return widget
    
    def accept(self):
        """Accept the configuration and update the step."""
        # Update enabled state
        self.step.enabled = self.enabled_checkbox.isChecked()
        
        # Update parameters
        for param_name, widget in self.parameter_widgets.items():
            if isinstance(widget, QSpinBox):
                self.step.parameters[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                self.step.parameters[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.step.parameters[param_name] = widget.isChecked()
        
        super().accept()
