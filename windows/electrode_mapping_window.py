"""
Electrode mapping window for MEEG Analysis GUI application.

This module contains the electrode mapping window that allows users to
map electrodes to data channels and save the mapping to cache metadata.
"""

import json
import hashlib
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QMessageBox, QComboBox, QSplitter,
                            QTextEdit, QFormLayout, QInputDialog, QListWidget,
                            QListWidgetItem, QDialog, QFileDialog, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# Import theme system
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
from theme import preferences_manager


class AnalysisMethodDialog(QDialog):
    """Dialog for selecting analysis method with gallery-style buttons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_method = None
        
        # Get current theme
        self.current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        self.load_theme_colors()
        
        self.init_ui()
        
    def load_theme_colors(self):
        """Load colors based on the current theme."""
        if self.current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_COLORS as THEME_COLORS
            self.bg_primary = THEME_COLORS['bg_primary']
            self.bg_secondary = THEME_COLORS['bg_secondary']
            self.bg_tertiary = THEME_COLORS['bg_tertiary']
            self.fg_primary = THEME_COLORS['fg_primary']
            self.border = THEME_COLORS['border']
            self.hover = THEME_COLORS['hover']
            self.active = THEME_COLORS['active']
        elif self.current_theme == 'dark':
            from theme import DARK_COLORS as THEME_COLORS
            self.bg_primary = THEME_COLORS['bg_primary']
            self.bg_secondary = THEME_COLORS['bg_secondary']
            self.bg_tertiary = THEME_COLORS['bg_tertiary']
            self.fg_primary = THEME_COLORS['fg_primary']
            self.border = THEME_COLORS['border']
            self.hover = THEME_COLORS.get('accent_hover', '#707070')
            self.active = THEME_COLORS.get('accent_pressed', '#4a4a4a')
        else:  # normal
            from theme import NORMAL_COLORS as THEME_COLORS
            self.bg_primary = THEME_COLORS['bg_primary']
            self.bg_secondary = THEME_COLORS['bg_secondary']
            self.bg_tertiary = THEME_COLORS['bg_tertiary']
            self.fg_primary = THEME_COLORS['fg_primary']
            self.border = THEME_COLORS['border']
            self.hover = THEME_COLORS.get('accent_hover', '#858585')
            self.active = THEME_COLORS.get('accent_pressed', '#4a4a4a')
        
    def init_ui(self):
        """Initialize the dialog UI with gallery-style buttons."""
        self.setWindowTitle("Select Analysis Method")
        self.setModal(True)
        self.setMinimumSize(800, 350)
        
        layout = QVBoxLayout(self)
        
        # Apply dialog background
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.bg_primary};
                color: {self.fg_primary};
            }}
        """)
        
        # Title
        title_label = QLabel("Select Analysis Method")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        layout.addSpacing(20)
        
        # Gallery layout
        gallery_layout = QGridLayout()
        gallery_layout.setSpacing(20)
        
        # Get the asset path
        asset_path = Path(__file__).parent / "asset" / "fig"
        
        # Define the three analysis methods
        methods = [
            ("Sleep Label", str(asset_path / "Sleep_EEG.png")),
            ("Seizure Label", str(asset_path / "Seizure_Label.png")),
            ("Movement Label", str(asset_path / "Movement_Prediction.png"))
        ]
        
        # Create gallery buttons
        for i, (method_name, image_path) in enumerate(methods):
            # Create a single clickable button with image and text
            button = QPushButton()
            button.setMinimumSize(240, 280)
            button.setMaximumSize(240, 280)
            
            # Create layout for button content
            button_layout = QVBoxLayout(button)
            button_layout.setContentsMargins(5, 5, 5, 5)
            button_layout.setSpacing(10)
            
            # Image label inside button
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
            image_label.setScaledContents(False)
            
            # Load and set the image
            if Path(image_path).exists():
                pixmap = QPixmap(image_path)
                # Scale by width to fill horizontally, crop height if needed
                scaled_pixmap = pixmap.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation)
                # Crop to height if the image is taller than 200
                if scaled_pixmap.height() > 200:
                    scaled_pixmap = scaled_pixmap.copy(0, 0, 220, 200)
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("[Image not found]")
            
            image_label.setMinimumHeight(200)
            image_label.setMaximumHeight(200)
            button_layout.addWidget(image_label)
            
            # Text label inside button
            text_label = QLabel(method_name)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet(f"""
                QLabel {{
                    color: {self.fg_primary};
                    font-size: 14px;
                    font-weight: bold;
                    background: transparent;
                }}
            """)
            button_layout.addWidget(text_label)
            
            # Style the entire button with theme colors
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.bg_tertiary};
                    border: 3px solid {self.border};
                    border-radius: 10px;
                    padding: 5px;
                }}
                QPushButton:hover {{
                    background-color: {self.hover};
                    border: 3px solid {self.fg_primary};
                }}
                QPushButton:pressed {{
                    background-color: {self.active};
                    border: 3px solid {self.fg_primary};
                }}
            """)
            button.clicked.connect(lambda checked, m=method_name: self.select_method(m))
            
            # Add to gallery layout
            gallery_layout.addWidget(button, 0, i)
        
        layout.addLayout(gallery_layout)
        
        layout.addStretch()
        
        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(100)
        cancel_button.setMinimumHeight(35)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
    def select_method(self, method_name):
        """Handle method selection."""
        self.selected_method = method_name
        self.accept()
        
    def get_selected_method(self):
        """Return the selected method."""
        return self.selected_method


class ElectrodeVisualizationWidget(QWidget):
    """Widget that displays electrode positions on the mouse head image."""
    
    def __init__(self, electrode_positions, image_path=None):
        super().__init__()
        self.electrode_positions = electrode_positions or []
        self.image_path = image_path or str(Path(__file__).parent.parent / "img" / "Mouse_head_2.png")
        
        # Coordinate system parameters (same as electrode plotting window)
        self.y_min = -8
        self.y_max = 5
        self.x_min = -5
        self.x_max = 5
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the visualization display."""
        self.setMinimumSize(400, 400)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
        self.display_electrodes()
        
    def display_electrodes(self):
        """Display the mouse head image with electrodes."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        try:
            # Load and display the image
            img = plt.imread(self.image_path)
            self.ax.imshow(img, extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                          aspect='equal', origin='upper')
        except Exception as e:
            print(f"Warning: Could not load image {self.image_path}: {e}")
            # Display coordinate system without image
            
        # Set labels and title
        self.ax.set_xlabel('X Coordinate (ML)')
        self.ax.set_ylabel('Y Coordinate (AP)')
        self.ax.set_title('Electrode Positions')
        
        # Set coordinate limits
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Plot electrodes
        self.plot_electrodes()
        
        self.canvas.draw()
    
    def plot_electrodes(self):
        """Plot all electrodes on the image."""
        if not self.electrode_positions:
            return
            
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, electrode in enumerate(self.electrode_positions):
            x, y = electrode['x'], electrode['y']
            number = electrode['number']
            name = electrode.get('name', f"E{number}")
            color = colors[i % len(colors)]
            
            # Draw electrode circle with larger radius
            circle = patches.Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=2)
            self.ax.add_patch(circle)
            
            # Add electrode name
            self.ax.text(x, y, name, ha='center', va='center', 
                        fontsize=5, fontweight='bold', color='white')
            x, y = electrode['x'], electrode['y']
            number = electrode['number']
            name = electrode.get('name', f"E{number}")
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Draw electrode circle
            circle = patches.Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=2)
            self.ax.add_patch(circle)
            
            # Add electrode name
            self.ax.text(x, y, name, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
    
    def update_electrodes(self, electrode_positions):
        """Update electrode positions and redraw."""
        self.electrode_positions = electrode_positions or []
        self.display_electrodes()


class MosaicRelationshipDialog(QDialog):
    """Dialog for managing mosaic relationships between electrodes."""
    
    def __init__(self, electrode_positions, existing_relationships=None, parent=None):
        super().__init__(parent)
        self.electrode_positions = electrode_positions
        # Convert old format (tuples) to new format (dicts) if needed
        self.relationships = []
        if existing_relationships:
            for i, rel in enumerate(existing_relationships):
                if isinstance(rel, tuple):
                    # Old format: convert to new format with electrode name-based name
                    electrode_a_name = self.get_electrode_name(rel[0])
                    electrode_b_name = self.get_electrode_name(rel[1])
                    self.relationships.append({
                        'name': f"{electrode_a_name}-{electrode_b_name}",
                        'electrode_a': rel[0],
                        'electrode_b': rel[1]
                    })
                elif isinstance(rel, dict):
                    # New format: keep as is
                    self.relationships.append(rel.copy())
        self.init_ui()
    
    def get_electrode_name(self, electrode_num):
        """Get electrode name by number."""
        for electrode in self.electrode_positions:
            if electrode['number'] == electrode_num:
                return electrode.get('name', f"E{electrode_num}")
        return f"E{electrode_num}"
        
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Mosaic Relationships")
        self.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Create mosaic relationships between electrode pairs.\n"
            "Electrode numbers and positions will be shown in the selection lists.\n"
            "Select two electrodes to create a relationship between them."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Current relationships list
        relationships_group = QGroupBox("Current Relationships")
        relationships_layout = QVBoxLayout(relationships_group)
        
        self.relationships_list = QListWidget()
        self.update_relationships_list()
        relationships_layout.addWidget(self.relationships_list)
        
        # Buttons for relationship list
        list_buttons = QHBoxLayout()
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_relationship)
        list_buttons.addWidget(remove_button)
        
        clear_all_button = QPushButton("Clear All")
        clear_all_button.clicked.connect(self.clear_all_relationships)
        list_buttons.addWidget(clear_all_button)
        
        list_buttons.addStretch()
        relationships_layout.addLayout(list_buttons)
        
        layout.addWidget(relationships_group)
        
        # Add new relationship
        add_group = QGroupBox("Add New Relationship")
        add_layout = QVBoxLayout(add_group)
        
        add_button = QPushButton("➕ Add Electrode Pair")
        add_button.clicked.connect(self.add_relationship)
        add_layout.addWidget(add_button)
        
        layout.addWidget(add_group)
        
        # Dialog buttons
        buttons_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        buttons_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        layout.addLayout(buttons_layout)
        
    def update_relationships_list(self):
        """Update the relationships list display."""
        self.relationships_list.clear()
        for rel in self.relationships:
            electrode_a = rel['electrode_a']
            electrode_b = rel['electrode_b']
            name = rel['name']
            
            # Find position info for each electrode
            pos_a = next((e for e in self.electrode_positions if e['number'] == electrode_a), None)
            pos_b = next((e for e in self.electrode_positions if e['number'] == electrode_b), None)
            
            if pos_a and pos_b:
                name_a = pos_a.get('name', f"E{electrode_a}")
                name_b = pos_b.get('name', f"E{electrode_b}")
                item_text = (f"{name}: {name_a} (X: {pos_a['x']:.2f}, Y: {pos_a['y']:.2f}) ↔ "
                           f"{name_b} (X: {pos_b['x']:.2f}, Y: {pos_b['y']:.2f})")
            else:
                # Fallback if position data is not available
                item_text = f"{name}: Electrode {electrode_a} ↔ Electrode {electrode_b}"
            
            self.relationships_list.addItem(item_text)
            
    def add_relationship(self):
        """Add a new electrode relationship."""
        if len(self.electrode_positions) < 2:
            QMessageBox.warning(self, "Insufficient Electrodes", 
                              "Need at least 2 electrodes to create a relationship.")
            return
            
        # Get available electrode numbers with position info
        electrode_options = []
        electrode_map = {}
        
        for electrode in self.electrode_positions:
            number = electrode['number']
            name = electrode.get('name', f"E{number}")
            x = electrode['x']
            y = electrode['y']
            display_text = f"{name} (X: {x:.2f}, Y: {y:.2f})"
            electrode_options.append(display_text)
            electrode_map[display_text] = str(number)
        
        # Select first electrode
        electrode_a_display, ok1 = QInputDialog.getItem(
            self, "Select First Electrode", 
            "Choose working electrode (electrode numbers and positions shown):", 
            electrode_options, 0, False
        )
        
        if not ok1:
            return
            
        electrode_a = electrode_map[electrode_a_display]
        
        # Select second electrode (excluding the first one)
        remaining_options = [opt for opt in electrode_options if electrode_map[opt] != electrode_a]
        
        if not remaining_options:
            QMessageBox.warning(self, "No Available Electrodes", 
                              "No other electrodes available for pairing.")
            return
            
        electrode_b_display, ok2 = QInputDialog.getItem(
            self, "Select Second Electrode", 
            "Choose reference electrode (electrode numbers and positions shown):", 
            remaining_options, 0, False
        )
        
        if not ok2:
            return
            
        electrode_b = electrode_map[electrode_b_display]
        
        if not ok2:
            return
            
        # Convert to integers
        electrode_a_num = int(electrode_a)
        electrode_b_num = int(electrode_b)
        
        # Check if relationship already exists
        existing_pairs = [(rel['electrode_a'], rel['electrode_b']) for rel in self.relationships]
        existing_pairs.extend([(rel['electrode_b'], rel['electrode_a']) for rel in self.relationships])
        
        if (electrode_a_num, electrode_b_num) in existing_pairs:
            QMessageBox.warning(self, "Relationship Exists", 
                              f"Relationship between electrodes {electrode_a} and {electrode_b} already exists.")
            return
        
        # Ask for relationship name using electrode names
        electrode_a_name = self.get_electrode_name(electrode_a_num)
        electrode_b_name = self.get_electrode_name(electrode_b_num)
        default_name = f"{electrode_a_name}-{electrode_b_name}"
        relationship_name, ok3 = QInputDialog.getText(
            self, "Name the Relationship", 
            f"Enter a name for this relationship (default: {default_name}):",
            text=default_name
        )
        
        if not ok3:
            return
            
        # Use default name if empty
        if not relationship_name.strip():
            relationship_name = default_name
        
        # Create the relationship dictionary
        new_relationship = {
            'name': relationship_name.strip(),
            'electrode_a': electrode_a_num,
            'electrode_b': electrode_b_num
        }
        
        # Add the relationship
        self.relationships.append(new_relationship)
        self.update_relationships_list()
        
    def remove_selected_relationship(self):
        """Remove the selected relationship."""
        current_row = self.relationships_list.currentRow()
        if current_row >= 0:
            self.relationships.pop(current_row)
            self.update_relationships_list()
            
    def clear_all_relationships(self):
        """Clear all relationships."""
        self.relationships.clear()
        self.update_relationships_list()
        
    def get_relationships(self):
        """Get the current relationships."""
        return self.relationships.copy()


class ElectrodeMappingWindow(QWidget):
    """Window for mapping electrodes to data channels."""
    
    def __init__(self, electrode_positions, current_data, cache_directory, cache_key=None, directory_path=None):
        super().__init__()
        self.electrode_positions = electrode_positions or []
        self.current_data = current_data
        self.cache_directory = Path(cache_directory)
        self.cache_key = cache_key  # Cache key if loaded from cache
        self.directory_path = directory_path  # Directory path if freshly loaded
        self.channel_mapping = {}  # Dictionary mapping electrode number to channel name
        self.mosaic_relationships = []  # List of relationship dicts with names and electrode pairs
        self.imported_config_file = None  # Path to imported JSON configuration file
        
        # Get current theme
        self.current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        # Extract available channels from data
        self.available_channels = self.extract_channels_from_data()
        
        self.init_ui()
        self.load_existing_mapping()
        # Update initial visualization
        self.update_electrode_visualization()
        self.apply_theme()
        
    def extract_channels_from_data(self):
        """Extract available channel names from loaded data."""
        channels = []
        
        if not self.current_data:
            return channels
            
        try:
            # current_data is a list of tuples (filename, result, data_present)
            for filename, result, data_present in self.current_data:
                if data_present and 'amplifier_channels' in result:
                    for channel in result['amplifier_channels']:
                        channel_name = channel.get('native_channel_name', '')
                        if channel_name and channel_name not in channels:
                            channels.append(channel_name)
        except Exception as e:
            print(f"Error extracting channels: {e}")
            
        return sorted(channels)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("MEEG Analysis - Electrode Mapping")
        self.setGeometry(200, 200, 1400, 800)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Electrode to Channel Mapping")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create splitter for layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Electrode visualization and information
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Electrode visualization
        visualization_group = QGroupBox("Electrode Visualization")
        visualization_layout = QVBoxLayout(visualization_group)
        
        self.electrode_visualization = ElectrodeVisualizationWidget(self.electrode_positions)
        visualization_layout.addWidget(self.electrode_visualization)
        
        left_layout.addWidget(visualization_group)
        
        # Electrode positions info
        electrode_info_group = QGroupBox("Electrode Positions")
        electrode_info_layout = QVBoxLayout(electrode_info_group)
        
        self.electrode_info_text = QTextEdit()
        self.electrode_info_text.setReadOnly(True)
        self.electrode_info_text.setMaximumHeight(150)
        self.update_electrode_info_display()
        electrode_info_layout.addWidget(self.electrode_info_text)
        
        left_layout.addWidget(electrode_info_group)
        
        # Available channels info
        channels_info_group = QGroupBox("Available Data Channels")
        channels_info_layout = QVBoxLayout(channels_info_group)
        
        self.channels_info_text = QTextEdit()
        self.channels_info_text.setReadOnly(True)
        self.channels_info_text.setMaximumHeight(150)
        self.update_channels_info_display()
        channels_info_layout.addWidget(self.channels_info_text)
        
        left_layout.addWidget(channels_info_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Mapping interface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Mapping table
        mapping_group = QGroupBox("Electrode-Channel Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        
        # Instructions
        instructions = QLabel(
            "Map each electrode to a data channel. Leave unmapped electrodes empty.\n"
            "Only mapped electrodes will be used in analysis."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; font-style: italic;")
        mapping_layout.addWidget(instructions)
        
        # Mapping table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels([
            "Electrode", "X (ML)", "Y (AP)", "Data Channel"
        ])
        
        # Set column widths
        header = self.mapping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.mapping_table.setColumnWidth(0, 100)
        self.mapping_table.setColumnWidth(1, 100)
        self.mapping_table.setColumnWidth(2, 100)
        
        mapping_layout.addWidget(self.mapping_table)
        
        self.populate_mapping_table()
        
        right_layout.addWidget(mapping_group)
        
        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # First row - mapping controls
        mapping_controls = QHBoxLayout()
        
        clear_mapping_button = QPushButton("Clear All Mappings")
        clear_mapping_button.clicked.connect(self.clear_all_mappings)
        mapping_controls.addWidget(clear_mapping_button)
        
        auto_map_button = QPushButton("Auto-Map Sequential")
        auto_map_button.clicked.connect(self.auto_map_sequential)
        mapping_controls.addWidget(auto_map_button)
        
        mosaic_button = QPushButton("🔗 Mosaic Relationships")
        mosaic_button.clicked.connect(self.manage_mosaic_relationships)
        mapping_controls.addWidget(mosaic_button)
        
        mapping_controls.addStretch()
        controls_layout.addLayout(mapping_controls)
        
        # Second row - file operations
        file_controls = QHBoxLayout()
        
        export_button = QPushButton("📤 Export Configuration")
        export_button.clicked.connect(self.export_configuration)
        export_button.setToolTip("Export electrode positions, mappings, and relationships to JSON file")
        file_controls.addWidget(export_button)
        
        import_button = QPushButton("📥 Import Configuration")
        import_button.clicked.connect(self.import_configuration)
        import_button.setToolTip("Import electrode positions, mappings, and relationships from JSON file")
        file_controls.addWidget(import_button)
        
        file_controls.addStretch()
        
        save_button = QPushButton("💾 Save Mapping")
        save_button.clicked.connect(self.save_mapping)
        save_button.setMinimumHeight(40)
        save_button.setToolTip("Save or update current configuration file")
        file_controls.addWidget(save_button)
        
        save_as_button = QPushButton("💾 Save As")
        save_as_button.clicked.connect(self.save_mapping_as)
        save_as_button.setMinimumHeight(40)
        save_as_button.setToolTip("Save configuration to a new file")
        file_controls.addWidget(save_as_button)
        
        controls_layout.addLayout(file_controls)
        
        right_layout.addWidget(controls_group)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (give more space to visualization)
        splitter.setSizes([600, 800])
        main_layout.addWidget(splitter)
        
        # Footer buttons
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        
        analysis_method_button = QPushButton("Select Analysis Method")
        analysis_method_button.clicked.connect(self.open_analysis_method_dialog)
        analysis_method_button.setMinimumHeight(40)
        analysis_method_button.setMinimumWidth(180)
        footer_layout.addWidget(analysis_method_button)
        
        label_button = QPushButton("Data Preview")
        label_button.clicked.connect(self.open_labeling_window)
        label_button.setMinimumHeight(40)
        label_button.setMinimumWidth(120)
        footer_layout.addWidget(label_button)
        
        main_layout.addLayout(footer_layout)
    
    def apply_theme(self):
        """Apply the current theme to the window."""
        # Import appropriate theme
        if self.current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_STYLES as THEME_STYLES, TOKYO_NIGHT_COLORS as THEME_COLORS
        elif self.current_theme == 'dark':
            from theme import DARK_THEME_STYLES as THEME_STYLES, DARK_COLORS as THEME_COLORS
        else:  # normal
            from theme import NORMAL_THEME_STYLES as THEME_STYLES, NORMAL_COLORS as THEME_COLORS
        
        # Apply window background
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME_COLORS['bg_primary']};
                color: {THEME_COLORS['fg_primary']};
            }}
            {THEME_STYLES.get('form_layout', '')}
        """)
        
        # Apply group box styles
        for widget in self.findChildren(QGroupBox):
            widget.setStyleSheet(THEME_STYLES['group_box'])
        
        # Apply label styles to ensure consistent backgrounds
        for widget in self.findChildren(QLabel):
            if not widget.styleSheet():  # Only if no custom style is set
                widget.setStyleSheet(THEME_STYLES.get('label', ''))
        
        # Apply button styles
        for widget in self.findChildren(QPushButton):
            widget.setStyleSheet(THEME_STYLES['button_primary'])
        
        # Apply table styles
        for widget in self.findChildren(QTableWidget):
            widget.setStyleSheet(THEME_STYLES.get('table', ''))
        
        # Apply text edit styles
        for widget in self.findChildren(QTextEdit):
            widget.setStyleSheet(THEME_STYLES.get('text_edit', ''))
        
    def update_electrode_info_display(self):
        """Update the electrode information display."""
        if not self.electrode_positions:
            self.electrode_info_text.setText("No electrode positions loaded.")
            return
            
        info_text = f"Total Electrodes: {len(self.electrode_positions)}\n"
        
        # Show mapping status
        mapped_count = len(self.channel_mapping)
        relationships_count = len(self.mosaic_relationships)
        
        if mapped_count > 0:
            info_text += f"Mapped Electrodes: {mapped_count}\n"
            info_text += "✅ Existing mappings loaded from cache\n"
        else:
            info_text += "Mapped Electrodes: 0\n"
            
        if relationships_count > 0:
            info_text += f"Mosaic Relationships: {relationships_count}\n"
            info_text += "🔗 Relationships shown as blue lines\n"
        else:
            info_text += "Mosaic Relationships: 0\n"
            
        info_text += "\nElectrode Positions:\n"
        for electrode in self.electrode_positions:
            electrode_num = electrode['number']
            electrode_name = electrode.get('name', f"E{electrode_num}")
            status = ""
            # Check if electrode is mapped
            if electrode_num in self.channel_mapping or str(electrode_num) in self.channel_mapping:
                channel = self.channel_mapping.get(electrode_num) or self.channel_mapping.get(str(electrode_num))
                status = f" → {channel}"
            
            info_text += f"{electrode_name}: "
            info_text += f"ML={electrode['x']:.3f}, AP={electrode['y']:.3f}{status}\n"
            
        self.electrode_info_text.setText(info_text)
        
    def update_channels_info_display(self):
        """Update the channels information display."""
        if not self.available_channels:
            self.channels_info_text.setText("No data channels available.")
            return
            
        info_text = f"Total Channels: {len(self.available_channels)}\n\n"
        info_text += "Available channels:\n"
        for i, channel in enumerate(self.available_channels, 1):
            info_text += f"{i:2d}. {channel}\n"
            
        self.channels_info_text.setText(info_text)
    
    def get_electrode_name(self, electrode_num):
        """Get electrode name by number."""
        for electrode in self.electrode_positions:
            if electrode['number'] == electrode_num:
                return electrode.get('name', f"E{electrode_num}")
        return f"E{electrode_num}"
        
    def populate_mapping_table(self):
        """Populate the mapping table with electrode data."""
        self.mapping_table.setRowCount(len(self.electrode_positions))
        
        for i, electrode in enumerate(self.electrode_positions):
            # Electrode name (with number as fallback)
            electrode_name = electrode.get('name', f"E{electrode['number']}")
            electrode_item = QTableWidgetItem(electrode_name)
            electrode_item.setFlags(electrode_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 0, electrode_item)
            
            # X coordinate
            x_item = QTableWidgetItem(f"{electrode['x']:.3f}")
            x_item.setFlags(x_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 1, x_item)
            
            # Y coordinate
            y_item = QTableWidgetItem(f"{electrode['y']:.3f}")
            y_item.setFlags(y_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 2, y_item)
            
            # Channel dropdown
            channel_combo = QComboBox()
            channel_combo.addItem("")  # Empty option for no mapping
            channel_combo.addItems(self.available_channels)
            
            # Set existing mapping if available
            electrode_num = electrode['number']
            # Check both integer and string keys (JSON converts int keys to strings)
            channel_name = None
            if electrode_num in self.channel_mapping:
                channel_name = self.channel_mapping[electrode_num]
            elif str(electrode_num) in self.channel_mapping:
                channel_name = self.channel_mapping[str(electrode_num)]
                
            if channel_name:
                index = channel_combo.findText(channel_name)
                if index >= 0:
                    channel_combo.setCurrentIndex(index)
            
            # Connect change signal
            channel_combo.currentTextChanged.connect(
                lambda text, row=i: self.on_mapping_changed(row, text)
            )
            
            self.mapping_table.setCellWidget(i, 3, channel_combo)
            
    def on_mapping_changed(self, row, channel_name):
        """Handle mapping changes in the table."""
        # Get electrode number from the electrode_positions data, not from the table text
        if row < len(self.electrode_positions):
            electrode_num = self.electrode_positions[row]['number']
        else:
            return  # Invalid row
        
        if channel_name:
            self.channel_mapping[electrode_num] = channel_name
        else:
            # Remove mapping if empty selection
            self.channel_mapping.pop(electrode_num, None)
        
        # Update displays to show mapping status
        self.update_electrode_info_display()
        self.update_electrode_visualization()
            
    def update_electrode_visualization(self):
        """Update the electrode visualization to show mapping status."""
        # Clear the current plot
        self.electrode_visualization.figure.clear()
        self.electrode_visualization.ax = self.electrode_visualization.figure.add_subplot(111)
        
        try:
            # Load and display the image
            img = plt.imread(self.electrode_visualization.image_path)
            self.electrode_visualization.ax.imshow(
                img, extent=[self.electrode_visualization.x_min, self.electrode_visualization.x_max, 
                           self.electrode_visualization.y_min, self.electrode_visualization.y_max], 
                aspect='equal', origin='upper'
            )
        except Exception as e:
            print(f"Warning: Could not load image: {e}")
            
        # Set labels and title
        self.electrode_visualization.ax.set_xlabel('X Coordinate (ML)')
        self.electrode_visualization.ax.set_ylabel('Y Coordinate (AP)')
        self.electrode_visualization.ax.set_title('Electrode Positions \n (Green=Mapped, Red=Unmapped)')
        
        # Set coordinate limits
        self.electrode_visualization.ax.set_xlim(self.electrode_visualization.x_min, self.electrode_visualization.x_max)
        self.electrode_visualization.ax.set_ylim(self.electrode_visualization.y_min, self.electrode_visualization.y_max)
        
        # Add grid
        self.electrode_visualization.ax.grid(True, alpha=0.3)
        
        # Plot electrodes with colors based on mapping status
        for electrode in self.electrode_positions:
            x, y = electrode['x'], electrode['y']
            number = electrode['number']
            
            # Choose color based on mapping status
            if number in self.channel_mapping:
                color = 'green'  # Mapped electrode
                edge_color = 'darkgreen'
            else:
                color = 'red'    # Unmapped electrode
                edge_color = 'darkred'
            
            # Draw electrode circle
            circle = patches.Circle((x, y), 0.3, facecolor=color, edgecolor=edge_color, linewidth=2, alpha=0.6)
            self.electrode_visualization.ax.add_patch(circle)
            
            # Add electrode name
            name = electrode.get('name', f"E{number}")
            self.electrode_visualization.ax.text(x, y, name, ha='center', va='center', 
                                               fontsize=10, fontweight='bold', color='black')
        
        # Draw mosaic relationships as lines
        self.draw_mosaic_relationships()
        
        self.electrode_visualization.canvas.draw()
    
    def draw_mosaic_relationships(self):
        """Draw lines connecting electrodes with mosaic relationships."""
        if not self.mosaic_relationships:
            return
            
        # Create a dictionary for quick electrode position lookup
        electrode_positions = {e['number']: (e['x'], e['y']) for e in self.electrode_positions}
        
        for rel in self.mosaic_relationships:
            # Handle both old tuple format and new dict format
            if isinstance(rel, dict):
                electrode_a = rel['electrode_a']
                electrode_b = rel['electrode_b']
                name = rel['name']
            else:
                # Old tuple format for backward compatibility
                electrode_a, electrode_b = rel
                name = f"{electrode_a}↔{electrode_b}"
                
            if electrode_a in electrode_positions and electrode_b in electrode_positions:
                x1, y1 = electrode_positions[electrode_a]
                x2, y2 = electrode_positions[electrode_b]
                
                # Draw line connecting the electrodes
                self.electrode_visualization.ax.plot(
                    [x1, x2], [y1, y2], 
                    color='blue', linewidth=2, alpha=0.7, linestyle='-'
                )
                
                # Add a small label at the midpoint of the line with the relationship name
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                self.electrode_visualization.ax.text(
                    mid_x, mid_y, name, 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7)
                )
            
    def clear_all_mappings(self):
        """Clear all electrode-channel mappings."""
        self.channel_mapping.clear()
        self.populate_mapping_table()
        self.update_electrode_info_display()
        self.update_electrode_visualization()
        
    def auto_map_sequential(self):
        """Automatically map electrodes to channels sequentially."""
        if not self.available_channels:
            QMessageBox.warning(self, "No Channels", "No data channels available for mapping.")
            return
            
        self.channel_mapping.clear()
        
        for i, electrode in enumerate(self.electrode_positions):
            if i < len(self.available_channels):
                electrode_num = electrode['number']
                self.channel_mapping[electrode_num] = self.available_channels[i]
                
        self.populate_mapping_table()
        self.update_electrode_info_display()
        self.update_electrode_visualization()
        
    def manage_mosaic_relationships(self):
        """Open dialog to manage mosaic relationships."""
        dialog = MosaicRelationshipDialog(
            self.electrode_positions, 
            self.mosaic_relationships, 
            self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.mosaic_relationships = dialog.get_relationships()
            print(f"Updated mosaic relationships: {self.mosaic_relationships}")
            # Update visualization to show relationships
            self.update_electrode_visualization()
        
    def load_existing_mapping(self):
        """Load existing mapping from cache metadata if available."""
        if not self.current_data:
            return
            
        try:
            # Find cache file based on current data
            cache_file = self.find_cache_metadata_file()
            if cache_file and cache_file.exists():
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load electrode positions if they exist in cache (with names)
                if 'electrode_positions' in metadata:
                    # Update electrode_positions with data from cache, preserving names
                    cached_positions = metadata['electrode_positions']
                    # Create a mapping of number to cached electrode data
                    cached_by_number = {e['number']: e for e in cached_positions}
                    
                    # Update existing electrode_positions with cached data
                    for i, electrode in enumerate(self.electrode_positions):
                        num = electrode['number']
                        if num in cached_by_number:
                            # Update with cached data, including name if it exists
                            cached_electrode = cached_by_number[num]
                            if 'name' in cached_electrode:
                                self.electrode_positions[i]['name'] = cached_electrode['name']
                    
                    print(f"Updated electrode positions with cached names")
                    
                # Load electrode mapping if it exists
                if 'electrode_mapping' in metadata:
                    raw_mapping = metadata['electrode_mapping']
                    # Normalize keys to integers (JSON converts int keys to strings)
                    self.channel_mapping = {}
                    for key, value in raw_mapping.items():
                        try:
                            int_key = int(key)
                            self.channel_mapping[int_key] = value
                        except (ValueError, TypeError):
                            # If key can't be converted to int, keep as string
                            self.channel_mapping[key] = value
                    
                    print(f"Loaded existing electrode mapping: {self.channel_mapping}")
                    
                # Load mosaic relationships if they exist
                if 'mosaic_relationships' in metadata:
                    raw_relationships = metadata['mosaic_relationships']
                    self.mosaic_relationships = []
                    
                    for i, rel in enumerate(raw_relationships):
                        if isinstance(rel, list) and len(rel) == 2:
                            # Old tuple format (stored as list in JSON): convert to new format
                            electrode_a_name = self.get_electrode_name(rel[0])
                            electrode_b_name = self.get_electrode_name(rel[1])
                            self.mosaic_relationships.append({
                                'name': f"{electrode_a_name}-{electrode_b_name}",
                                'electrode_a': rel[0],
                                'electrode_b': rel[1]
                            })
                        elif isinstance(rel, dict):
                            # New format: keep as is
                            self.mosaic_relationships.append(rel.copy())
                    
                    print(f"Loaded existing mosaic relationships: {self.mosaic_relationships}")
                    
                # Update UI components
                if 'electrode_mapping' in metadata or 'mosaic_relationships' in metadata:
                    # Repopulate the table to show existing mappings
                    self.populate_mapping_table()
                    # Update electrode info display to show mapping status
                    self.update_electrode_info_display()
                    # Update visualization after loading mapping and relationships
                    self.update_electrode_visualization()
                    
        except Exception as e:
            print(f"Error loading existing mapping: {e}")
            
    def find_cache_metadata_file(self):
        """Find the cache metadata file for current data."""
        try:
            # If we have a cache key, use it directly
            if self.cache_key:
                cache_file = self.cache_directory / f"{self.cache_key}_metadata.json"
                return cache_file
            
            # If we have a directory path, generate the cache key
            elif self.directory_path:
                cache_hash = hashlib.md5(self.directory_path.encode()).hexdigest()
                cache_file = self.cache_directory / f"{cache_hash}_metadata.json"
                return cache_file
            
            # Fallback: try to extract from current_data structure
            elif self.current_data and len(self.current_data) > 0:
                first_item = self.current_data[0]
                if len(first_item) >= 2:
                    result = first_item[1]  # The actual data result
                    if isinstance(result, dict) and 'directory_path' in result:
                        directory_path = result['directory_path']
                        cache_hash = hashlib.md5(directory_path.encode()).hexdigest()
                        cache_file = self.cache_directory / f"{cache_hash}_metadata.json"
                        return cache_file
            
            return None
                
        except Exception as e:
            print(f"Error finding cache file: {e}")
            return None
        
    def save_mapping(self):
        """Save the electrode mapping and mosaic relationships to JSON configuration file."""
        if not self.electrode_positions:
            QMessageBox.warning(self, "No Data", "No electrode positions to save.")
            return
            
        try:
            # Determine which file to save to
            save_file = None
            
            if self.imported_config_file and Path(self.imported_config_file).exists():
                # Update the existing imported JSON file
                save_file = self.imported_config_file
                action_text = "update"
            else:
                # Ask user where to save
                default_filename = "mosaic_config.json"
                save_file, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Mosaic Configuration",
                    str(Path.home() / default_filename),
                    "JSON Files (*.json);;All Files (*)"
                )
                if not save_file:
                    return  # User cancelled
                action_text = "save"
            
            # Create configuration dictionary
            # Only include names if they exist - preserve original structure
            electrode_positions_to_save = []
            for electrode in self.electrode_positions:
                electrode_copy = electrode.copy()
                # Keep the name only if it was already in the original data
                electrode_positions_to_save.append(electrode_copy)
            
            config = {
                "electrode_positions": electrode_positions_to_save,
                "channel_mapping": {},
                "mosaic_relationships": self.mosaic_relationships.copy() if self.mosaic_relationships else [],
                "metadata": {
                    "description": "MEEG Mosaic Configuration",
                    "version": "1.0",
                    "total_electrodes": len(self.electrode_positions),
                    "mapped_electrodes": len(self.channel_mapping),
                    "relationships": len(self.mosaic_relationships) if self.mosaic_relationships else 0
                }
            }
            
            # Convert channel mapping keys to strings for JSON compatibility
            for key, value in self.channel_mapping.items():
                config["channel_mapping"][str(key)] = value
            
            # Save to file
            with open(save_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update the imported_config_file reference
            self.imported_config_file = save_file
            
            # Show summary
            mapped_count = len(self.channel_mapping)
            relationships_count = len(self.mosaic_relationships)
            total_electrodes = len(self.electrode_positions)
            
            summary_text = f"Configuration {action_text}d successfully!\n\n"
            summary_text += f"File: {Path(save_file).name}\n"
            summary_text += f"Electrodes: {total_electrodes}\n"
            summary_text += f"Mapped: {mapped_count}\n"
            summary_text += f"Relationships: {relationships_count}"
            
            QMessageBox.information(self, "Configuration Saved", summary_text)
            print(f"Configuration saved to: {save_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving configuration: {str(e)}")
            print(f"Error saving configuration: {e}")
    
    def save_mapping_as(self):
        """Save the electrode mapping and mosaic relationships to a new JSON file."""
        if not self.electrode_positions:
            QMessageBox.warning(self, "No Data", "No electrode positions to save.")
            return
            
        try:
            # Always ask user where to save (new file)
            default_filename = "mosaic_config.json"
            save_file, _ = QFileDialog.getSaveFileName(
                self,
                "Save Mosaic Configuration As",
                str(Path.home() / default_filename),
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not save_file:
                return  # User cancelled
            
            # Create configuration dictionary
            # Only include names if they exist - preserve original structure
            electrode_positions_to_save = []
            for electrode in self.electrode_positions:
                electrode_copy = electrode.copy()
                # Keep the name only if it was already in the original data
                electrode_positions_to_save.append(electrode_copy)
            
            config = {
                "electrode_positions": electrode_positions_to_save,
                "channel_mapping": {},
                "mosaic_relationships": self.mosaic_relationships.copy() if self.mosaic_relationships else [],
                "metadata": {
                    "description": "MEEG Mosaic Configuration",
                    "version": "1.0",
                    "total_electrodes": len(self.electrode_positions),
                    "mapped_electrodes": len(self.channel_mapping),
                    "relationships": len(self.mosaic_relationships) if self.mosaic_relationships else 0
                }
            }
            
            # Convert channel mapping keys to strings for JSON compatibility
            for key, value in self.channel_mapping.items():
                config["channel_mapping"][str(key)] = value
            
            # Save to file
            with open(save_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update the imported_config_file reference to the new file
            self.imported_config_file = save_file
            
            # Show summary
            mapped_count = len(self.channel_mapping)
            relationships_count = len(self.mosaic_relationships)
            total_electrodes = len(self.electrode_positions)
            
            summary_text = f"Configuration saved successfully!\n\n"
            summary_text += f"File: {Path(save_file).name}\n"
            summary_text += f"Electrodes: {total_electrodes}\n"
            summary_text += f"Mapped: {mapped_count}\n"
            summary_text += f"Relationships: {relationships_count}"
            
            QMessageBox.information(self, "Configuration Saved", summary_text)
            print(f"Configuration saved to: {save_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving configuration: {str(e)}")
            print(f"Error saving configuration: {e}")
    
    def open_analysis_method_dialog(self):
        """Open the analysis method selection dialog."""
        # Check if we have required data
        if not self.electrode_positions:
            QMessageBox.warning(self, "No Electrode Positions", 
                              "Please configure electrode positions first before opening analysis windows.")
            return
            
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data available for analysis.")
            return
        
        dialog = AnalysisMethodDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            method = dialog.get_selected_method()
            if method:
                try:
                    if method == "Sleep Label":
                        from .M1.sleep_label_main import SleepLabelWindow
                        self.analysis_window = SleepLabelWindow(
                            electrode_positions=self.electrode_positions,
                            current_data=self.current_data,
                            channel_mapping=self.channel_mapping,
                            mosaic_relationships=self.mosaic_relationships,
                            parent=None
                        )
                    elif method == "Seizure Label":
                        from .M2.epilepsy_label_main import EpilepsyLabelWindow
                        self.analysis_window = EpilepsyLabelWindow(
                            electrode_positions=self.electrode_positions,
                            current_data=self.current_data,
                            channel_mapping=self.channel_mapping,
                            mosaic_relationships=self.mosaic_relationships,
                            parent=None
                        )
                    elif method == "Movement Label":
                        from .M3.motion_label_main import MotionLabelWindow
                        self.analysis_window = MotionLabelWindow(
                            electrode_positions=self.electrode_positions,
                            current_data=self.current_data,
                            channel_mapping=self.channel_mapping,
                            mosaic_relationships=self.mosaic_relationships,
                            parent=None
                        )
                    
                    self.analysis_window.show()
                    self.analysis_window.raise_()
                    self.analysis_window.activateWindow()
                    print(f"Opened {method} window")
                    
                except ImportError as e:
                    QMessageBox.warning(
                        self,
                        "Module Not Found",
                        f"The {method} module is not yet implemented.\n\nError: {str(e)}"
                    )
                    print(f"Import error for {method}: {e}")
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Error opening {method} window:\n{str(e)}"
                    )
                    print(f"Error opening {method} window: {e}")
    
    def open_labeling_window(self):
        """Open the labeling window for data analysis (mosaic or single electrode mode)."""
        # Check if we have electrode positions (required for either mode)
        if not self.electrode_positions:
            QMessageBox.warning(self, "No Electrode Positions", 
                              "Please configure electrode positions first before opening the labeling window.")
            return
            
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No data available for analysis.")
            return
            
        # Create and show the newly adapted preview window
        try:
            from .preview.preview_main import PreviewWindow
            self.labeling_window = PreviewWindow(
                mosaic_relationships=self.mosaic_relationships,
                electrode_positions=self.electrode_positions,
                current_data=self.current_data,
                channel_mapping=self.channel_mapping,
                parent=None  # Make it independent
            )
            self.labeling_window.show()
            self.labeling_window.raise_()  # Bring to front
            self.labeling_window.activateWindow()  # Activate the window
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening integrated data view window: {str(e)}")
            print(f"Error opening integrated data view window: {e}")
    
    def export_configuration(self):
        """Export the complete mosaic configuration to a JSON file."""
        if not self.electrode_positions:
            QMessageBox.warning(self, "No Data", "No electrode positions to export.")
            return
        
        # Create configuration dictionary
        # Only include names if they exist - preserve original structure
        electrode_positions_to_save = []
        for electrode in self.electrode_positions:
            electrode_copy = electrode.copy()
            # Keep the name only if it was already in the original data
            electrode_positions_to_save.append(electrode_copy)
        
        config = {
            "electrode_positions": electrode_positions_to_save,
            "channel_mapping": {},
            "mosaic_relationships": self.mosaic_relationships.copy() if self.mosaic_relationships else [],
            "metadata": {
                "description": "MEEG Mosaic Configuration",
                "version": "1.0",
                "total_electrodes": len(self.electrode_positions),
                "mapped_electrodes": len(self.channel_mapping),
                "relationships": len(self.mosaic_relationships) if self.mosaic_relationships else 0
            }
        }
        
        # Convert channel mapping keys to strings for JSON compatibility
        for key, value in self.channel_mapping.items():
            config["channel_mapping"][str(key)] = value
        
        # Open file dialog to save
        default_filename = "mosaic_config.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mosaic Configuration",
            str(Path.home() / default_filename),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Configuration exported successfully!\n\n"
                f"File: {Path(file_path).name}\n"
                f"Electrodes: {config['metadata']['total_electrodes']}\n"
                f"Mapped: {config['metadata']['mapped_electrodes']}\n"
                f"Relationships: {config['metadata']['relationships']}"
            )
            print(f"Configuration exported to: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting configuration:\n{str(e)}")
            print(f"Error exporting configuration: {e}")
    
    def import_configuration(self):
        """Import mosaic configuration from a JSON file."""
        # Open file dialog to select file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mosaic Configuration",
            str(Path.home()),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # Load configuration from file
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration structure
            if not isinstance(config, dict):
                raise ValueError("Invalid configuration format: must be a JSON object")
            
            # Check for required fields
            if "electrode_positions" not in config:
                raise ValueError("Configuration missing 'electrode_positions' field")
            
            # Validate electrode positions structure
            electrode_positions = config["electrode_positions"]
            if not isinstance(electrode_positions, list):
                raise ValueError("electrode_positions must be a list")
            
            for i, electrode in enumerate(electrode_positions):
                if not isinstance(electrode, dict):
                    raise ValueError(f"Electrode {i} must be a dictionary")
                if not all(key in electrode for key in ['number', 'x', 'y']):
                    raise ValueError(f"Electrode {i} missing required fields (number, x, y)")
            
            # Ask user to confirm import
            metadata = config.get("metadata", {})
            total_electrodes = len(electrode_positions)
            mapped_electrodes = len(config.get("channel_mapping", {}))
            relationships = len(config.get("mosaic_relationships", []))
            
            confirm_msg = (
                f"Import configuration from {Path(file_path).name}?\n\n"
                f"Electrodes: {total_electrodes}\n"
                f"Mapped channels: {mapped_electrodes}\n"
                f"Mosaic relationships: {relationships}\n\n"
                f"This will replace current electrode positions and mappings."
            )
            
            reply = QMessageBox.question(
                self,
                "Confirm Import",
                confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Apply the configuration - preserve original structure
            # Do NOT add default names - keep JSON as-is
            self.electrode_positions = electrode_positions
            
            # Track the imported file path
            self.imported_config_file = file_path
            
            # Load channel mapping
            self.channel_mapping = {}
            if "channel_mapping" in config:
                raw_mapping = config["channel_mapping"]
                for key, value in raw_mapping.items():
                    try:
                        int_key = int(key)
                        self.channel_mapping[int_key] = value
                    except (ValueError, TypeError):
                        self.channel_mapping[key] = value
            
            # Load mosaic relationships
            if "mosaic_relationships" in config:
                raw_relationships = config["mosaic_relationships"]
                self.mosaic_relationships = []
                
                for i, rel in enumerate(raw_relationships):
                    if isinstance(rel, list) and len(rel) == 2:
                        # Old tuple format (stored as list in JSON)
                        electrode_a_name = self.get_electrode_name(rel[0])
                        electrode_b_name = self.get_electrode_name(rel[1])
                        self.mosaic_relationships.append({
                            'name': f"{electrode_a_name}-{electrode_b_name}",
                            'electrode_a': rel[0],
                            'electrode_b': rel[1]
                        })
                    elif isinstance(rel, dict):
                        # New format
                        self.mosaic_relationships.append(rel.copy())
            
            # Update all UI components
            self.populate_mapping_table()
            self.update_electrode_info_display()
            self.update_channels_info_display()
            self.update_electrode_visualization()
            
            QMessageBox.information(
                self,
                "Import Successful",
                f"Configuration imported successfully!\n\n"
                f"Electrodes: {total_electrodes}\n"
                f"Mapped: {len(self.channel_mapping)}\n"
                f"Relationships: {len(self.mosaic_relationships)}"
            )
            print(f"Configuration imported from: {file_path}")
            
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Import Error", f"Invalid JSON file:\n{str(e)}")
            print(f"JSON decode error: {e}")
        except ValueError as e:
            QMessageBox.critical(self, "Import Error", f"Invalid configuration:\n{str(e)}")
            print(f"Validation error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing configuration:\n{str(e)}")
            print(f"Error importing configuration: {e}")
