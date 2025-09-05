"""
Electrode plotting window for EEG Analysis GUI application.

This module contains the electrode positioning window that allows users to
visually place electrodes on a mouse head image and specify coordinates.
"""

import os
import json
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QGroupBox, QSpinBox, QFormLayout,
                            QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
                            QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


class ClickableImageWidget(QWidget):
    """Custom widget that displays an image and allows clicking to place electrodes."""
    
    electrode_placed = pyqtSignal(float, float)  # Signal emitted when electrode is placed
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.electrodes = []  # List of (x, y) electrode positions
        self.electrode_colors = []  # List of colors for each electrode
        
        # Coordinate system parameters
        self.y_min = -8
        self.y_max = 5
        self.x_min = -5
        self.x_max = 5
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the image display."""
        self.setMinimumSize(600, 600)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
        # Connect mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.load_and_display_image()
        
    def load_and_display_image(self):
        """Load and display the mouse head image."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        try:
            # Load the image
            img = plt.imread(self.image_path)
            
            # Display image with specified coordinate system
            self.ax.imshow(img, extent=[self.x_min, self.x_max, self.y_min, self.y_max], 
                          aspect='equal', origin='upper')
            
            # Set labels and title
            self.ax.set_xlabel('X Coordinate')
            self.ax.set_ylabel('Y Coordinate')
            self.ax.set_title('Mouse Head - Click to Place Electrodes')
            
            # Set coordinate limits
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Display empty plot with coordinate system if image fails to load
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)
            self.ax.set_xlabel('X Coordinate')
            self.ax.set_ylabel('Y Coordinate')
            self.ax.set_title('Mouse Head - Image not found')
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events to place electrodes."""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Add electrode at clicked position
        self.add_electrode(x, y)
        self.electrode_placed.emit(x, y)
    
    def add_electrode(self, x, y):
        """Add an electrode at the specified coordinates."""
        self.electrodes.append((x, y))
        
        # Assign color based on electrode number
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color = colors[len(self.electrodes) % len(colors)]
        self.electrode_colors.append(color)
        
        # Draw the electrode
        circle = patches.Circle((x, y), 0.1, facecolor=color, edgecolor='black', linewidth=2)
        self.ax.add_patch(circle)
        
        # Add electrode number
        self.ax.text(x, y, str(len(self.electrodes)), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        
        self.canvas.draw()
    
    def clear_electrodes(self):
        """Clear all placed electrodes."""
        self.electrodes.clear()
        self.electrode_colors.clear()
        self.load_and_display_image()
    
    def remove_last_electrode(self):
        """Remove the last placed electrode."""
        if self.electrodes:
            self.electrodes.pop()
            self.electrode_colors.pop()
            self.load_and_display_image()
            # Redraw all remaining electrodes
            for i, (x, y) in enumerate(self.electrodes):
                circle = patches.Circle((x, y), 0.1, facecolor=self.electrode_colors[i], 
                                      edgecolor='black', linewidth=2)
                self.ax.add_patch(circle)
                self.ax.text(x, y, str(i+1), ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
            self.canvas.draw()


class ElectrodePlottingWindow(QWidget):
    """Main window for electrode positioning and coordinate management."""
    
    # Signal emitted when electrode positions are updated
    positions_updated = pyqtSignal(list)
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window  # Reference to main window
        self.electrodes_data = []  # List of electrode data dictionaries
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Electrode Positioning Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Image and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image widget
        image_path = Path(__file__).parent.parent / "img" / "Mouse_head_2.png"
        self.image_widget = ClickableImageWidget(str(image_path))
        self.image_widget.electrode_placed.connect(self.on_electrode_placed)
        left_layout.addWidget(self.image_widget)
        
        # Image controls
        image_controls = QGroupBox("Image Controls")
        controls_layout = QHBoxLayout(image_controls)
        
        clear_button = QPushButton("Clear All Electrodes")
        clear_button.clicked.connect(self.clear_all_electrodes)
        controls_layout.addWidget(clear_button)
        
        remove_last_button = QPushButton("Remove Last")
        remove_last_button.clicked.connect(self.remove_last_electrode)
        controls_layout.addWidget(remove_last_button)
        
        left_layout.addWidget(image_controls)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Electrode management
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Manual coordinate input
        manual_input_group = QGroupBox("Manual Electrode Input")
        manual_layout = QFormLayout(manual_input_group)
        
        self.x_coord_input = QLineEdit()
        self.x_coord_input.setPlaceholderText("-5.0 to 5.0")
        manual_layout.addRow("ML:", self.x_coord_input)
        
        self.y_coord_input = QLineEdit()
        self.y_coord_input.setPlaceholderText("-8.0 to 5.0")
        manual_layout.addRow("AP:", self.y_coord_input)
        
        add_manual_button = QPushButton("Add Electrode")
        add_manual_button.clicked.connect(self.add_manual_electrode)
        manual_layout.addRow(add_manual_button)
        
        right_layout.addWidget(manual_input_group)
        
        # Electrode count
        count_group = QGroupBox("Electrode Count")
        count_layout = QFormLayout(count_group)
        
        self.electrode_count_label = QLabel("0")
        count_layout.addRow("Total Electrodes:", self.electrode_count_label)
        
        right_layout.addWidget(count_group)
        
        # Electrodes table
        table_group = QGroupBox("Electrode Positions")
        table_layout = QVBoxLayout(table_group)
        
        self.electrodes_table = QTableWidget()
        self.electrodes_table.setColumnCount(3)
        self.electrodes_table.setHorizontalHeaderLabels(["Electrode #", "ML", "AP"])
        
        # Make table columns stretch to fill available space
        header = self.electrodes_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        table_layout.addWidget(self.electrodes_table)
        
        # Table controls
        table_controls = QHBoxLayout()
        
        delete_selected_button = QPushButton("Delete Selected")
        delete_selected_button.clicked.connect(self.delete_selected_electrode)
        table_controls.addWidget(delete_selected_button)
        
        table_controls.addStretch()
        table_layout.addLayout(table_controls)
        
        right_layout.addWidget(table_group)
        
        # Save/Load controls
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout(file_group)
        
        self.save_button = QPushButton("💾 Save Electrodes")
        self.save_button.clicked.connect(self.save_electrodes)
        file_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("📁 Load Electrodes")
        self.load_button.clicked.connect(self.load_electrodes)
        file_layout.addWidget(self.load_button)
        
        right_layout.addWidget(file_group)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (image takes more space)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        self.update_electrode_count()
    
    def on_electrode_placed(self, x, y):
        """Handle electrode placement from image click."""
        self.add_electrode_to_data(x, y)
    
    def add_manual_electrode(self):
        """Add electrode from manual coordinate input."""
        try:
            x = float(self.x_coord_input.text())
            y = float(self.y_coord_input.text())
            
            # Validate coordinates
            if not (-5 <= x <= 5):
                QMessageBox.warning(self, "Invalid Coordinate", "X coordinate must be between -5 and 5")
                return
            if not (-8 <= y <= 5):
                QMessageBox.warning(self, "Invalid Coordinate", "Y coordinate must be between -8 and 5")
                return
            
            # Add to image and data
            self.image_widget.add_electrode(x, y)
            self.add_electrode_to_data(x, y)
            
            # Clear input fields
            self.x_coord_input.clear()
            self.y_coord_input.clear()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric coordinates")
    
    def add_electrode_to_data(self, x, y):
        """Add electrode to the data list and update table."""
        electrode_data = {
            'number': len(self.electrodes_data) + 1,
            'x': round(x, 3),
            'y': round(y, 3)
        }
        
        self.electrodes_data.append(electrode_data)
        self.update_electrodes_table()
        self.update_electrode_count()
        # Reset button styles since electrodes have been modified
        self.reset_button_styles()
        # Emit signal with updated positions
        self.positions_updated.emit(self.electrodes_data.copy())
    
    def reset_button_styles(self):
        """Reset the file operation buttons to their default appearance."""
        self.load_button.setStyleSheet("")  # Reset to default style
        self.load_button.setText("📁 Load Electrodes")
        self.save_button.setStyleSheet("")  # Reset to default style
        self.save_button.setText("💾 Save Electrodes")
    
    def clear_all_electrodes(self):
        """Clear all electrodes from image and data."""
        self.image_widget.clear_electrodes()
        self.electrodes_data.clear()
        self.update_electrodes_table()
        self.update_electrode_count()
        # Reset load button appearance
        self.reset_button_styles()
        # Emit signal with empty positions
        self.positions_updated.emit(self.electrodes_data.copy())
    
    def remove_last_electrode(self):
        """Remove the last electrode."""
        if self.electrodes_data:
            self.electrodes_data.pop()
            self.image_widget.remove_last_electrode()
            self.update_electrodes_table()
            self.update_electrode_count()
            # Reset button styles since electrodes have been modified
            self.reset_button_styles()
            # Emit signal with updated positions
            self.positions_updated.emit(self.electrodes_data.copy())
    
    def delete_selected_electrode(self):
        """Delete the selected electrode from the table."""
        current_row = self.electrodes_table.currentRow()
        if current_row >= 0:
            # Remove from data
            self.electrodes_data.pop(current_row)
            
            # Update electrode numbers
            for i, electrode in enumerate(self.electrodes_data):
                electrode['number'] = i + 1
            
            # Rebuild image
            self.rebuild_image_electrodes()
            self.update_electrodes_table()
            self.update_electrode_count()
            # Reset button styles since electrodes have been modified
            self.reset_button_styles()
            # Emit signal with updated positions
            self.positions_updated.emit(self.electrodes_data.copy())
    
    def rebuild_image_electrodes(self):
        """Rebuild all electrodes on the image."""
        self.image_widget.clear_electrodes()
        for electrode in self.electrodes_data:
            self.image_widget.add_electrode(electrode['x'], electrode['y'])
    
    def update_electrodes_table(self):
        """Update the electrodes table with current data."""
        self.electrodes_table.setRowCount(len(self.electrodes_data))
        
        for i, electrode in enumerate(self.electrodes_data):
            self.electrodes_table.setItem(i, 0, QTableWidgetItem(str(electrode['number'])))
            self.electrodes_table.setItem(i, 1, QTableWidgetItem(str(electrode['x'])))
            self.electrodes_table.setItem(i, 2, QTableWidgetItem(str(electrode['y'])))
    
    def update_electrode_count(self):
        """Update the electrode count label."""
        self.electrode_count_label.setText(str(len(self.electrodes_data)))
    
    def save_electrodes(self):
        """Save electrode positions to a JSON file."""
        if not self.electrodes_data:
            QMessageBox.warning(self, "No Data", "No electrodes to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Electrode Positions", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.electrodes_data, f, indent=2)
                
                # Change button color to indicate successful saving
                self.save_button.setStyleSheet(
                    "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
                )
                self.save_button.setText("✅ Saved")
                
                QMessageBox.information(self, "Success", f"Electrodes saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving file: {str(e)}")
    
    def load_electrodes(self):
        """Load electrode positions from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Electrode Positions", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Validate data format
                if not isinstance(loaded_data, list):
                    raise ValueError("Invalid file format")
                
                # Clear existing electrodes
                self.clear_all_electrodes()
                
                # Load electrodes
                for electrode in loaded_data:
                    if isinstance(electrode, dict) and 'x' in electrode and 'y' in electrode:
                        x, y = electrode['x'], electrode['y']
                        self.image_widget.add_electrode(x, y)
                        self.add_electrode_to_data(x, y)
                
                # Change button color to indicate successful loading
                self.load_button.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                )
                self.load_button.setText("✅ Electrodes Loaded")
                
                QMessageBox.information(self, "Success", f"Loaded {len(loaded_data)} electrodes")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
