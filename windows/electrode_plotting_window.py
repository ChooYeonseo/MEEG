"""
Electrode plotting window for EEG Analysis GUI application.

This module contains the electrode positioning window that allows users to
visually place electrodes on a mouse head image and specify coordinates.
"""

import os
import json
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QLineEdit, QGroupBox, QSpinBox, QFormLayout,
                            QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
                            QFileDialog, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# Import theme system
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
from theme import preferences_manager


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
    
    def add_electrode(self, x, y, number=None, name=None):
        """Add an electrode at the specified coordinates."""
        self.electrodes.append((x, y))
        
        # Assign color based on electrode number
        colors = ['black']#['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color = colors[len(self.electrodes) % len(colors)]
        self.electrode_colors.append(color)
        
        # Draw the electrode
        circle = patches.Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=2)
        self.ax.add_patch(circle)
        
        # Add electrode label based on display preferences
        label_text = self.get_electrode_label(number or len(self.electrodes), name)
        self.ax.text(x, y, label_text, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        
        self.canvas.draw()
    
    def get_electrode_label(self, number, name):
        """Generate electrode label based on display preferences."""
        if not hasattr(self, 'parent_window'):
            return str(number)
        
        show_number = getattr(self.parent_window, 'show_number', True)
        show_name = getattr(self.parent_window, 'show_name', False)
        
        if show_number and show_name and name:
            return f"{number}\n{name}"
        elif show_name and name:
            return name
        else:
            return str(number)
    
    def set_parent_window(self, parent):
        """Set reference to parent window for display preferences."""
        self.parent_window = parent
    
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
            if hasattr(self, 'parent_window') and hasattr(self.parent_window, 'electrodes_data'):
                for i, electrode_data in enumerate(self.parent_window.electrodes_data):
                    x, y = electrode_data['x'], electrode_data['y']
                    number = electrode_data.get('number', i+1)
                    name = electrode_data.get('name', '')
                    circle = patches.Circle((x, y), 0.4, facecolor=self.electrode_colors[i], 
                                          edgecolor='black', linewidth=2)
                    self.ax.add_patch(circle)
                    label_text = self.get_electrode_label(number, name)
                    self.ax.text(x, y, label_text, ha='center', va='center', 
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
        
        # Get current theme
        self.current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        self.init_ui()
        self.apply_theme()
        
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
        controls_layout = QVBoxLayout(image_controls)
        
        # Button row
        button_row = QHBoxLayout()
        clear_button = QPushButton("Clear All Electrodes")
        clear_button.clicked.connect(self.clear_all_electrodes)
        button_row.addWidget(clear_button)
        
        remove_last_button = QPushButton("Remove Last")
        remove_last_button.clicked.connect(self.remove_last_electrode)
        button_row.addWidget(remove_last_button)
        controls_layout.addLayout(button_row)
        
        # Display options row
        display_row = QHBoxLayout()
        display_label = QLabel("Display on electrodes:")
        display_row.addWidget(display_label)
        
        self.show_number_checkbox = QCheckBox("Number")
        self.show_number_checkbox.setChecked(True)
        self.show_number_checkbox.stateChanged.connect(self.on_display_option_changed)
        display_row.addWidget(self.show_number_checkbox)
        
        self.show_name_checkbox = QCheckBox("Name")
        self.show_name_checkbox.setChecked(False)
        self.show_name_checkbox.stateChanged.connect(self.on_display_option_changed)
        display_row.addWidget(self.show_name_checkbox)
        
        display_row.addStretch()
        controls_layout.addLayout(display_row)
        
        left_layout.addWidget(image_controls)
        
        # Set parent window reference for image widget
        self.image_widget.set_parent_window(self)
        
        # Initialize display preferences
        self.show_number = True
        self.show_name = False
        
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
        self.electrodes_table.setColumnCount(4)
        self.electrodes_table.setHorizontalHeaderLabels(["Number", "Name", "ML", "AP"])
        
        # Connect cell change signal to update plot
        self.electrodes_table.cellChanged.connect(self.on_table_cell_changed)
        
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
    
    def apply_theme(self):
        """Apply the current theme to the window."""
        # Import appropriate theme
        if self.current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_STYLES as THEME_STYLES, TOKYO_NIGHT_COLORS as THEME_COLORS
        elif self.current_theme == 'dark':
            from theme import DARK_THEME_STYLES as THEME_STYLES, DARK_COLORS as THEME_COLORS
        else:  # normal
            from theme import NORMAL_THEME_STYLES as THEME_STYLES, NORMAL_COLORS as THEME_COLORS
        
        # Store theme styles and colors for later use
        self.theme_styles = THEME_STYLES
        self.theme_colors = THEME_COLORS
        
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
            # Check if button has custom style (like saved/loaded state)
            if not widget.styleSheet() or 'background-color: #' not in widget.styleSheet():
                widget.setStyleSheet(THEME_STYLES['button_primary'])
        
        # Apply table styles
        if hasattr(self, 'electrodes_table'):
            self.electrodes_table.setStyleSheet(THEME_STYLES.get('table', ''))
    
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
    
    def on_display_option_changed(self):
        """Handle changes to display options (number/name checkboxes)."""
        self.show_number = self.show_number_checkbox.isChecked()
        self.show_name = self.show_name_checkbox.isChecked()
        # Rebuild electrodes with new display settings
        if self.electrodes_data:  # Only rebuild if there are electrodes
            self.rebuild_image_electrodes()
    
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
                self.show_message_box(QMessageBox.Icon.Warning, "Invalid Coordinate", "X coordinate must be between -5 and 5")
                return
            if not (-8 <= y <= 5):
                self.show_message_box(QMessageBox.Icon.Warning, "Invalid Coordinate", "Y coordinate must be between -8 and 5")
                return
            
            # Add to image and data
            self.image_widget.add_electrode(x, y)
            self.add_electrode_to_data(x, y)
            
            # Clear input fields
            self.x_coord_input.clear()
            self.y_coord_input.clear()
            
        except ValueError:
            self.show_message_box(QMessageBox.Icon.Warning, "Invalid Input", "Please enter valid numeric coordinates")
    
    def add_electrode_to_data(self, x, y, name=None):
        """Add electrode to the data list and update table."""
        electrode_num = len(self.electrodes_data) + 1
        electrode_data = {
            'number': electrode_num,
            'name': name if name else f"E{electrode_num}",
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
        # Apply theme button style
        if hasattr(self, 'theme_styles'):
            button_style = self.theme_styles.get('button_primary', '')
            self.load_button.setStyleSheet(button_style)
            self.save_button.setStyleSheet(button_style)
        self.load_button.setText("📁 Load Electrodes")
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
            self.image_widget.add_electrode(
                electrode['x'], 
                electrode['y'],
                number=electrode.get('number'),
                name=electrode.get('name', '')
            )
    
    def update_electrodes_table(self):
        """Update the electrodes table with current data."""
        # Temporarily disconnect signal to prevent recursive updates
        self.electrodes_table.cellChanged.disconnect(self.on_table_cell_changed)
        
        self.electrodes_table.setRowCount(len(self.electrodes_data))
        
        for i, electrode in enumerate(self.electrodes_data):
            # Electrode number (read-only)
            number_item = QTableWidgetItem(str(electrode['number']))
            number_item.setFlags(number_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.electrodes_table.setItem(i, 0, number_item)
            
            # Name (editable)
            name_item = QTableWidgetItem(electrode.get('name', f"E{electrode['number']}"))
            self.electrodes_table.setItem(i, 1, name_item)
            
            # X and Y coordinates (editable)
            self.electrodes_table.setItem(i, 2, QTableWidgetItem(str(electrode['x'])))
            self.electrodes_table.setItem(i, 3, QTableWidgetItem(str(electrode['y'])))
        
        # Reconnect signal
        self.electrodes_table.cellChanged.connect(self.on_table_cell_changed)
    
    def on_table_cell_changed(self, row, column):
        """Handle changes to table cells and update the plot."""
        # Process changes to Name (column 1), X (column 2) or Y (column 3)
        if column not in [1, 2, 3]:
            return
        
        try:
            # Get the new value
            item = self.electrodes_table.item(row, column)
            if item is None:
                return
            
            if column == 1:  # Name field
                new_name = item.text().strip()
                if not new_name:
                    # Don't allow empty names, restore default
                    new_name = f"E{self.electrodes_data[row]['number']}"
                    item.setText(new_name)
                self.electrodes_data[row]['name'] = new_name
                # Don't rebuild plot for name changes - only update data
                # Plot will be rebuilt when save is pressed or display options change
            else:
                # Coordinate fields
                new_value = float(item.text())
                
                # Validate coordinate ranges
                if column == 2:  # X coordinate (ML)
                    if not (-5 <= new_value <= 5):
                        self.show_message_box(QMessageBox.Icon.Warning, "Invalid Coordinate", "ML coordinate must be between -5 and 5")
                        # Restore old value
                        item.setText(str(self.electrodes_data[row]['x']))
                        return
                    self.electrodes_data[row]['x'] = round(new_value, 3)
                else:  # column == 3, Y coordinate (AP)
                    if not (-8 <= new_value <= 5):
                        self.show_message_box(QMessageBox.Icon.Warning, "Invalid Coordinate", "AP coordinate must be between -8 and 5")
                        # Restore old value
                        item.setText(str(self.electrodes_data[row]['y']))
                        return
                    self.electrodes_data[row]['y'] = round(new_value, 3)
                
                # Update the item with rounded value
                item.setText(str(round(new_value, 3)))
                
                # Rebuild the plot with updated coordinates
                self.rebuild_image_electrodes()
            
            # Reset button styles since electrodes have been modified
            self.reset_button_styles()
            
            # Emit signal with updated positions
            self.positions_updated.emit(self.electrodes_data.copy())
            
        except ValueError:
            self.show_message_box(QMessageBox.Icon.Warning, "Invalid Input", "Please enter a valid numeric value")
            # Restore old value
            if column == 2:
                item.setText(str(self.electrodes_data[row]['x']))
            elif column == 3:
                item.setText(str(self.electrodes_data[row]['y']))
    
    def update_electrode_count(self):
        """Update the electrode count label."""
        self.electrode_count_label.setText(str(len(self.electrodes_data)))
    
    def save_electrodes(self):
        """Save electrode positions to a JSON file."""
        if not self.electrodes_data:
            self.show_message_box(QMessageBox.Icon.Warning, "No Data", "No electrodes to save")
            return
        
        # Rebuild image to reflect any name changes before saving
        if self.show_name:
            self.rebuild_image_electrodes()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Electrode Positions", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.electrodes_data, f, indent=2)
                
                # Change button style to indicate successful saving using theme colors
                success_color = self.theme_colors.get('accent_blue', '#2196F3')
                self.save_button.setStyleSheet(
                    f"QPushButton {{ background-color: {success_color}; color: white; font-weight: bold; }}"
                )
                self.save_button.setText("✅ Saved")
                
                self.show_message_box(QMessageBox.Icon.Information, "Success", f"Electrodes saved to {file_path}")
            except Exception as e:
                self.show_message_box(QMessageBox.Icon.Critical, "Error", f"Error saving file: {str(e)}")
    
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
                        name = electrode.get('name', f"E{electrode.get('number', len(self.electrodes_data) + 1)}")
                        self.image_widget.add_electrode(x, y, number=electrode.get('number'), name=name)
                        self.add_electrode_to_data(x, y, name=name)
                
                # Change button style to indicate successful loading using theme colors
                success_color = self.theme_colors.get('accent_green', '#4CAF50')
                self.load_button.setStyleSheet(
                    f"QPushButton {{ background-color: {success_color}; color: white; font-weight: bold; }}"
                )
                self.load_button.setText("✅ Electrodes Loaded")
                
                self.show_message_box(QMessageBox.Icon.Information, "Success", f"Loaded {len(loaded_data)} electrodes")
                
            except Exception as e:
                self.show_message_box(QMessageBox.Icon.Critical, "Error", f"Error loading file: {str(e)}")
