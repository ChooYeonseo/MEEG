"""
Integrated Data View window for MEEG Analysis GUI application.

This module contains the integrated data view window that allows users to
visualize and label both mosaic data (differential signals between electrode pairs)
and individual electrode data, depending on the configuration provided.
"""

import numpy as np
import pandas as pd
import traceback
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QGroupBox, QSplitter, QTextEdit,
                            QMessageBox, QFormLayout, QLineEdit, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QFont, QShortcut, QKeySequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from scipy import signal
except ImportError:
    print("Warning: scipy not available. STFT functionality will not work.")

# Add utils directory to Python path
current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

# Import theme system
sys.path.insert(0, str(current_dir))
from theme import preferences_manager

try:
    from utils import read_intan
    from utils import widget
except ImportError as e:
    print(f"Import error in labeling_window: {e}")


class STFTSpectrogramWindow(QWidget):
    """Separate window for STFT spectrogram display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STFT Spectrogram Analysis")
        self.setGeometry(200, 200, 1000, 700)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowType.Window | 
                           Qt.WindowType.WindowMinimizeButtonHint | 
                           Qt.WindowType.WindowMaximizeButtonHint | 
                           Qt.WindowType.WindowCloseButtonHint)
        
        # Power range parameters (in dB)
        self.power_min = -30  # Lower bound (constant)
        self.power_max = 30   # Upper bound (adjustable)
        
        # Store current data for re-plotting
        self.current_data = None
        
        layout = QVBoxLayout(self)
        
        # Title and controls
        header_layout = QVBoxLayout()
        
        title = QLabel("STFT Spectrogram (2 epochs before to 2 epochs after current)")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        # Power range control buttons
        controls_layout = QHBoxLayout()
        
        # Power range display
        self.power_range_label = QLabel(f"Power Range: {self.power_min}dB to {self.power_max}dB")
        self.power_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.power_range_label)
        
        # Dimmer button (decrease upper bound)
        self.dimmer_button = QPushButton("🔅 Dimmer (-2dB)")
        self.dimmer_button.clicked.connect(self.decrease_power_range)
        self.dimmer_button.setToolTip("Decrease power range upper bound by 2dB")
        controls_layout.addWidget(self.dimmer_button)
        
        # Brighter button (increase upper bound)
        self.brighter_button = QPushButton("🔆 Brighter (+2dB)")
        self.brighter_button.clicked.connect(self.increase_power_range)
        self.brighter_button.setToolTip("Increase power range upper bound by 2dB")
        controls_layout.addWidget(self.brighter_button)
        
        # Reset button
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.clicked.connect(self.reset_power_range)
        self.reset_button.setToolTip("Reset power range to default (-30dB to +30dB)")
        controls_layout.addWidget(self.reset_button)
        
        header_layout.addLayout(controls_layout)
        layout.addLayout(header_layout)
        
        # Spectrogram plot
        self.spectrogram_figure = Figure(figsize=(12, 8))
        self.spectrogram_canvas = FigureCanvas(self.spectrogram_figure)
        layout.addWidget(self.spectrogram_canvas)
    
    def decrease_power_range(self):
        """Decrease the power range upper bound by 2dB."""
        self.power_max -= 2
        self.update_power_range_display()
        if self.current_data:
            self.replot_spectrogram()
    
    def increase_power_range(self):
        """Increase the power range upper bound by 2dB."""
        self.power_max += 2
        self.update_power_range_display()
        if self.current_data:
            self.replot_spectrogram()
    
    def reset_power_range(self):
        """Reset power range to default values."""
        self.power_min = -30
        self.power_max = 30
        self.update_power_range_display()
        if self.current_data:
            self.replot_spectrogram()
    
    def update_power_range_display(self):
        """Update the power range label."""
        self.power_range_label.setText(f"Power Range: {self.power_min}dB to {self.power_max}dB")
    
    def replot_spectrogram(self):
        """Replot the spectrogram with current power range settings."""
        if not self.current_data:
            return
            
        times, frequencies, power_spectrum_db, start_time, epoch_length, working_electrode, spectrogram_start, spectrogram_end = self.current_data
        
        # Clear current figure
        self.spectrogram_figure.clear()
        ax = self.spectrogram_figure.add_subplot(1, 1, 1)
        
        # Plot spectrogram with fixed power range
        im = ax.pcolormesh(times, frequencies, power_spectrum_db, shading='gouraud', cmap='viridis',
                          vmin=self.power_min, vmax=self.power_max)
        
        # Add colorbar
        cbar = self.spectrogram_figure.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', rotation=270, labelpad=15)
        
        # Mark current epoch boundaries with red dashed lines
        ax.axvline(start_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(start_time + epoch_length, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 30)
        ax.set_title(f'Working Electrode E{working_electrode} - STFT Spectrogram ({spectrogram_start:.1f}s - {spectrogram_end:.1f}s)')
        
        self.spectrogram_figure.tight_layout()
        self.spectrogram_canvas.draw()
        
    def update_spectrogram(self, times, frequencies, power_spectrum_db, start_time, epoch_length, working_electrode, spectrogram_start, spectrogram_end):
        """Update the spectrogram display with new data."""
        # Store current data for re-plotting when power range changes
        self.current_data = (times, frequencies, power_spectrum_db, start_time, epoch_length, working_electrode, spectrogram_start, spectrogram_end)
        
        # Plot with current power range settings
        self.replot_spectrogram()
        
        # Show the window
        self.show()
        self.raise_()
        self.activateWindow()


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
        
        # Title - show different text based on whether mosaic relationships exist
        if self.mosaic_relationships and len(self.mosaic_relationships) > 0:
            title_text = "Electrode & Mosaic Configuration"
        else:
            title_text = "Electrode Configuration"
        
        title = QLabel(title_text)
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
        
        # Determine if mosaic configuration is being used
        # MOSAIC_USAGE is False when only electrode maps are provided (no mosaic relationships)
        # MOSAIC_USAGE is True when mosaic relationships exist
        self.MOSAIC_USAGE = bool(mosaic_relationships and len(mosaic_relationships) > 0)
        
        # Get current theme
        self.current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        # Time navigation parameters
        self.current_start_time = 0.0
        self.current_epoch_length = 4.0  # Default baseline epoch length (4 seconds)
        self.sample_rate = None
        self.total_duration = 0.0
        self.file_metadata = []
        self.cached_segments = {}
        self.mosaic_data_cache = {}
        self.global_sigma = None  # Fixed global sigma for consistent scaling
        
        # Fixed y-axis scaling parameters for consistent plots
        self.fixed_y_limits = None  # Store (y_bottom, y_top) for consistent scaling
        self.fixed_total_height = None  # Store total height for consistent red line positioning
        
        # STFT power range parameters (in dB)
        self.stft_power_min = -30  # Lower bound (adjustable)
        self.stft_power_max = 30   # Upper bound (constant)
        self.current_stft_data = None  # Store current STFT data for re-plotting
        
        self.setup_ui()
        self.initialize_data()
        self.apply_theme()
        
    def setup_ui(self):
        """Set up the labeling window UI."""
        # Set window title based on MOSAIC_USAGE
        if self.MOSAIC_USAGE:
            self.setWindowTitle("Mosaic Data Labeling")
        else:
            self.setWindowTitle("Electrode Data Labeling")
        
        self.setGeometry(150, 150, 1400, 1200)  # Increased height for vertical stacking
        
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
        self.start_time_input.returnPressed.connect(self.update_all_plots)  # Auto-update on Enter
        time_controls.addWidget(self.start_time_input)
        nav_layout.addRow("Start Time:", time_controls)
        
        # Epoch length input
        epoch_controls = QHBoxLayout()
        self.epoch_length_input = QLineEdit("4.0")  # Default to 4.0 seconds for baseline epoch
        self.epoch_length_input.setValidator(QDoubleValidator(0.1, 100.0, 2))
        self.epoch_length_input.setMaximumWidth(100)
        self.epoch_length_input.returnPressed.connect(self.update_all_plots)  # Auto-update on Enter
        epoch_controls.addWidget(self.epoch_length_input)
        nav_layout.addRow("Epoch Length:", epoch_controls)
        
        # Mosaic selection (only show if MOSAIC_USAGE is True)
        if self.MOSAIC_USAGE:
            mosaic_controls = QHBoxLayout()
            self.mosaic_combo = QComboBox()
            self.populate_mosaic_combo()
            mosaic_controls.addWidget(QLabel("Mosaic Pair:"))
            mosaic_controls.addWidget(self.mosaic_combo)
            nav_layout.addRow("Active Mosaic:", mosaic_controls)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("←")
        self.prev_button.clicked.connect(self.go_previous_epoch)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("→")
        self.next_button.clicked.connect(self.go_next_epoch)
        button_layout.addWidget(self.next_button)
        
        self.update_button = QPushButton("🔄")
        self.update_button.clicked.connect(self.update_all_plots)
        button_layout.addWidget(self.update_button)
        
        self.recalc_sigma_button = QPushButton("🔄σ")
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
        
        # Scaling control buttons
        scaling_buttons = QHBoxLayout()
        apply_scaling_button = QPushButton("Apply Scaling")
        apply_scaling_button.clicked.connect(self.apply_scaling_changes)
        apply_scaling_button.setToolTip("Apply the new scaling parameters to the plot")
        scaling_buttons.addWidget(apply_scaling_button)
        
        reset_scaling_button = QPushButton("🔄 Reset Y-Scale")
        reset_scaling_button.clicked.connect(self.reset_fixed_scaling)
        reset_scaling_button.setToolTip("Reset fixed y-axis scaling to current parameters")
        scaling_buttons.addWidget(reset_scaling_button)
        
        scaling_layout.addRow("Controls:", scaling_buttons)
        
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
        
        # Individual electrode selection controls (only show in mosaic mode)
        if self.MOSAIC_USAGE:
            selection_group = QGroupBox("Individual Electrode Analysis (Auto-Generated)")
            selection_layout = QHBoxLayout(selection_group)
            
            selection_layout.addWidget(QLabel("Select Mosaic:"))
            self.individual_mosaic_combo = QComboBox()
            self.populate_individual_mosaic_combo()
            # Connect the combo box to auto-update plots when selection changes
            self.individual_mosaic_combo.currentIndexChanged.connect(self.update_individual_plots_if_selected)
            selection_layout.addWidget(self.individual_mosaic_combo)
            
            plot_layout.addWidget(selection_group)
        
        # Create a splitter for main plot and individual analysis
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Main plot widget
        main_plot_widget = QWidget()
        main_plot_layout = QVBoxLayout(main_plot_widget)
        if self.MOSAIC_USAGE:
            main_plot_layout.addWidget(QLabel("All Mosaic Relationships (3 Epochs)"))
        else:
            main_plot_layout.addWidget(QLabel("All Electrodes (3 Epochs)"))
        
        # Matplotlib figure for data
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        main_plot_layout.addWidget(self.canvas)
        
        plot_splitter.addWidget(main_plot_widget)
        
        # Individual analysis widget (only in mosaic mode - electrode plots and STFT spectrogram)
        if self.MOSAIC_USAGE:
            individual_widget = QWidget()
            individual_layout = QVBoxLayout(individual_widget)
            
            # Create a splitter for electrode plots and STFT
            individual_splitter = QSplitter(Qt.Orientation.Vertical)
            
            # Electrode plots section
            electrode_widget = QWidget()
            electrode_layout = QVBoxLayout(electrode_widget)        
            # Combined electrode plot (working and reference in same figure)
            self.individual_figure = Figure(figsize=(12, 4), tight_layout=True)
            self.individual_figure.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)
            self.individual_canvas = FigureCanvas(self.individual_figure)
            electrode_layout.addWidget(self.individual_canvas)
            
            individual_splitter.addWidget(electrode_widget)
            
            # STFT section
            stft_widget = QWidget()
            stft_layout = QVBoxLayout(stft_widget)
            
            # STFT Spectrogram controls and plot
            stft_group = QGroupBox("STFT Spectrogram Controls")
            stft_controls_layout = QHBoxLayout(stft_group)
            
            # Power range display
            self.power_range_label = QLabel("Power Range: -30dB to +30dB")
            stft_controls_layout.addWidget(self.power_range_label)
            
            # Dimmer button (narrow range - increase lower bound)
            self.dimmer_button = QPushButton("🔅 Dimmer (+2dB lower)")
            self.dimmer_button.clicked.connect(self.decrease_stft_range)
            self.dimmer_button.setToolTip("Narrow power range by increasing lower bound by 2dB")
            stft_controls_layout.addWidget(self.dimmer_button)
            
            # Brighter button (widen range - decrease lower bound)  
            self.brighter_button = QPushButton("🔆 Brighter (-2dB lower)")
            self.brighter_button.clicked.connect(self.increase_stft_range)
            self.brighter_button.setToolTip("Widen power range by decreasing lower bound by 2dB")
            stft_controls_layout.addWidget(self.brighter_button)
            
            # Reset button
            self.reset_stft_button = QPushButton("🔄 Reset")
            self.reset_stft_button.clicked.connect(self.reset_stft_range)
            self.reset_stft_button.setToolTip("Reset power range to default (-30dB to +30dB)")
            stft_controls_layout.addWidget(self.reset_stft_button)
            
            stft_layout.addWidget(stft_group)
            
            # STFT Spectrogram plot
            self.spectrogram_figure = Figure(figsize=(12, 4), tight_layout=True)
            self.spectrogram_figure.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.15)
            self.spectrogram_canvas = FigureCanvas(self.spectrogram_figure)
            stft_layout.addWidget(QLabel("STFT Spectrogram (2 epochs before to 2 epochs after current)"))
            stft_layout.addWidget(self.spectrogram_canvas)
            
            individual_splitter.addWidget(stft_widget)
            
            # Set proportions for individual splitter (electrode:STFT = 1.5:3.5)
            # Using values proportional to 1.5:3.5 = 150:350
            individual_splitter.setSizes([150, 350])
            
            individual_layout.addWidget(individual_splitter)
            plot_splitter.addWidget(individual_widget)
            
            # Set splitter proportions (main plot:individual_analysis = 5:5)
            # Using values proportional to 5:5 = 500:500
            plot_splitter.setSizes([500, 500])
        
        plot_layout.addWidget(plot_splitter)
        
        splitter.addWidget(plot_widget)
        
        # Set splitter proportions
        splitter.setSizes([350, 1050])
        main_layout.addWidget(splitter)
        
        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
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
        
        # Apply text edit styles
        for widget in self.findChildren(QTextEdit):
            widget.setStyleSheet(THEME_STYLES.get('text_edit', ''))
        
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
            # Update appropriate plot based on mode
            if self.MOSAIC_USAGE:
                self.update_mosaic_plot()
            else:
                self.update_single_electrode_plot()
        except ValueError:
            pass
    
    def decrease_spacing(self):
        """Decrease channel spacing by 0.5 sigma."""
        try:
            current_value = float(self.spacing_multiplier_input.text() or "5")
            new_value = max(current_value - 0.5, 0.1)  # Minimum 0.1
            self.spacing_multiplier_input.setText(f"{new_value:.1f}")
            # Update appropriate plot based on mode
            if self.MOSAIC_USAGE:
                self.update_mosaic_plot()
            else:
                self.update_single_electrode_plot()
        except ValueError:
            pass
    
    def decrease_stft_range(self):
        """Narrow STFT power range by increasing lower bound by 2dB (dimmer)."""
        self.stft_power_min += 2
        self.update_stft_power_range_display()
        if self.current_stft_data:
            self.replot_stft_spectrogram()
    
    def increase_stft_range(self):
        """Widen STFT power range by decreasing lower bound by 2dB (brighter)."""
        self.stft_power_min -= 2
        self.update_stft_power_range_display()
        if self.current_stft_data:
            self.replot_stft_spectrogram()
    
    def reset_stft_range(self):
        """Reset STFT power range to default values."""
        self.stft_power_min = -30
        self.stft_power_max = 30
        self.update_stft_power_range_display()
        if self.current_stft_data:
            self.replot_stft_spectrogram()
    
    def update_stft_power_range_display(self):
        """Update the STFT power range label."""
        self.power_range_label.setText(f"Power Range: {self.stft_power_min}dB to {self.stft_power_max}dB")
    
    def replot_stft_spectrogram(self):
        """Replot the STFT spectrogram with current power range settings."""
        if not self.current_stft_data:
            return
            
        times, frequencies, power_spectrum_db, start_time, epoch_length, working_electrode, spectrogram_start, spectrogram_end = self.current_stft_data
        
        # Clear current figure
        self.spectrogram_figure.clear()
        ax = self.spectrogram_figure.add_subplot(1, 1, 1)
        
        # Plot spectrogram with fixed power range
        im = ax.pcolormesh(times, frequencies, power_spectrum_db, shading='gouraud', cmap='viridis',
                          vmin=self.stft_power_min, vmax=self.stft_power_max)
        
        # Add colorbar
        cbar = self.spectrogram_figure.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', rotation=270, labelpad=15)
        
        # Mark current epoch boundaries with red dashed lines
        ax.axvline(start_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(start_time + epoch_length, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Labels and title
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 30)
        
        # Adjust layout to remove margins and fit window size
        self.spectrogram_figure.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.15)
        self.spectrogram_figure.tight_layout(pad=0.5)
        self.spectrogram_canvas.draw()
        
    def populate_mosaic_combo(self):
        """Populate the mosaic selection combo box with relationships in original order."""
        if not self.MOSAIC_USAGE:
            return  # Skip if not in mosaic mode
        
        self.mosaic_combo.clear()
        # Use original order, no sorting
        for rel in self.mosaic_relationships:
            display_text = f"{rel['name']} (E{rel['electrode_a']} - E{rel['electrode_b']})"
            self.mosaic_combo.addItem(display_text)
    
    def populate_individual_mosaic_combo(self):
        """Populate the individual mosaic selection combo box with relationships sorted by name."""
        if not self.MOSAIC_USAGE:
            return  # Skip if not in mosaic mode
        
        self.individual_mosaic_combo.clear()
        # Use original order, no sorting
        for rel in self.mosaic_relationships:
            display_text = f"{rel['name']} (Working: E{rel['electrode_a']}, Ref: E{rel['electrode_b']})"
            self.individual_mosaic_combo.addItem(display_text, rel)  # Store the relationship data
            
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
            
            # Update plots based on mode
            if self.MOSAIC_USAGE:
                self.update_mosaic_plot()
                
                # Populate combo boxes for mosaic mode
                self.populate_individual_mosaic_combo()
                
                # Auto-select first mosaic if available and plot individual electrodes
                if self.individual_mosaic_combo.count() > 0:
                    self.individual_mosaic_combo.setCurrentIndex(0)
                    self.update_individual_plots_if_selected()
            else:
                # Single electrode mode - just update the main plot
                self.update_single_electrode_plot()
            
        except Exception as e:
            print(f"Error initializing labeling window data: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Error", f"Error processing data: {str(e)}")
    
    def calculate_global_sigma(self):
        """Calculate global sigma from all available data for consistent scaling."""
        try:
            print("Calculating global sigma from all available data...")
            
            # Use a larger time window to sample more data for statistics
            sample_duration = min(10.0, self.total_duration)  # Sample up to 10 seconds
            sample_start = 0.0
            sample_end = sample_duration
            
            all_values = []
            
            if self.MOSAIC_USAGE:
                # Mosaic mode: Calculate from mosaic differential data
                for relationship in self.mosaic_relationships:
                    time_array, mosaic_data = self.calculate_mosaic_data(relationship, sample_start, sample_end)
                    if time_array is not None and mosaic_data is not None:
                        all_values.extend(mosaic_data)
            else:
                # Single electrode mode: Calculate from individual electrode data
                for electrode in self.electrode_positions:
                    electrode_number = electrode['number']
                    time_array, electrode_data = self.get_channel_data_for_electrode(electrode_number, sample_start, sample_end)
                    if time_array is not None and electrode_data is not None:
                        all_values.extend(electrode_data)
            
            if all_values:
                self.global_sigma = np.std(all_values)
                print(f"Global sigma calculated: {self.global_sigma:.2f} μV")
            else:
                self.global_sigma = 100.0  # Default fallback value
                print("No data available for global sigma calculation, using default: 100.0 μV")
                
        except Exception as e:
            print(f"Error calculating global sigma: {e}")
            import traceback
            traceback.print_exc()
            self.global_sigma = 100.0  # Default fallback value
    
    def recalculate_global_sigma(self):
        """Recalculate global sigma and update the plot."""
        self.calculate_global_sigma()
        # Reset fixed scaling since sigma changed
        self.reset_fixed_scaling()
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
    
    def apply_scaling_changes(self):
        """Apply scaling changes and reset fixed y-axis limits."""
        # Reset fixed scaling to force recalculation with new parameters
        self.reset_fixed_scaling()
        # Update the plot with new scaling based on mode
        if self.MOSAIC_USAGE:
            self.update_mosaic_plot()
        else:
            self.update_single_electrode_plot()
    
    def reset_fixed_scaling(self):
        """Reset the fixed y-axis scaling to allow recalculation."""
        self.fixed_y_limits = None
        self.fixed_total_height = None
        print("Reset fixed y-axis scaling - will recalculate on next plot update")
    
    def add_epoch_markers(self, ax, plot_start, plot_end, start_time, baseline_epoch_length):
        """Add only the red dashed epoch divider line."""
        try:
            # Add only vertical dashed line at start_time (current epoch marker)
            ax.axvline(start_time, color='red', linestyle='--', alpha=1.0, linewidth=2)
            ax.axvline(start_time + float(self.epoch_length_input.text() or "4.0"), color='red', linestyle='--', alpha=1.0, linewidth=2)
            
            
        except Exception as e:
            print(f"Warning: Could not add epoch markers: {e}")
    
    def plot_individual_electrodes(self):
        """Plot individual working and reference electrodes in a single figure for the selected mosaic pair."""
        # Only run in mosaic mode
        if not self.MOSAIC_USAGE:
            return
        
        try:
            # Get selected mosaic relationship
            current_index = self.individual_mosaic_combo.currentIndex()
            if current_index < 0:
                # Clear plot if no selection
                self.individual_figure.clear()
                self.individual_canvas.draw()
                return
            
            relationship = self.individual_mosaic_combo.itemData(current_index)
            if not relationship:
                # Clear plot if invalid selection
                self.individual_figure.clear()
                self.individual_canvas.draw()
                return
            
            # Get current time parameters
            start_time = float(self.start_time_input.text() or "0.0")
            baseline_epoch_length = float(self.epoch_length_input.text() or "4.0")
            
            # Plot data for the current 3-epoch range
            plot_start = max(0, start_time - baseline_epoch_length)
            plot_end = start_time + (2 * baseline_epoch_length)
            
            # Get electrode data
            working_electrode = relationship.get('electrode_a')
            reference_electrode = relationship.get('electrode_b')
            mosaic_name = relationship.get('name', f'E{working_electrode}-E{reference_electrode}')
            
            print(f"Plotting individual electrodes: Working E{working_electrode}, Reference E{reference_electrode}")
            
            # Get data for both electrodes
            working_time, working_data = self.get_channel_data_for_electrode(working_electrode, plot_start, plot_end)
            reference_time, reference_data = self.get_channel_data_for_electrode(reference_electrode, plot_start, plot_end)
            
            if working_time is None or reference_time is None:
                print("Warning: Could not retrieve data for the selected electrodes.")
                # Clear plot
                self.individual_figure.clear()
                self.individual_canvas.draw()
                return
            
            # Create DataFrames for the standard plot function
            # Combine both electrodes in a single DataFrame
            working_channel_name = f"Working E{working_electrode}"
            reference_channel_name = f"Reference E{reference_electrode}"
            
            # Ensure both time arrays have the same length
            min_length = min(len(working_time), len(reference_time))
            combined_df = pd.DataFrame({
                'time': working_time[:min_length],
                working_channel_name: working_data[:min_length],
                reference_channel_name: reference_data[:min_length]
            })
            
            # Use the widget plotting function with same parameters as mosaic plot
            from utils.widget import plot_standard_eeg_data
            
            # Get user-defined sigma multipliers to match mosaic plot
            try:
                sigma_multiplier = float(self.spacing_multiplier_input.text() or "5")
                y_range_multiplier = float(self.ylim_multiplier_input.text() or "5")
            except ValueError:
                sigma_multiplier = 5
                y_range_multiplier = 5
            
            # Plot using standard function with same scaling as mosaic plot
            plot_standard_eeg_data(
                self.individual_figure, combined_df, {},
                sigma_multiplier=sigma_multiplier, 
                y_range_multiplier=y_range_multiplier,
                global_sigma=self.global_sigma,
                title=False
            )
            
            # Add epoch boundary markers
            ax = self.individual_figure.get_axes()[0]
            self.add_epoch_markers(ax, plot_start, plot_end, start_time, baseline_epoch_length)
            
            # Adjust layout to remove margins and fit window size
            self.individual_figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.1)
            self.individual_figure.tight_layout(pad=0.5)
            
            # # Update title to show mosaic name and electrode info
            # start_time_display = working_time[0] if len(working_time) > 0 else plot_start
            # end_time_display = working_time[-1] if len(working_time) > 0 else plot_end
            # title = f'{mosaic_name}: Working E{working_electrode} & Reference E{reference_electrode} ({start_time_display:.2f}s - {end_time_display:.2f}s)'
            # self.individual_figure.suptitle(title, fontsize=10, fontweight='bold')
            
            self.individual_canvas.draw()
            
            print("Individual electrode plot completed successfully")
            
        except Exception as e:
            print(f"Error plotting individual electrodes: {e}")
            traceback.print_exc()
            # Clear plot on error
            self.individual_figure.clear()
            self.individual_canvas.draw()
    
    def plot_stft_spectrogram(self):
        """Plot STFT spectrogram for the working electrode covering 2 epochs before to 2 epochs after current."""
        # Only run in mosaic mode
        if not self.MOSAIC_USAGE:
            return
        
        try:
            # Get selected mosaic relationship
            current_index = self.individual_mosaic_combo.currentIndex()
            if current_index < 0:
                # No selection - clear the plot
                self.spectrogram_figure.clear()
                self.spectrogram_canvas.draw()
                return
            
            relationship = self.individual_mosaic_combo.itemData(current_index)
            if not relationship:
                # Invalid selection - clear the plot
                self.spectrogram_figure.clear()
                self.spectrogram_canvas.draw()
                return
            
            # Get current time parameters
            start_time = float(self.start_time_input.text() or "0.0")
            baseline_epoch_length = float(self.epoch_length_input.text() or "4.0")
            
            # Calculate time range: 2 epochs before to 2 epochs after current epoch
            spectrogram_start = max(0, start_time - 2 * baseline_epoch_length)  # 2 epochs before
            spectrogram_end = start_time + 3 * baseline_epoch_length  # 2 epochs after (current + 2 more)
            
            # Clamp to available data
            if self.total_duration > 0:
                spectrogram_end = min(spectrogram_end, self.total_duration)
                # Ensure we have at least 5 epochs if possible
                min_duration = 5 * baseline_epoch_length
                if spectrogram_end - spectrogram_start < min_duration:
                    spectrogram_start = max(0, spectrogram_end - min_duration)
            
            working_electrode = relationship.get('electrode_a')
            print(f"Plotting STFT spectrogram for working electrode E{working_electrode}")
            print(f"Time range: {spectrogram_start:.2f}s to {spectrogram_end:.2f}s")
            
            # Get data for working electrode
            time_array, electrode_data = self.get_channel_data_for_electrode(working_electrode, spectrogram_start, spectrogram_end)
            
            if time_array is None or electrode_data is None:
                print("Warning: Could not retrieve data for STFT analysis.")
                self.spectrogram_figure.clear()
                self.spectrogram_canvas.draw()
                return
            
            # Perform STFT
            # STFT parameters
            fs = self.sample_rate or 30000  # Sample rate
            nperseg = int(baseline_epoch_length * fs)  # Window size = epoch length
            noverlap = int(nperseg * 0.75)  # 75% overlap
            
            print(f"STFT parameters: fs={fs}, nperseg={nperseg}, noverlap={noverlap}")
            
            # Compute STFT
            frequencies, times, Zxx = signal.stft(electrode_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # Filter frequencies to 0-30 Hz range
            freq_mask = frequencies <= 30
            frequencies = frequencies[freq_mask]
            Zxx = Zxx[freq_mask, :]
            
            # Convert to power spectrum (magnitude squared)
            power_spectrum = np.abs(Zxx)**2
            
            # Convert to dB scale
            power_spectrum_db = 10 * np.log10(power_spectrum + 1e-12)  # Add small value to avoid log(0)
            
            # Adjust time array to match spectrogram_start
            times = times + spectrogram_start
            
            # Store current data for re-plotting when power range changes
            self.current_stft_data = (times, frequencies, power_spectrum_db, start_time, baseline_epoch_length, working_electrode, spectrogram_start, spectrogram_end)
            
            # Plot the spectrogram
            self.replot_stft_spectrogram()
            
            print("STFT spectrogram completed successfully")
            
        except Exception as e:
            print(f"Error plotting STFT spectrogram: {e}")
            traceback.print_exc()
            # Clear plot on error
            self.spectrogram_figure.clear()
            self.spectrogram_canvas.draw()
    
    def update_mosaic_plot(self):
        """Update the mosaic data plot showing 3 epochs with current epoch highlighted."""
        # Don't run if not in mosaic mode
        if not self.MOSAIC_USAGE:
            return
        
        try:
            # Get current parameters
            start_time = float(self.start_time_input.text() or "0.0")
            baseline_epoch_length = float(self.epoch_length_input.text() or "4.0")  # Default to 4 seconds
            
            # Calculate 3-epoch time range (12 seconds total with 4-second epochs)
            total_plot_duration = 3 * baseline_epoch_length  # 12 seconds
            
            # Current epoch is in the middle, so we show 1 epoch before and 1 epoch after
            plot_start = max(0, start_time - baseline_epoch_length)  # Don't go below 0
            plot_end = start_time + (2 * baseline_epoch_length)
            
            # Clamp plot_end to available data duration if we know it
            if self.total_duration > 0:
                plot_end = min(plot_end, self.total_duration)
                # If we can't show the full 3 epochs at the end, adjust start time
                if plot_end - plot_start < total_plot_duration:
                    plot_start = max(0, plot_end - total_plot_duration)
            
            # Ensure valid time range
            if plot_start >= plot_end:
                # Adjust if start time is near the end of data
                if start_time >= self.total_duration - baseline_epoch_length:
                    plot_end = self.total_duration
                    plot_start = max(0, plot_end - total_plot_duration)
                else:
                    plot_start = 0
                    plot_end = total_plot_duration
            
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

            print(f"Debug: Plotting relationships in original order: {[rel.get('name', '') for rel in valid_relationships]}")

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
            
            # Calculate total height for scale bars and set fixed Y-axis limits FIRST
            total_height = (len(valid_relationships) - 1) * channel_spacing + 2 * y_range
            y_center = 0
            
            # Use fixed y-limits if they exist, otherwise calculate and store them
            if self.fixed_y_limits is None or self.fixed_total_height is None:
                # First time or recalculation - set and store the limits
                self.fixed_total_height = total_height
                y_bottom = y_center - total_height/2
                y_top = y_center + total_height/2
                self.fixed_y_limits = (y_bottom, y_top)
                print(f"Setting fixed y-limits: {y_bottom:.1f} to {y_top:.1f}")
            else:
                # Use stored fixed limits for consistent scaling
                y_bottom, y_top = self.fixed_y_limits
                total_height = self.fixed_total_height
            
            # Set Y-axis limits BEFORE plotting anything to ensure consistent scaling
            ax.set_ylim(y_bottom, y_top)
            
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
            
            # Add vertical lines at epoch boundaries (gray background lines)
            # Epoch boundaries are at: plot_start + baseline_epoch_length, plot_start + 2*baseline_epoch_length
            epoch_boundary_1 = plot_start + baseline_epoch_length
            epoch_boundary_2 = plot_start + 2 * baseline_epoch_length
            
            # # Add only vertical dashed line at the center time (start_time) to mark current epoch
            # ax.axvline(start_time, color='red', linestyle='--', alpha=1.0, linewidth=2)
            
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
                self.add_scale_bars(ax, time_array, global_sigma, 0, self.fixed_total_height)
            
            # Add overall title (matching widget.py style)
            title = f'3-Epoch Mosaic Data: {plot_start:.2f}s - {plot_end:.2f}s (Current: {start_time:.2f}s-{start_time + baseline_epoch_length:.2f}s, σ={global_sigma:.1f}μV)'
            self.figure.suptitle(title, fontsize=12, fontweight='bold')
            
            # Update current parameters
            self.current_start_time = start_time
            self.current_epoch_length = baseline_epoch_length
            
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
        
        # Update plots based on mode
        if self.MOSAIC_USAGE:
            self.update_mosaic_plot()
            self.update_individual_plots_if_selected()
        else:
            self.update_single_electrode_plot()
        
    def go_next_epoch(self):
        """Navigate to next epoch."""
        new_start = self.current_start_time + self.current_epoch_length
        
        # Check against total duration
        max_end = new_start + self.current_epoch_length
        if max_end > self.total_duration:
            new_start = max(0, self.total_duration - 2 * self.current_epoch_length)
            
        self.start_time_input.setText(f"{new_start:.3f}")
        
        # Update plots based on mode
        if self.MOSAIC_USAGE:
            self.update_mosaic_plot()
            self.update_individual_plots_if_selected()
        else:
            self.update_single_electrode_plot()
    
    def update_individual_plots_if_selected(self):
        """Update individual electrode plots and STFT spectrogram if a mosaic pair is currently selected."""
        if not self.MOSAIC_USAGE:
            return  # Skip if not in mosaic mode
        
        try:
            # Check if a mosaic is selected
            current_index = self.individual_mosaic_combo.currentIndex()
            if current_index >= 0:
                # Auto-update individual plots and spectrogram
                self.plot_individual_electrodes()
                self.plot_stft_spectrogram()
                
        except Exception as e:
            print(f"Warning: Could not auto-update individual plots: {e}")
    
    def update_all_plots(self):
        """Update plots based on MOSAIC_USAGE mode."""
        if self.MOSAIC_USAGE:
            # Mosaic mode: plot differential signals between electrode pairs
            self.update_mosaic_plot()
            self.update_individual_plots_if_selected()
        else:
            # Single electrode mode: plot individual electrodes like plotting_window.py
            self.update_single_electrode_plot()
    
    def update_single_electrode_plot(self):
        """Update plot showing individual electrodes (non-mosaic mode)."""
        try:
            # Get current parameters
            start_time = float(self.start_time_input.text() or "0.0")
            baseline_epoch_length = float(self.epoch_length_input.text() or "4.0")
            
            # Calculate 3-epoch time range
            total_plot_duration = 3 * baseline_epoch_length
            plot_start = max(0, start_time - baseline_epoch_length)
            plot_end = start_time + (2 * baseline_epoch_length)
            
            # Clamp plot_end to available data duration
            if self.total_duration > 0:
                plot_end = min(plot_end, self.total_duration)
                if plot_end - plot_start < total_plot_duration:
                    plot_start = max(0, plot_end - total_plot_duration)
            
            # Ensure valid time range
            if plot_start >= plot_end:
                if start_time >= self.total_duration - baseline_epoch_length:
                    plot_end = self.total_duration
                    plot_start = max(0, plot_end - total_plot_duration)
                else:
                    plot_start = 0
                    plot_end = total_plot_duration
            
            if not self.electrode_positions:
                self.show_message("No Electrodes", "No electrode positions available.")
                return
            
            print(f"Debug: Plotting {len(self.electrode_positions)} electrodes")
            print(f"Debug: Time range: {plot_start:.3f}s to {plot_end:.3f}s")
            
            # Clear previous plots
            self.figure.clear()
            
            # Collect data for all electrodes
            electrode_data_dict = {}
            time_array = None
            
            for electrode in self.electrode_positions:
                electrode_number = electrode['number']
                print(f"Debug: Fetching data for electrode E{electrode_number}")
                t, data = self.get_channel_data_for_electrode(electrode_number, plot_start, plot_end)
                if t is not None and data is not None:
                    print(f"Debug: Got {len(data)} samples for E{electrode_number}")
                    electrode_data_dict[f'E{electrode_number}'] = data
                    if time_array is None:
                        time_array = t
                else:
                    print(f"Warning: No data for electrode E{electrode_number}")
            
            if not electrode_data_dict or time_array is None:
                print(f"Error: No electrode data collected. Dict size: {len(electrode_data_dict)}, Time array: {time_array is not None}")
                self.show_message("No Data", "Could not load electrode data for the selected time range.")
                return
            
            print(f"Debug: Collected data for {len(electrode_data_dict)} electrodes")
            
            # Create DataFrame for plotting - be more careful about data alignment
            # First, ensure all data arrays have the same length
            min_length = len(time_array)
            for key, data in list(electrode_data_dict.items()):
                if len(data) < min_length:
                    print(f"Warning: {key} has only {len(data)} samples, padding to {min_length}")
                    # Pad with the last value
                    padded = np.pad(data, (0, min_length - len(data)), mode='edge')
                    electrode_data_dict[key] = padded
                elif len(data) > min_length:
                    print(f"Warning: {key} has {len(data)} samples, truncating to {min_length}")
                    electrode_data_dict[key] = data[:min_length]
            
            # Build DataFrame with all columns at once
            df_dict = {'time': time_array}
            df_dict.update(electrode_data_dict)
            df_data = pd.DataFrame(df_dict)
            
            print(f"Debug: DataFrame created with shape {df_data.shape}, columns: {list(df_data.columns)}")
            
            # Get user-defined sigma multipliers
            try:
                sigma_multiplier = float(self.spacing_multiplier_input.text() or "5")
                y_range_multiplier = float(self.ylim_multiplier_input.text() or "5")
            except ValueError:
                sigma_multiplier = 5
                y_range_multiplier = 5
            
            # Plot using standard function (same as plotting_window.py)
            from utils.widget import plot_standard_eeg_data
            plot_standard_eeg_data(
                self.figure, df_data, {},
                sigma_multiplier=sigma_multiplier,
                y_range_multiplier=y_range_multiplier,
                global_sigma=self.global_sigma,
                title=False
            )
            
            # Add epoch boundary markers if axes were created
            axes = self.figure.get_axes()
            if axes and len(axes) > 0:
                ax = axes[0]
                self.add_epoch_markers(ax, plot_start, plot_end, start_time, baseline_epoch_length)
            
            # Add title
            title = f'3-Epoch Electrode Data: {plot_start:.2f}s - {plot_end:.2f}s (Current: {start_time:.2f}s-{start_time + baseline_epoch_length:.2f}s)'
            if self.global_sigma:
                title += f', σ={self.global_sigma:.1f}μV'
            self.figure.suptitle(title, fontsize=12, fontweight='bold')
            
            # Update current parameters
            self.current_start_time = start_time
            self.current_epoch_length = baseline_epoch_length
            
            # Refresh canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update info display
            self.update_info_display()
            
        except Exception as e:
            print(f"Error updating single electrode plot: {e}")
            import traceback
            traceback.print_exc()
            self.show_message("Error", f"Error updating plot: {str(e)}")
    
    def update_info_display(self):
        """Update the information display."""
        try:
            start_time = float(self.start_time_input.text() or "0.0")
            baseline_epoch_length = float(self.epoch_length_input.text() or "4.0")
            
            # Calculate actual plot range with boundary checks (3 epochs total)
            total_plot_duration = 3 * baseline_epoch_length
            plot_start = max(0, start_time - baseline_epoch_length)
            plot_end = start_time + (2 * baseline_epoch_length)
            if self.total_duration > 0:
                plot_end = min(plot_end, self.total_duration)
                if plot_end - plot_start < total_plot_duration:
                    plot_start = max(0, plot_end - total_plot_duration)
            
            # Calculate epoch boundaries
            current_epoch_start = start_time
            current_epoch_end = start_time + baseline_epoch_length
            
            # Get current scaling parameters
            try:
                spacing_multiplier = float(self.spacing_multiplier_input.text() or "5")
                ylim_multiplier = float(self.ylim_multiplier_input.text() or "5")
            except ValueError:
                spacing_multiplier = 5
                ylim_multiplier = 5
            
            # Base info lines (common to both modes)
            if self.MOSAIC_USAGE:
                info_lines = [
                    f"📊 Mosaic Data Analysis - 3 Epoch View",
                    f"═══════════════════════════════════════",
                ]
            else:
                info_lines = [
                    f"📊 Electrode Data Analysis - 3 Epoch View",
                    f"═══════════════════════════════════════",
                ]
            
            info_lines.extend([
                f"⏱️ Total Duration: {self.total_duration:.2f}s",
                f"📡 Sample Rate: {self.sample_rate} Hz" if self.sample_rate else "📡 Sample Rate: Unknown",
                f"📁 Files: {len(self.file_metadata)}",
                f"📏 Global Sigma: {self.global_sigma:.2f} μV" if self.global_sigma else "📏 Global Sigma: Not calculated",
                "",
                f"📐 Plot Scaling:",
                f"  • Channel Spacing: {spacing_multiplier}σ",
                f"  • Y-axis Range: {ylim_multiplier}σ",
                "",
                f"🎯 3-Epoch Analysis:",
                f"  • Baseline Epoch Length: {baseline_epoch_length:.2f}s",
                f"  • Total Plot Duration: {total_plot_duration:.2f}s (3 epochs)",
                f"  • Plot Range: {plot_start:.3f}s to {plot_end:.3f}s",
                f"  • 🔴 Current Epoch: {current_epoch_start:.3f}s to {current_epoch_end:.3f}s",
                f"  • Epoch Boundaries: {plot_start + baseline_epoch_length:.3f}s, {plot_start + 2*baseline_epoch_length:.3f}s",
                "",
            ])
            
            # Mode-specific information
            if self.MOSAIC_USAGE:
                # Mosaic mode: show mosaic relationships
                info_lines.append(f"🔗 All Mosaic Relationships ({len(self.mosaic_relationships)}) - Original Order:")
                
                # Use original order, no sorting
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
                    "📊 Individual Analysis (Auto-generated):",
                    "  • Working & Reference electrodes in single plot",
                    "  • STFT Spectrogram opens in separate window",
                    "  • STFT shows 2 epochs before to 2 epochs after current",
                    "  • Select different mosaic pair to change analysis focus",
                    "",
                    "🎛️ Scaling Controls:",
                    "  • Adjust Channel Spacing: σ multiplier for vertical separation",
                    "  • Adjust Y-axis Range: σ multiplier for plot height",
                    "  • Apply Scaling: Update plot with new parameters",
                    "  • + Key: Increase channel spacing by 0.5σ",
                    "  • - Key: Decrease channel spacing by 0.5σ"
                ])
            else:
                # Single electrode mode: show electrode list
                info_lines.append(f"🔌 All Electrodes ({len(self.electrode_positions)}):")
                
                for electrode in self.electrode_positions:
                    electrode_num = electrode['number']
                    x, y = electrode.get('x', 0), electrode.get('y', 0)
                    info_lines.append(f"  • E{electrode_num}: Position ({x:.2f}, {y:.2f})")
                    
                    # Get channel mapping info if available
                    channel = self.channel_mapping.get(electrode_num)
                    if channel is not None:
                        info_lines.append(f"    Channel: {channel}")
                    else:
                        info_lines.append(f"    ⚠️ Channel: Not mapped")
                
                info_lines.extend([
                    "",
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
