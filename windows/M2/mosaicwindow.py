"""
Mosaic Data Plotter Window for MEEG Seizure Labeling.

This module displays the EEG montage data similar to Figure1D.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class MosaicPlotterWidget(QWidget):
    """Widget for displaying EEG mosaic montage data."""
    
    # Signal emitted when epoch is clicked
    epoch_clicked = pyqtSignal(int)  # epoch index
    # Signal emitted when epoch is double-clicked (for video tab switch)
    epoch_double_clicked = pyqtSignal(int)  # epoch index
    
    def __init__(self, df=None, mosaic_relationships=None, sampling_rate=None, theme_colors=None, parent=None):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.mosaic_relationships = mosaic_relationships or []
        self.sampling_rate = sampling_rate or 1000.0
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.epoch_length = 1  # seconds
        self.current_epoch = 0
        self.epochs_to_show = 15
        
        # Display parameters
        self.spacing = 300  # spacing within each montage pair (µV)
        self.spacing_cluster = 600  # spacing between montage clusters (µV)
        self.limit = 400  # Y-axis margin
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("EEG Montage Display")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create matplotlib figure with white background
        self.figure = Figure(figsize=(12, 6), facecolor='white', edgecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        layout.addWidget(self.canvas)
        
        # Initialize the plot
        self.update_plot()
        
    def on_click(self, event):
        """Handle mouse click events on the plot."""
        if event.inaxes == self.ax and event.xdata is not None:
            # Calculate which epoch was clicked
            clicked_epoch = int(np.clip(event.xdata / self.epoch_length, 0, self.get_n_epochs() - 1))
            self.epoch_clicked.emit(clicked_epoch)
            
            # Check for double-click
            if event.dblclick:
                self.epoch_double_clicked.emit(clicked_epoch)
            
    def get_n_epochs(self):
        """Get total number of epochs in the data."""
        if self.df.empty or self.sampling_rate == 0:
            return 0
        total_samples = len(self.df)
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        return max(1, total_samples // samples_per_epoch)
        
    def update_plot(self):
        """Update the montage plot display."""
        self.ax.clear()
        self.ax.set_facecolor('white')
        
        if self.df.empty or not self.mosaic_relationships:
            self.ax.text(0.5, 0.5, 'No data to display', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', color='black', fontsize=12)
            self.canvas.draw()
            return
            
        # Calculate time range to display
        start_epoch = max(0, self.current_epoch - self.epochs_to_show // 2)
        end_epoch = min(self.get_n_epochs(), start_epoch + self.epochs_to_show)
        start_epoch = max(0, end_epoch - self.epochs_to_show)
        
        start_sample = int(start_epoch * self.epoch_length * self.sampling_rate)
        end_sample = int(end_epoch * self.epoch_length * self.sampling_rate)
        end_sample = min(end_sample, len(self.df))
        
        if start_sample >= end_sample:
            self.canvas.draw()
            return
            
        # Get data slice
        df_slice = self.df.iloc[start_sample:end_sample]
        time_array = np.arange(len(df_slice)) / self.sampling_rate + start_epoch * self.epoch_length
        
        # Group mosaic relationships (similar to Figure1D.py)
        montage_groups = self.group_mosaic_relationships()
        
        # Plot each montage group
        y_ticks = []
        y_labels = []
        
        # Calculate total pairs for initial offset
        total_pairs = sum(len(group) for group in montage_groups)
        group_pairs_info = [(idx, len(group)) for idx, group in enumerate(montage_groups)]
        
        current_y_offset = total_pairs * self.spacing + (len(group_pairs_info) - 1) * self.spacing_cluster
        
        for info_idx, (group_idx, num_pairs_in_group) in enumerate(group_pairs_info):
            group = montage_groups[group_idx]
            
            for pair_idx_in_group, (mosaic_name, ch1, ch2) in enumerate(group):
                # Check if channels exist in DataFrame
                if ch1 not in df_slice.columns or ch2 not in df_slice.columns:
                    continue
                    
                # Get data for both channels
                data_ch1 = df_slice[ch1].values
                data_ch2 = df_slice[ch2].values
                
                # Calculate montage: ch1 - ch2
                montage_data = data_ch1 - data_ch2
                
                # Plot the montage data
                self.ax.plot(time_array, montage_data + current_y_offset, 
                           color='black', linewidth=0.8)
                
                # Save y-axis tick and label
                y_ticks.append(current_y_offset)
                y_labels.append(f'{ch1}-{ch2}')
                
                # Move down for next pair
                current_y_offset -= self.spacing
            
            # Move down extra for cluster spacing gap
            if info_idx < len(group_pairs_info) - 1:
                current_y_offset -= (self.spacing_cluster - self.spacing)
        
        # Draw vertical grid lines every 1 second
        duration = (end_epoch - start_epoch) * self.epoch_length
        for t in np.arange(start_epoch * self.epoch_length, 
                          (end_epoch * self.epoch_length) + 1, 1):
            if t % 5 == 0:
                # Every 5 seconds: darker and thicker
                self.ax.axvline(x=t, color='#1a1a1a', alpha=0.8, linewidth=1.5, linestyle='-')
            else:
                # Every 1 second: lighter
                self.ax.axvline(x=t, color='#404040', alpha=0.4, linewidth=0.8, linestyle='-')
        
        # Highlight current epoch
        epoch_start_time = self.current_epoch * self.epoch_length
        epoch_end_time = (self.current_epoch + 1) * self.epoch_length
        self.ax.axvspan(epoch_start_time, epoch_end_time, 
                       alpha=0.2, color='yellow', zorder=0)
        
        # Add scale bar (200 µV)
        if y_ticks:
            scale_bar_height = 200  # µV
            scale_bar_x = start_epoch * self.epoch_length + 0.5
            scale_bar_y_start = y_ticks[0] + self.limit * 0.5
            scale_bar_y_end = scale_bar_y_start - scale_bar_height
            
            self.ax.plot([scale_bar_x, scale_bar_x], 
                       [scale_bar_y_start, scale_bar_y_end], 
                       'r-', linewidth=2)
            self.ax.text(scale_bar_x + 0.3, (scale_bar_y_start + scale_bar_y_end) / 2,
                       '200µV', color='red', fontsize=8, va='center')
        
        # Set axis properties
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(y_labels, fontsize=9, color='black')
        self.ax.set_xlabel('Time (s)', fontsize=10, color='black')
        self.ax.set_ylabel('Montage', fontsize=10, color='black')
        self.ax.set_xlim(start_epoch * self.epoch_length, end_epoch * self.epoch_length)
        
        # Set x-ticks at every 1 second
        x_start = start_epoch * self.epoch_length
        x_end = end_epoch * self.epoch_length
        x_ticks = np.arange(np.ceil(x_start), x_end + 1, 1)
        self.ax.set_xticks(x_ticks)
        
        if y_ticks:
            y_min = min(y_ticks) - self.limit * 1.2
            y_max = max(y_ticks) + self.limit * 1.2
            self.ax.set_ylim(y_min, y_max)
        
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.tick_params(colors='black')
        self.ax.spines['bottom'].set_color('black')
        self.ax.spines['left'].set_color('black')
        self.ax.spines['top'].set_color('black')
        self.ax.spines['right'].set_color('black')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def group_mosaic_relationships(self):
        """Group mosaic relationships into clusters.
        
        Returns a list of groups, where each group is a list of 
        (mosaic_name, electrode_a_name, electrode_b_name) tuples.
        """
        # Simple grouping - you can implement more sophisticated logic
        # For now, treat all relationships as one group
        groups = []
        current_group = []
        
        for mosaic_name, ch1, ch2 in self.mosaic_relationships:
            current_group.append((mosaic_name, ch1, ch2))
            
        if current_group:
            groups.append(current_group)
            
        return groups
        
    def set_epoch(self, epoch_idx):
        """Set the current epoch and update display."""
        self.current_epoch = epoch_idx
        self.update_plot()
        
    def set_epochs_to_show(self, n_epochs):
        """Set number of epochs to display."""
        self.epochs_to_show = max(1, n_epochs)
        self.update_plot()
    
    def set_epoch_length(self, epoch_length):
        """Set the epoch length in seconds."""
        self.epoch_length = epoch_length
        self.update_plot()
        
    def set_data(self, df, mosaic_relationships, sampling_rate):
        """Update the data to display."""
        self.df = df if df is not None else pd.DataFrame()
        self.mosaic_relationships = mosaic_relationships or []
        self.sampling_rate = sampling_rate or 1000.0
        self.update_plot()
