"""
PSD Analysis Widget for EEG Preview.

This module contains the widget for displaying Power Spectral Density (PSD)
for the currently selected epoch.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class PSDWidget(QWidget):
    """Widget for displaying Power Spectral Density (PSD) of the current epoch."""
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.data = None
        self.sampling_rate = None
        self.epoch_length = 5.0
        self.epoch_idx = 0
        
        # Frequency bands for coloring
        self.bands = {
            'Delta': (0.5, 4, '#ffcccc'),   # Light red
            'Theta': (4, 8, '#ffebcc'),     # Light orange
            'Alpha': (8, 13, '#ccffcc'),    # Light green
            'Beta': (13, 30, '#cce5ff'),    # Light blue
            'Gamma': (30, 100, '#e5ccff')   # Light purple
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("Power Spectral Density (PSD)")
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 3), facecolor=self.theme_colors['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout.addWidget(self.canvas)
        
        self.update_plot()
        
    def set_data(self, data, sampling_rate, epoch_length=None):
        """Set the EEG data (full dataset) and parameters.
        
        Parameters:
        -----------
        data : np.ndarray
            Full EEG data array (1D for single channel)
        sampling_rate : float
            Sampling rate in Hz
        epoch_length : float, optional
            Epoch length in seconds
        """
        self.data = data
        self.sampling_rate = sampling_rate
        if epoch_length:
            self.epoch_length = epoch_length
        self.update_plot()
        
    def set_epoch(self, epoch_idx):
        """Set the current epoch index."""
        self.epoch_idx = epoch_idx
        self.update_plot()
        
    def set_epoch_length(self, length):
        """Update epoch length."""
        self.epoch_length = length
        self.update_plot()
        
    def update_plot(self):
        """Compute and plot PSD for the current epoch."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        
        # Style axis
        self.ax.tick_params(colors=self.theme_colors['fg_primary'], labelsize=8)
        self.ax.spines['bottom'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['left'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['top'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['right'].set_color(self.theme_colors['fg_primary'])
        self.ax.set_xlabel('Frequency (Hz)', fontsize=8, color=self.theme_colors['fg_primary'])
        self.ax.set_ylabel('Power (dB)', fontsize=8, color=self.theme_colors['fg_primary'])
        self.ax.grid(True, alpha=0.3, color='gray', linestyle='--')
        
        if self.data is None or self.sampling_rate is None:
            self.ax.text(0.5, 0.5, 'No Data', transform=self.ax.transAxes,
                        ha='center', va='center', color=self.theme_colors['fg_primary'])
            self.canvas.draw()
            return
            
        # Extract epoch data
        total_samples = len(self.data)
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        
        start_idx = self.epoch_idx * samples_per_epoch
        end_idx = min(start_idx + samples_per_epoch, total_samples)
        
        if start_idx >= total_samples:
            self.ax.text(0.5, 0.5, 'Epoch out of range', transform=self.ax.transAxes,
                        ha='center', va='center', color=self.theme_colors['fg_primary'])
            self.canvas.draw()
            return
            
        epoch_data = self.data[start_idx:end_idx]
        
        if len(epoch_data) == 0:
            return
            
        # Compute PSD using Welch's method
        try:
            freqs, psd = signal.welch(epoch_data, fs=self.sampling_rate, 
                                     nperseg=min(len(epoch_data), int(4 * self.sampling_rate)))
            
            # Plot PSD (in log scale)
            # Avoid log(0)
            psd_db = 10 * np.log10(psd + 1e-12)
            
            # Draw colored background bands
            for name, (low, high, color) in self.bands.items():
                self.ax.axvspan(low, high, alpha=0.2, color=color)
                
                # Add label at top
                mid = (low + high) / 2
                if mid < 60:  # Only label visible bands
                    self.ax.text(mid, 0.95, name, transform=self.ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=6, color=self.theme_colors['fg_primary'],
                                alpha=0.7)
            
            self.ax.plot(freqs, psd_db, color='#00ff00', linewidth=1.5)
            self.ax.set_xlim(0, 60) # Limit x-axis to 60Hz as typically relevant for EEG
            
            # Auto-scale Y with some margin
            db_min = np.min(psd_db[freqs <= 60])
            db_max = np.max(psd_db[freqs <= 60])
            margin = (db_max - db_min) * 0.1
            if margin == 0: margin = 5
            self.ax.set_ylim(db_min - margin, db_max + margin)
            
        except Exception as e:
            print(f"Error computing PSD: {e}")
            self.ax.text(0.5, 0.5, 'Computation Error', transform=self.ax.transAxes,
                        ha='center', va='center', color='red')
            
        self.figure.tight_layout()
        self.canvas.draw()
