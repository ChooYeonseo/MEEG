"""
Spectrogram and Label Windows for MEEG Seizure Labeling.

This module contains the spectrogram display and Racine score labeling widgets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class SpectrogramWidget(QWidget):
    """Widget for displaying EEG power spectrogram."""
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.data = None
        self.sampling_rate = 1000.0
        self.epoch_length = 10
        self.current_epoch = 0
        self.epochs_to_show = 5
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("Power Spectrogram (2Hz bins)")
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 2), facecolor=self.theme_colors['bg_primary'],
                           edgecolor=self.theme_colors['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        layout.addWidget(self.canvas)
        
        # Initialize the plot
        self.update_plot()
        
    def update_plot(self):
        """Update the spectrogram display."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        
        if self.data is None or len(self.data) == 0:
            self.ax.text(0.5, 0.5, 'No data to display', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', color=self.theme_colors['fg_primary'], fontsize=10)
            self.canvas.draw()
            return
        
        # Calculate time range
        start_epoch = max(0, self.current_epoch - self.epochs_to_show // 2)
        end_epoch = min(self.get_n_epochs(), start_epoch + self.epochs_to_show)
        
        start_sample = int(start_epoch * self.epoch_length * self.sampling_rate)
        end_sample = int(end_epoch * self.epoch_length * self.sampling_rate)
        end_sample = min(end_sample, len(self.data))
        
        if start_sample >= end_sample:
            self.canvas.draw()
            return
        
        data_slice = self.data[start_sample:end_sample]
        
        # Compute spectrogram with 2Hz bins
        nperseg = int(self.sampling_rate)  # 1 second window for 1Hz resolution
        # Use noverlap to get 2Hz resolution
        noverlap = int(nperseg * 0.5)
        
        f, t, Sxx = signal.spectrogram(data_slice, self.sampling_rate,
                                       nperseg=nperseg, noverlap=noverlap)
        
        # Limit frequency range to 0-30 Hz
        freq_mask = f <= 30
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Convert power to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Plot spectrogram
        extent = [start_epoch * self.epoch_length, 
                 start_epoch * self.epoch_length + len(data_slice) / self.sampling_rate,
                 f[0], f[-1]]
        
        im = self.ax.imshow(Sxx_db, aspect='auto', origin='lower', 
                          extent=extent, cmap='jet', interpolation='bilinear')
        
        # Highlight current epoch
        epoch_start_time = self.current_epoch * self.epoch_length
        epoch_end_time = (self.current_epoch + 1) * self.epoch_length
        self.ax.axvline(epoch_start_time, color='white', linewidth=2, linestyle='--')
        self.ax.axvline(epoch_end_time, color='white', linewidth=2, linestyle='--')
        
        self.ax.set_xlabel('Time (s)', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_ylabel('Frequency (Hz)', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.tick_params(colors=self.theme_colors['fg_primary'], labelsize=8)
        self.ax.spines['bottom'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['left'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['top'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['right'].set_color(self.theme_colors['fg_primary'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def get_n_epochs(self):
        """Get total number of epochs."""
        if self.data is None or self.sampling_rate == 0:
            return 0
        total_samples = len(self.data)
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        return max(1, total_samples // samples_per_epoch)
        
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
        
    def set_data(self, data, sampling_rate):
        """Update the data to display."""
        self.data = data
        self.sampling_rate = sampling_rate
        self.update_plot()


class LabelWidget(QWidget):
    """Widget for displaying and editing Racine seizure scores (0-8)."""
    
    # Signal emitted when a label is changed
    label_changed = pyqtSignal(int, int)  # epoch, score
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.n_epochs = 0
        self.labels = np.array([])  # 0-8 Racine scores
        self.current_epoch = 0
        self.epochs_to_show = 5
        
        # Color scheme: green for 0, red gradient for 1-8
        self.colors = self.generate_colors()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("Racine Score Labeling (0-8)")
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 2), facecolor=self.theme_colors['bg_primary'],
                           edgecolor=self.theme_colors['bg_primary'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        layout.addWidget(self.canvas)
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Initialize the plot
        self.update_plot()
        
    def generate_colors(self):
        """Generate color scheme: green for 0, red gradient for 1-8."""
        colors = {}
        colors[0] = np.array([0.0, 0.8, 0.0, 1.0])  # Green
        
        # Red gradient from light to dark
        for i in range(1, 9):
            # Darker red as score increases
            intensity = 1.0 - (i - 1) / 8.0 * 0.6  # From 1.0 to 0.4
            colors[i] = np.array([1.0, intensity * 0.2, intensity * 0.2, 1.0])
            
        return colors
        
    def on_click(self, event):
        """Handle mouse click to select epoch."""
        if event.inaxes == self.ax and event.xdata is not None:
            clicked_epoch = int(np.clip(event.xdata, 0, self.n_epochs - 1))
            self.current_epoch = clicked_epoch
            self.update_plot()
        # Set focus to this widget so keyboard events work
        self.setFocus()
            
    def keyPressEvent(self, event):
        """Handle keyboard input for labeling using Qt events."""
        key = event.key()
        
        # Handle number keys 0-8 for labeling
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            score = key - Qt.Key.Key_0
            if score <= 8 and 0 <= self.current_epoch < self.n_epochs:
                self.labels[self.current_epoch] = score
                self.label_changed.emit(self.current_epoch, score)
                # Move to next epoch
                if self.current_epoch < self.n_epochs - 1:
                    self.current_epoch += 1
                    # Emit signal to sync all plots to new epoch
                    self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
                self.update_plot()
        # Handle arrow keys for navigation
        elif key == Qt.Key.Key_Left and self.current_epoch > 0:
            self.current_epoch -= 1
            # Emit signal to sync all plots
            self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
            self.update_plot()
        elif key == Qt.Key.Key_Right and self.current_epoch < self.n_epochs - 1:
            self.current_epoch += 1
            # Emit signal to sync all plots
            self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
            self.update_plot()
        else:
            # Let parent handle other keys
            super().keyPressEvent(event)
            
    def update_plot(self):
        """Update the label display."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        
        if self.n_epochs == 0:
            self.ax.text(0.5, 0.5, 'No epochs to label', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', color=self.theme_colors['fg_primary'], fontsize=10)
            self.canvas.draw()
            return
        
        # Calculate visible epoch range
        start_epoch = max(0, self.current_epoch - self.epochs_to_show // 2)
        end_epoch = min(self.n_epochs, start_epoch + self.epochs_to_show)
        start_epoch = max(0, end_epoch - self.epochs_to_show)
        
        # Create label image (9 rows for scores 0-8, n_epochs columns)
        label_img = np.zeros((9, end_epoch - start_epoch, 4))
        
        for i, epoch_idx in enumerate(range(start_epoch, end_epoch)):
            score = int(self.labels[epoch_idx])
            for row in range(9):
                if row == score:
                    label_img[row, i, :] = self.colors[score]
                else:
                    label_img[row, i, :] = [0.1, 0.1, 0.1, 1.0]  # Dark gray
        
        # Display the label image
        extent = [start_epoch, end_epoch, 0, 9]
        self.ax.imshow(label_img, aspect='auto', origin='lower', extent=extent,
                      interpolation='nearest')
        
        # Highlight current epoch
        self.ax.axvline(self.current_epoch + 0.5, color='yellow', linewidth=3, linestyle='-')
        
        # Set axis properties
        self.ax.set_xlabel('Epoch', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_ylabel('Racine Score', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        self.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8'], color=self.theme_colors['fg_primary'])
        self.ax.set_xlim(start_epoch, end_epoch)
        self.ax.set_ylim(0, 9)
        self.ax.tick_params(colors=self.theme_colors['fg_primary'], labelsize=8)
        self.ax.spines['bottom'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['left'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['top'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['right'].set_color(self.theme_colors['fg_primary'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def set_epoch(self, epoch_idx):
        """Set the current epoch."""
        self.current_epoch = np.clip(epoch_idx, 0, self.n_epochs - 1)
        self.update_plot()
        
    def set_epochs_to_show(self, n_epochs):
        """Set number of epochs to display."""
        self.epochs_to_show = max(1, n_epochs)
        self.update_plot()
        
    def initialize_labels(self, n_epochs):
        """Initialize label array with zeros."""
        self.n_epochs = n_epochs
        self.labels = np.zeros(n_epochs, dtype=int)
        self.current_epoch = 0
        self.update_plot()
    
    def set_labels(self, labels):
        """Set labels from array (for loading from file)."""
        self.labels = np.array(labels, dtype=int)
        self.n_epochs = len(self.labels)
        self.current_epoch = 0
        self.update_plot()
        
    def get_labels(self):
        """Get the current labels array."""
        return self.labels.copy()
        
    def set_labels(self, labels):
        """Set the labels array."""
        if len(labels) == self.n_epochs:
            self.labels = labels.copy()
            self.update_plot()
