"""
Spectrogram and Label Windows for MEEG Seizure Labeling.

This module contains the spectrogram display and seizure phase labeling widgets.
Labeling: Interictal, Preictal, Ictal, Postictal
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QCursor


class SpectrogramWidget(QWidget):
    """Widget for displaying EEG power spectrogram."""
    
    # Signal emitted when user selects an epoch by clicking
    epoch_selected = pyqtSignal(int)  # epoch_idx
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.data = None
        self.sampling_rate = None  # Will be set when data is loaded
        self.epoch_length = 1
        self.current_epoch = 0
        self.epochs_to_show = 15
        
        # Color limits for brightness control
        self.vmin = None  # Auto by default
        self.vmax = None  # Auto by default
        self.last_computed_vmin = None  # Track last auto-computed values
        self.last_computed_vmax = None
        
        # Pre-computed spectrogram matrix
        self.precomputed_spectrogram = None  # Shape: (n_freq_bins, n_epochs)
        self.freq_bins = None  # Frequency bin edges
        
        # Selection mode for clicking to select epoch
        self.selection_mode = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
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
        
    def precompute_spectrogram(self):
        """Pre-compute spectrogram for all epochs to speed up plotting."""
        if self.data is None or len(self.data) == 0 or self.sampling_rate is None:
            return
        
        print("Pre-computing spectrogram for all epochs...")
        
        # Calculate frequency bins with 2 Hz resolution
        max_freq = 30  # Hz
        self.freq_bins = np.arange(0, max_freq + 2, 2)  # 0-2, 2-4, 4-6, ..., 28-30 Hz
        n_freq_bins = len(self.freq_bins) - 1
        
        # Get total number of epochs
        n_epochs = self.get_n_epochs()
        
        # Initialize pre-computed matrix
        self.precomputed_spectrogram = np.zeros((n_freq_bins, n_epochs))
        
        # Compute power for each epoch
        from scipy.signal import welch
        
        for epoch_idx in range(n_epochs):
            # Get data for this epoch
            epoch_start_sample = int(epoch_idx * self.epoch_length * self.sampling_rate)
            epoch_end_sample = int((epoch_idx + 1) * self.epoch_length * self.sampling_rate)
            epoch_end_sample = min(epoch_end_sample, len(self.data))
            
            if epoch_start_sample >= len(self.data):
                break
            
            epoch_data = self.data[epoch_start_sample:epoch_end_sample]
            
            if len(epoch_data) == 0:
                continue
            
            # Compute PSD using Welch's method
            frequencies, psd = welch(epoch_data, fs=self.sampling_rate, 
                                    nperseg=min(len(epoch_data), int(self.sampling_rate)))
            
            # Bin the power into 2 Hz bins
            for bin_idx in range(n_freq_bins):
                freq_low = self.freq_bins[bin_idx]
                freq_high = self.freq_bins[bin_idx + 1]
                
                # Find frequencies in this bin
                freq_mask = (frequencies >= freq_low) & (frequencies < freq_high)
                
                if freq_mask.any():
                    # Average power in this bin
                    self.precomputed_spectrogram[bin_idx, epoch_idx] = np.mean(psd[freq_mask])
        
        # Convert power to dB scale
        self.precomputed_spectrogram = 10 * np.log10(self.precomputed_spectrogram + 1e-10)
        
        print(f"Spectrogram pre-computation complete: {n_epochs} epochs, {n_freq_bins} frequency bins")
    
    def update_plot(self):
        """Update the spectrogram display using pre-computed data."""
        self.ax.clear()
        self.ax.set_facecolor(self.theme_colors['bg_primary'])
        
        if self.precomputed_spectrogram is None or self.freq_bins is None:
            self.ax.text(0.5, 0.5, 'No data to display', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', color=self.theme_colors['fg_primary'], fontsize=10)
            self.canvas.draw()
            return
        
        # Calculate time range - show epochs around current epoch
        n_total_epochs = self.precomputed_spectrogram.shape[1]
        start_epoch = max(0, self.current_epoch - self.epochs_to_show // 2)
        end_epoch = min(n_total_epochs, start_epoch + self.epochs_to_show)
        
        # Extract the relevant portion of pre-computed spectrogram
        power_matrix_db = self.precomputed_spectrogram[:, start_epoch:end_epoch]
        
        # Plot as image with blue-green-yellow colormap
        extent = [start_epoch, end_epoch, self.freq_bins[0], self.freq_bins[-1]]
        
        # Apply color limits if set
        if self.vmin is None or self.vmax is None:
            im = self.ax.imshow(power_matrix_db, aspect='auto', origin='lower', 
                              extent=extent, cmap='viridis', interpolation='nearest')
            # Store auto-computed values
            self.last_computed_vmin = im.get_clim()[0]
            self.last_computed_vmax = im.get_clim()[1]
        else:
            im = self.ax.imshow(power_matrix_db, aspect='auto', origin='lower', 
                              extent=extent, cmap='viridis', interpolation='nearest',
                              vmin=self.vmin, vmax=self.vmax)
            # Update stored values
            self.last_computed_vmin = self.vmin
            self.last_computed_vmax = self.vmax
        
        # Highlight current epoch with vertical lines
        self.ax.axvline(self.current_epoch, color='white', linewidth=2, linestyle='-', alpha=0.8)
        self.ax.axvline(self.current_epoch + 1, color='white', linewidth=2, linestyle='-', alpha=0.8)
        
        # Set axis labels and ticks
        self.ax.set_xlabel('Epoch', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_ylabel('Frequency (Hz)', fontsize=9, color=self.theme_colors['fg_primary'])
        
        # Set y-axis ticks at 5 Hz intervals (0, 5, 10, 15, 20, 25, 30)
        self.ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
        
        self.ax.tick_params(colors=self.theme_colors['fg_primary'], labelsize=8)
        self.ax.spines['bottom'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['left'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['top'].set_color(self.theme_colors['fg_primary'])
        self.ax.spines['right'].set_color(self.theme_colors['fg_primary'])
        
        # Disable scientific notation on x-axis
        from matplotlib.ticker import ScalarFormatter, MaxNLocator
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        self.ax.xaxis.set_major_formatter(formatter)
        
        self.figure.tight_layout()
        
        # Add diamond marker above the plot after tight_layout
        # This ensures it's positioned correctly relative to the axis
        epoch_center = self.current_epoch + 0.5
        self.ax.plot(epoch_center, 32, marker='D', markersize=6, 
                    color='yellow', markeredgecolor='white', markeredgewidth=1,
                    clip_on=False, zorder=10)
        
        self.canvas.draw()
        
    def get_n_epochs(self):
        """Get total number of epochs."""
        if self.data is None or self.sampling_rate is None or self.sampling_rate == 0:
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
        # Pre-compute spectrogram for all epochs
        self.precompute_spectrogram()
        self.update_plot()
    
    def get_color_limits(self):
        """Get current color limits (vmin, vmax).
        
        Returns:
        --------
        tuple : (vmin, vmax)
            Current color scale limits
        """
        return self.last_computed_vmin, self.last_computed_vmax
    
    def set_color_limits(self, vmin, vmax):
        """Set color limits for brightness control.
        
        Parameters:
        -----------
        vmin : float or None
            Minimum value for color scale (None for auto)
        vmax : float or None
            Maximum value for color scale (None for auto)
        """
        self.vmin = vmin
        self.vmax = vmax
        self.update_plot()
    
    def on_click(self, event):
        """Handle mouse click on spectrogram."""
        if not self.selection_mode or event.xdata is None:
            return
        
        # Calculate which epoch was clicked
        if self.data is None or self.sampling_rate is None:
            return
        
        clicked_time = event.xdata
        clicked_epoch = int(clicked_time / self.epoch_length)
        
        # Calculate total number of epochs
        total_samples = len(self.data)
        samples_per_epoch = int(self.sampling_rate * self.epoch_length)
        n_epochs = max(1, total_samples // samples_per_epoch)
        
        # Validate epoch range
        if 0 <= clicked_epoch < n_epochs:
            # Emit signal to change current epoch
            self.epoch_selected.emit(clicked_epoch)
            # Disable selection mode after selection
            self.selection_mode = False
            # Restore cursor to arrow
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            print(f"Selected epoch {clicked_epoch} at time {clicked_time:.2f}s")
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        key = event.key()
        
        if key == Qt.Key.Key_A:
            # Toggle selection mode
            self.selection_mode = not self.selection_mode
            if self.selection_mode:
                # Change cursor to crosshair
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                print("Selection mode enabled. Click on spectrogram to select epoch.")
            else:
                # Restore default cursor
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                print("Selection mode disabled.")
        else:
            # Let parent handle other keys
            super().keyPressEvent(event)


class LabelWidget(QWidget):
    """Widget for displaying and editing seizure phase labels.
    
    Labels:
        0 = Interictal (key 1, green)
        1 = Preictal (key 2, orange)
        2 = Ictal (key 3, red)
        3 = Postictal (key 4, dark purple)
        4 = NaN/unusable (key N, gray)
    """
    
    # Signal emitted when a label is changed
    label_changed = pyqtSignal(int, int)  # epoch, phase
    # Signal emitted when spacebar is pressed (for video tab switch)
    spacebar_pressed = pyqtSignal(int)  # current epoch
    
    # Label constants for seizure phases (CSV values)
    INTERICTAL = 0  # Key 1
    PREICTAL = 1    # Key 2
    ICTAL = 2       # Key 3
    POSTICTAL = 3   # Key 4
    NAN_LABEL = 4   # Key N - unusable data
    
    # Phase names for display
    PHASE_NAMES = ['Interictal', 'Preictal', 'Ictal', 'Postictal', 'NaN']
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.n_epochs = 0
        self.labels = np.array([])  # 0-3 phase labels, 4 = NaN
        self.current_epoch = 0
        self.epochs_to_show = 5
        self.mosaic_epochs_to_show = 15  # For < > key navigation
        
        # Range selection state
        self.range_start_epoch = None  # Set by Q key
        self.range_end_epoch = None    # Set by W or E key
        self.range_selection_active = False
        
        # Color scheme for seizure phases
        self.colors = self.generate_colors()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("Seizure Phase Labeling (1-4: Interictal/Preictal/Ictal/Postictal)")
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
        """Generate color scheme for seizure phases.
        
        Colors:
            Interictal (0): Green
            Preictal (1): Orange
            Ictal (2): Red
            Postictal (3): Dark Purple
            NaN (4): Gray
        """
        colors = {}
        colors[self.INTERICTAL] = np.array([0.0, 0.8, 0.0, 1.0])      # Green
        colors[self.PREICTAL] = np.array([1.0, 0.5, 0.0, 1.0])        # Orange
        colors[self.ICTAL] = np.array([1.0, 0.0, 0.0, 1.0])           # Red
        colors[self.POSTICTAL] = np.array([0.4, 0.0, 0.6, 1.0])       # Dark Purple
        colors[self.NAN_LABEL] = np.array([0.5, 0.5, 0.5, 1.0])       # Gray for NaN
            
        return colors
        
    def on_click(self, event):
        """Handle mouse click to select epoch."""
        if event.inaxes == self.ax and event.xdata is not None:
            clicked_epoch = int(np.clip(event.xdata, 0, self.n_epochs - 1))
            self.current_epoch = clicked_epoch
            # Emit signal to sync all plots
            self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
            self.update_plot()
        # Set focus to this widget so keyboard events work
        self.setFocus()
            
    def keyPressEvent(self, event):
        """Handle keyboard input for labeling using Qt events.
        
        Key mappings:
            1 = Interictal (CSV value 0)
            2 = Preictal (CSV value 1)
            3 = Ictal (CSV value 2)
            4 = Postictal (CSV value 3)
            N = NaN/unusable
        """
        key = event.key()
        
        # Handle number keys 1-4 for seizure phase labeling
        if key == Qt.Key.Key_1:
            self._apply_label(self.INTERICTAL)  # Key 1 -> Interictal (CSV 0)
        elif key == Qt.Key.Key_2:
            self._apply_label(self.PREICTAL)    # Key 2 -> Preictal (CSV 1)
        elif key == Qt.Key.Key_3:
            self._apply_label(self.ICTAL)       # Key 3 -> Ictal (CSV 2)
        elif key == Qt.Key.Key_4:
            self._apply_label(self.POSTICTAL)   # Key 4 -> Postictal (CSV 3)
        
        # N key - mark as NaN (unusable data)
        elif key == Qt.Key.Key_N:
            self._apply_label(self.NAN_LABEL)
        
        # Q key - set range start
        elif key == Qt.Key.Key_Q:
            self.range_start_epoch = self.current_epoch
            self.range_end_epoch = None
            self.range_selection_active = True
            self.update_plot()
            print(f"Range start set to epoch {self.range_start_epoch}")
        
        # W key - set range end and show message
        elif key == Qt.Key.Key_W:
            if self.range_start_epoch is not None:
                self.range_end_epoch = self.current_epoch
                self._show_range_selection_message()
            else:
                print("Press Q first to set range start")
        
        # E key - select to last epoch
        elif key == Qt.Key.Key_E:
            if self.range_start_epoch is not None:
                self.range_end_epoch = self.n_epochs - 1
                self._show_range_selection_message()
            else:
                print("Press Q first to set range start")
        
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
        # Handle < and > keys for full plot shift
        elif key == Qt.Key.Key_Less or key == Qt.Key.Key_Comma:  # < key (Shift+,)
            # Shift backward by mosaic plot width (mosaic_epochs_to_show)
            new_epoch = max(0, self.current_epoch - self.mosaic_epochs_to_show)
            if new_epoch != self.current_epoch:
                self.current_epoch = new_epoch
                # Emit signal to sync all plots
                self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
                self.update_plot()
        elif key == Qt.Key.Key_Greater or key == Qt.Key.Key_Period:  # > key (Shift+.)
            # Shift forward by mosaic plot width (mosaic_epochs_to_show)
            new_epoch = min(self.n_epochs - 1, self.current_epoch + self.mosaic_epochs_to_show)
            if new_epoch != self.current_epoch:
                self.current_epoch = new_epoch
                # Emit signal to sync all plots
                self.label_changed.emit(self.current_epoch, int(self.labels[self.current_epoch]))
                self.update_plot()
        # Handle spacebar for video tab switch
        elif key == Qt.Key.Key_Space:
            self.spacebar_pressed.emit(self.current_epoch)
        
        # Escape key - cancel range selection
        elif key == Qt.Key.Key_Escape:
            self._cancel_range_selection()
        else:
            # Let parent handle other keys
            super().keyPressEvent(event)
    
    def _apply_label(self, phase):
        """Apply a phase label to current epoch or selected range."""
        if self.range_selection_active and self.range_start_epoch is not None and self.range_end_epoch is not None:
            # Apply to entire range
            start = min(self.range_start_epoch, self.range_end_epoch)
            end = max(self.range_start_epoch, self.range_end_epoch)
            
            for epoch_idx in range(start, end + 1):
                if 0 <= epoch_idx < self.n_epochs:
                    self.labels[epoch_idx] = phase
            
            label_name = self.PHASE_NAMES[phase] if phase < len(self.PHASE_NAMES) else str(phase)
            print(f"Applied label {label_name} to epochs {start}-{end} ({end - start + 1} epochs)")
            
            # Clear range selection
            self._cancel_range_selection()
            
            # Emit signal for last epoch in range
            self.label_changed.emit(end, phase)
            self.update_plot()
        elif 0 <= self.current_epoch < self.n_epochs:
            # Apply to single epoch
            self.labels[self.current_epoch] = phase
            self.label_changed.emit(self.current_epoch, phase)
            self.update_plot()
    
    def _show_range_selection_message(self):
        """Show message about selected range and prompt for label."""
        if self.range_start_epoch is None or self.range_end_epoch is None:
            return
        
        start = min(self.range_start_epoch, self.range_end_epoch)
        end = max(self.range_start_epoch, self.range_end_epoch)
        n_selected = end - start + 1
        
        self.update_plot()  # Update to show selection highlight
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Range Selected")
        msg.setText(f"You have selected {n_selected} epochs (epoch {start} to {end}).")
        msg.setInformativeText("Press 1-4 to label seizure phase:\n1=Interictal, 2=Preictal, 3=Ictal, 4=Postictal\nor press N for NaN. Press ESC to cancel.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # Keep focus on this widget for label input
        self.setFocus()
        print(f"Range selected: epochs {start}-{end} ({n_selected} epochs). Press 1-4 (Interictal/Preictal/Ictal/Postictal) or N for NaN.")
    
    def _cancel_range_selection(self):
        """Cancel the current range selection."""
        self.range_start_epoch = None
        self.range_end_epoch = None
        self.range_selection_active = False
        self.update_plot()
        print("Range selection cancelled")
            
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
        
        visible_epochs = end_epoch - start_epoch
        
        # Create label image (4 rows for seizure phases, visible_epochs columns)
        n_phases = 4  # Interictal, Preictal, Ictal, Postictal
        label_img = np.zeros((n_phases, visible_epochs, 4))
        
        # Vectorized approach for faster rendering
        visible_labels = self.labels[start_epoch:end_epoch]
        
        for i, phase in enumerate(visible_labels):
            phase = int(phase)
            for row in range(n_phases):
                if phase == self.NAN_LABEL:
                    # NaN label - show gray across all rows
                    label_img[row, i, :] = self.colors[self.NAN_LABEL]
                elif row == phase:
                    label_img[row, i, :] = self.colors[phase]
                else:
                    label_img[row, i, :] = [0.1, 0.1, 0.1, 1.0]  # Dark gray
        
        # Display the label image
        extent = [start_epoch, end_epoch, 0, n_phases]
        self.ax.imshow(label_img, aspect='auto', origin='lower', extent=extent,
                      interpolation='nearest')
        
        # Highlight range selection if active
        if self.range_selection_active and self.range_start_epoch is not None:
            range_start = self.range_start_epoch
            range_end = self.range_end_epoch if self.range_end_epoch is not None else self.current_epoch
            
            # Ensure start <= end
            r_start = min(range_start, range_end)
            r_end = max(range_start, range_end)
            
            # Only draw highlight for visible range
            vis_r_start = max(r_start, start_epoch)
            vis_r_end = min(r_end, end_epoch - 1)
            
            if vis_r_start <= vis_r_end:
                # Draw single rectangle for efficiency
                self.ax.axvspan(vis_r_start, vis_r_end + 1, alpha=0.3, color='cyan', zorder=5)
        
        # Highlight current epoch
        self.ax.axvline(self.current_epoch + 0.5, color='yellow', linewidth=3, linestyle='-')
        
        # Set axis properties
        self.ax.set_xlabel('Epoch', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_ylabel('Seizure Phase', fontsize=9, color=self.theme_colors['fg_primary'])
        self.ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        self.ax.set_yticklabels(['Interictal', 'Preictal', 'Ictal', 'Postictal'], 
                                 color=self.theme_colors['fg_primary'], fontsize=7)
        
        # Set x-axis ticks - limit number of ticks for performance
        # Use ScalarFormatter to prevent scientific notation
        from matplotlib.ticker import ScalarFormatter, MaxNLocator
        
        # Use MaxNLocator to limit number of ticks for performance
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=min(visible_epochs, 15), integer=True))
        
        # Disable scientific notation
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        self.ax.xaxis.set_major_formatter(formatter)
        
        self.ax.set_xlim(start_epoch, end_epoch)
        self.ax.set_ylim(0, n_phases)
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
    
    def set_mosaic_epochs_to_show(self, n_epochs):
        """Set number of mosaic epochs for keyboard navigation."""
        self.mosaic_epochs_to_show = max(1, n_epochs)
        
    def initialize_labels(self, n_epochs):
        """Initialize label array with Interictal (0) for all epochs."""
        self.n_epochs = n_epochs
        self.labels = np.zeros(n_epochs, dtype=int)  # All start as Interictal (0)
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

