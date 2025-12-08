"""
Video Sync Widget for MEEG Seizure Labeling.

This module contains the video synchronization widget that allows users to
upload a video, sync it with EEG data, and view video frames at specific epochs.
Includes mosaic EEG display with time marker and combined video export.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, QSlider,
                             QDoubleSpinBox, QFileDialog, QSizePolicy, QSplitter,
                             QProgressDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .topowindow import TopographyWidget


class MosaicTimelineWidget(QWidget):
    """Widget for displaying mosaic EEG with a moving time marker."""
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        self.df = pd.DataFrame()
        self.mosaic_relationships = []
        self.sampling_rate = 1000.0
        self.epoch_length = 1.0
        self.current_time = 0.0  # Current time marker position in seconds
        self.time_window = 10.0  # Show ±5 seconds around current time
        
        # Display parameters
        self.spacing = 300
        self.spacing_cluster = 600
        self.limit = 400
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 4), facecolor='white', edgecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # Time marker line (will be updated)
        self.time_marker_line = None
        
    def set_data(self, df, mosaic_relationships, sampling_rate, epoch_length):
        """Set the EEG data."""
        self.df = df if df is not None else pd.DataFrame()
        self.mosaic_relationships = mosaic_relationships or []
        self.sampling_rate = sampling_rate or 1000.0
        self.epoch_length = epoch_length
        self.update_plot()
        
    def set_current_time(self, time_seconds):
        """Set current time and update the time marker."""
        self.current_time = time_seconds
        self.update_plot()
        
    def update_plot(self):
        """Update the mosaic plot with time marker."""
        self.ax.clear()
        self.ax.set_facecolor('white')
        
        if self.df.empty or not self.mosaic_relationships:
            self.ax.text(0.5, 0.5, 'No data', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', color='black', fontsize=10)
            self.canvas.draw()
            return
        
        # Calculate time range to display (centered on current time)
        half_window = self.time_window / 2
        start_time = max(0, self.current_time - half_window)
        end_time = start_time + self.time_window
        
        # Convert to samples
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int(end_time * self.sampling_rate)
        end_sample = min(end_sample, len(self.df))
        
        if start_sample >= end_sample:
            self.canvas.draw()
            return
        
        # Get data slice
        df_slice = self.df.iloc[start_sample:end_sample]
        time_array = np.arange(len(df_slice)) / self.sampling_rate + start_time
        
        # Plot each montage
        y_ticks = []
        y_labels = []
        total_pairs = len(self.mosaic_relationships)
        current_y_offset = total_pairs * self.spacing
        
        for mosaic_name, ch1, ch2 in self.mosaic_relationships:
            if ch1 not in df_slice.columns or ch2 not in df_slice.columns:
                continue
            
            data_ch1 = df_slice[ch1].values
            data_ch2 = df_slice[ch2].values
            montage_data = data_ch1 - data_ch2
            
            self.ax.plot(time_array, montage_data + current_y_offset, 
                        color='black', linewidth=0.6)
            
            y_ticks.append(current_y_offset)
            y_labels.append(f'{ch1}-{ch2}')
            current_y_offset -= self.spacing
        
        # Draw red time marker line
        if y_ticks:
            y_min = min(y_ticks) - self.limit
            y_max = max(y_ticks) + self.limit
            self.ax.axvline(x=self.current_time, color='red', linewidth=2, 
                           linestyle='-', alpha=0.9, zorder=10)
            self.ax.set_ylim(y_min, y_max)
        
        # Set axis properties
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(y_labels, fontsize=7, color='black')
        self.ax.set_xlabel('Time (s)', fontsize=9, color='black')
        self.ax.set_xlim(start_time, end_time)
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='black', labelsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()


class VideoSyncWidget(QWidget):
    """Widget for video playback synchronized with EEG epochs."""
    
    # Signal emitted when user wants to go back to label tab
    return_to_label = pyqtSignal()
    
    def __init__(self, theme_colors=None, parent=None):
        super().__init__(parent)
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        
        # Video parameters
        self.video_path = None
        self.time_offset = 0.0  # Video time = EEG time + offset (in seconds)
        self.epoch_length = 1.0  # seconds
        self.sampling_rate = 1000.0  # Hz
        self.current_epoch = 0
        
        # EEG data for mosaic display
        self.df = pd.DataFrame()
        self.mosaic_relationships = []
        self.electrode_positions = []
        self.last_displayed_epoch = -1  # Track epoch for topography updates
        
        # Timer for syncing mosaic with video
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self.sync_mosaic_with_video)
        
        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Main splitter: Video (left) | Mosaic (right)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === LEFT SIDE: Video Player ===
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("Video Synchronization")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(title_label)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_widget)
        
        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Connect signals
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderMoved.connect(self.seek_position)
        controls_layout.addWidget(self.seek_slider, stretch=1)
        
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        video_layout.addLayout(controls_layout)
        
        # Sync controls
        sync_group = QGroupBox("Synchronization")
        sync_layout = QFormLayout(sync_group)
        
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton("📁 Load Video")
        self.load_btn.clicked.connect(self.load_video_dialog)
        load_layout.addWidget(self.load_btn)
        
        self.video_path_label = QLabel("No video loaded")
        self.video_path_label.setStyleSheet("color: gray; font-style: italic;")
        load_layout.addWidget(self.video_path_label, stretch=1)
        sync_layout.addRow("Video:", load_layout)
        
        self.offset_spinbox = QDoubleSpinBox()
        self.offset_spinbox.setRange(-3600.0, 3600.0)
        self.offset_spinbox.setValue(0.0)
        self.offset_spinbox.setSuffix(" sec")
        self.offset_spinbox.setDecimals(2)
        self.offset_spinbox.setSingleStep(0.1)
        self.offset_spinbox.valueChanged.connect(self.on_offset_changed)
        sync_layout.addRow("Time Offset:", self.offset_spinbox)
        
        video_layout.addWidget(sync_group)
        
        # Navigation and export buttons
        nav_layout = QHBoxLayout()
        
        self.epoch_label = QLabel("Epoch: 0 | EEG: 0.00s | Video: 0.00s")
        nav_layout.addWidget(self.epoch_label)
        
        nav_layout.addStretch()
        
        self.fullview_btn = QPushButton("🎬 Full View Export")
        self.fullview_btn.clicked.connect(self.on_fullview_clicked)
        nav_layout.addWidget(self.fullview_btn)
        
        self.return_btn = QPushButton("← Return (ESC)")
        self.return_btn.clicked.connect(self.on_return_clicked)
        nav_layout.addWidget(self.return_btn)
        
        video_layout.addLayout(nav_layout)
        
        main_splitter.addWidget(video_container)
        
        # === RIGHT SIDE: Topography + Mosaic Timeline ===
        mosaic_container = QWidget()
        mosaic_layout = QVBoxLayout(mosaic_container)
        mosaic_layout.setContentsMargins(0, 0, 0, 0)
        
        # Topography widget (replaces title)
        self.topo_widget = TopographyWidget(self.electrode_positions, self.theme_colors)
        self.topo_widget.setMinimumHeight(200)
        self.topo_widget.setMaximumHeight(300)
        mosaic_layout.addWidget(self.topo_widget, stretch=1)
        
        self.mosaic_timeline = MosaicTimelineWidget(self.theme_colors)
        mosaic_layout.addWidget(self.mosaic_timeline, stretch=2)
        
        main_splitter.addWidget(mosaic_container)
        
        # Set splitter proportions (60% video, 40% mosaic)
        main_splitter.setSizes([600, 400])
        
        layout.addWidget(main_splitter)
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events - ESC to return to label."""
        if event.key() == Qt.Key.Key_Escape:
            self.return_to_label.emit()
        else:
            super().keyPressEvent(event)
            
    def set_eeg_data(self, df, mosaic_relationships):
        """Set EEG data for mosaic display."""
        self.df = df if df is not None else pd.DataFrame()
        self.mosaic_relationships = mosaic_relationships or []
        self.mosaic_timeline.set_data(self.df, self.mosaic_relationships, 
                                       self.sampling_rate, self.epoch_length)
        
    def set_electrode_positions(self, positions):
        """Set electrode positions for topography."""
        self.electrode_positions = positions or []
        # Update topo widget with electrode positions and re-prepare coordinates
        if hasattr(self, 'topo_widget'):
            self.topo_widget.electrode_positions = self.electrode_positions
            self.topo_widget._prepare_electrode_coords()
        
    def load_video_dialog(self):
        """Open file dialog to load a video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)
            
    def load_video(self, path):
        """Load a video file."""
        self.video_path = path
        self.video_path_label.setText(Path(path).name)
        self.video_path_label.setStyleSheet("color: green;")
        self.media_player.setSource(QUrl.fromLocalFile(path))
        print(f"Video loaded: {path}")
        
    def toggle_playback(self):
        """Toggle play/pause."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸ Pause")
            
    def stop_playback(self):
        """Stop playback."""
        self.media_player.stop()
        self.play_btn.setText("▶ Play")
        
    def seek_position(self, position):
        """Seek to a position in the video."""
        self.media_player.setPosition(position)
        
    def on_position_changed(self, position):
        """Handle position change."""
        self.seek_slider.setValue(position)
        self.update_time_labels(position)
        
    def on_duration_changed(self, duration):
        """Handle duration change."""
        self.seek_slider.setRange(0, duration)
        
    def on_playback_state_changed(self, state):
        """Handle playback state changes - start/stop sync timer."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.sync_timer.start(100)  # Update mosaic every 100ms
        else:
            self.sync_timer.stop()
            
    def sync_mosaic_with_video(self):
        """Sync mosaic time marker with current video position."""
        video_time_ms = self.media_player.position()
        video_time = video_time_ms / 1000.0
        
        # Calculate EEG time from video time
        eeg_time = video_time - self.time_offset
        
        # Update mosaic timeline
        self.mosaic_timeline.set_current_time(eeg_time)
        
        # Check if epoch changed and update topography
        current_epoch = int(eeg_time / self.epoch_length) if self.epoch_length > 0 else 0
        if current_epoch != self.last_displayed_epoch:
            self.update_topography_for_epoch(current_epoch, eeg_time)
            self.last_displayed_epoch = current_epoch
        
    def update_time_labels(self, position_ms):
        """Update time display labels."""
        duration = self.media_player.duration()
        
        pos_str = self.format_time(position_ms)
        dur_str = self.format_time(duration)
        self.time_label.setText(f"{pos_str} / {dur_str}")
        
        # Calculate EEG time
        video_time = position_ms / 1000.0
        eeg_time = video_time - self.time_offset
        epoch = int(eeg_time / self.epoch_length) if self.epoch_length > 0 else 0
        
        self.epoch_label.setText(f"Epoch: {epoch} | EEG: {eeg_time:.2f}s | Video: {video_time:.2f}s")
        
    def format_time(self, ms):
        """Format milliseconds to MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
        
    def on_offset_changed(self, value):
        """Handle time offset change."""
        self.time_offset = value
        self.seek_to_epoch(self.current_epoch)
        
    def set_time_offset(self, seconds):
        """Set the EEG-to-video time offset."""
        self.time_offset = seconds
        self.offset_spinbox.setValue(seconds)
        
    def set_epoch_length(self, seconds):
        """Set the epoch length in seconds."""
        self.epoch_length = seconds
        self.mosaic_timeline.epoch_length = seconds
        
    def set_sampling_rate(self, rate):
        """Set the sampling rate."""
        self.sampling_rate = rate
        self.mosaic_timeline.sampling_rate = rate
        
    def seek_to_epoch(self, epoch_idx):
        """Seek video to the time corresponding to the given epoch."""
        self.current_epoch = epoch_idx
        
        eeg_time = epoch_idx * self.epoch_length
        video_time = eeg_time + self.time_offset
        
        # Update mosaic timeline
        self.mosaic_timeline.set_current_time(eeg_time)
        
        # Update topography for this epoch
        self.update_topography_for_epoch(epoch_idx, eeg_time)
        self.last_displayed_epoch = epoch_idx
        
        # Seek video
        if video_time >= 0:
            video_time_ms = int(video_time * 1000)
            self.media_player.setPosition(video_time_ms)
        else:
            self.media_player.setPosition(0)
            
        print(f"Seeking to epoch {epoch_idx}: EEG={eeg_time:.2f}s, Video={video_time:.2f}s")
        
    def update_topography_for_epoch(self, epoch_idx, eeg_time):
        """Update topography widget for the given epoch."""
        if self.df.empty or not self.electrode_positions:
            return
            
        # Calculate the sample range for this epoch
        start_sample = int(eeg_time * self.sampling_rate)
        end_sample = int((eeg_time + self.epoch_length) * self.sampling_rate)
        
        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(end_sample, len(self.df))
        
        if start_sample >= end_sample:
            return
            
        # Get epoch data and update topography
        epoch_data = self.df.iloc[start_sample:end_sample].copy()
        self.topo_widget.set_epoch(epoch_idx)
        self.topo_widget.set_data(epoch_data, self.sampling_rate, self.epoch_length)
        
    def on_return_clicked(self):
        """Handle return to label button click."""
        self.return_to_label.emit()
        
    def on_fullview_clicked(self):
        """Open full view export dialog."""
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
            
        from .fullview_video_dialog import FullViewVideoDialog
        dialog = FullViewVideoDialog(
            video_path=self.video_path,
            df=self.df,
            mosaic_relationships=self.mosaic_relationships,
            electrode_positions=self.electrode_positions,
            sampling_rate=self.sampling_rate,
            epoch_length=self.epoch_length,
            time_offset=self.time_offset,
            theme_colors=self.theme_colors,
            parent=self
        )
        dialog.exec()
