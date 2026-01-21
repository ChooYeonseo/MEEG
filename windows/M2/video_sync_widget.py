"""
Video Sync Widget for MEEG Seizure Labeling.

This module contains the video synchronization widget that allows users to
upload a video, sync it with EEG data, and view video frames at specific epochs.
Includes mosaic EEG display with time marker and combined video export.

Uses OpenCV for reliable video playback across all systems.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, QSlider,
                             QDoubleSpinBox, QFileDialog, QSizePolicy, QSplitter,
                             QProgressDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent, QImage, QPixmap

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
            
            self.ax.plot(time_array, -montage_data + current_y_offset, 
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
    """Widget for video playback synchronized with EEG epochs using OpenCV."""
    
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
        
        # OpenCV video capture
        self.cap = None
        self.video_fps = 30.0
        self.video_duration = 0.0
        self.video_frame_count = 0
        self.current_frame_idx = 0
        self.is_playing = False
        
        # Window length for video display (±N seconds around current time)
        self.video_window_length = 60.0  # Default ±60 seconds
        
        # EEG data for mosaic display
        self.df = pd.DataFrame()
        self.mosaic_relationships = []
        self.electrode_positions = []
        self.last_displayed_epoch = -1  # Track epoch for topography updates
        
        # Timer for video playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_frame)
        
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
        title_label = QLabel("Video Synchronization (OpenCV)")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(title_label)
        
        # Video display label (for OpenCV frames)
        self.video_label = QLabel("Load video to start")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: #222; color: white; font-size: 14px;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label)
        
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
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
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
        
        # Window length selector
        window_layout = QHBoxLayout()
        self.window_length_combo = QComboBox()
        self.window_length_combo.addItems(["±30 sec", "±45 sec", "±60 sec", "±90 sec", "±120 sec", "Full Video"])
        self.window_length_combo.setCurrentIndex(2)  # Default ±60 sec
        self.window_length_combo.currentTextChanged.connect(self._on_window_length_changed)
        window_layout.addWidget(self.window_length_combo)
        sync_layout.addRow("Display Window:", window_layout)
        
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
        self.topo_widget.setMinimumHeight(300)
        self.topo_widget.setMaximumHeight(450)
        mosaic_layout.addWidget(self.topo_widget, stretch=3)
        
        self.mosaic_timeline = MosaicTimelineWidget(self.theme_colors)
        mosaic_layout.addWidget(self.mosaic_timeline, stretch=1)
        
        main_splitter.addWidget(mosaic_container)
        
        # Set splitter proportions (60% video, 40% mosaic)
        main_splitter.setSizes([600, 400])
        
        layout.addWidget(main_splitter)
        
        # Track slider dragging state
        self._slider_dragging = False
        
    def _on_window_length_changed(self, text):
        """Handle window length selection change."""
        if "Full" in text:
            self.video_window_length = float('inf')
        else:
            # Parse "±60 sec" -> 60.0
            try:
                value = int(text.replace("±", "").replace(" sec", ""))
                self.video_window_length = float(value)
            except ValueError:
                self.video_window_length = 60.0
        print(f"Video window length set to: ±{self.video_window_length}s")
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events - ESC to return to label."""
        if event.key() == Qt.Key.Key_Escape:
            self.return_to_label.emit()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key.Key_Left:
            self._seek_relative(-1.0)
        elif event.key() == Qt.Key.Key_Right:
            self._seek_relative(1.0)
        else:
            super().keyPressEvent(event)
            
    def _seek_relative(self, delta_seconds):
        """Seek video by a relative amount."""
        if self.cap is None:
            return
        current_time = self.current_frame_idx / self.video_fps
        new_time = max(0, min(current_time + delta_seconds, self.video_duration))
        new_frame = int(new_time * self.video_fps)
        self.current_frame_idx = new_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self._display_current_frame()
        self._update_time_display()
        self._sync_mosaic_and_topo()
            
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
        """Load a video file using OpenCV."""
        # Release previous capture if any
        if self.cap is not None:
            self.cap.release()
            
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Error", f"Failed to open video:\n{path}")
            self.video_path = None
            self.cap = None
            return
        
        # Get video properties
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.video_frame_count / self.video_fps
        
        # Update UI
        self.video_path_label.setText(Path(path).name)
        self.video_path_label.setStyleSheet("color: green;")
        
        # Set slider range (in milliseconds for precision)
        self.seek_slider.setRange(0, int(self.video_duration * 1000))
        
        # Display first frame
        self.current_frame_idx = 0
        self._display_current_frame()
        self._update_time_display()
        
        print(f"Video loaded: {path}")
        print(f"  FPS: {self.video_fps:.2f}, Duration: {self.video_duration:.2f}s, Frames: {self.video_frame_count}")
        
    def _display_current_frame(self):
        """Read and display the current frame."""
        if self.cap is None:
            return
            
        # Seek to current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get label size for scaling
        label_size = self.video_label.size()
        target_w = label_size.width() - 10
        target_h = label_size.height() - 10
        
        if target_w > 0 and target_h > 0:
            # Calculate aspect-preserving size
            h, w = frame_rgb.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w > 0 and new_h > 0:
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to QPixmap
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.video_label.setPixmap(pixmap)
        
    def _update_frame(self):
        """Timer callback: advance and display next frame."""
        if self.cap is None or not self.is_playing:
            return
            
        self.current_frame_idx += 1
        
        # Check bounds
        if self.current_frame_idx >= self.video_frame_count:
            self.stop_playback()
            return
            
        self._display_current_frame()
        self._update_time_display()
        
        # Sync mosaic and topography periodically (every 5 frames to reduce overhead)
        if self.current_frame_idx % 5 == 0:
            self._sync_mosaic_and_topo()
            
        # Update slider if not being dragged
        if not self._slider_dragging:
            current_time_ms = int((self.current_frame_idx / self.video_fps) * 1000)
            self.seek_slider.setValue(current_time_ms)
            
    def _sync_mosaic_and_topo(self):
        """Sync mosaic timeline and topography with current video time."""
        if self.cap is None:
            return
            
        video_time = self.current_frame_idx / self.video_fps
        eeg_time = video_time - self.time_offset
        
        # Update mosaic timeline
        self.mosaic_timeline.set_current_time(eeg_time)
        
        # Check if epoch changed and update topography
        current_epoch = int(eeg_time / self.epoch_length) if self.epoch_length > 0 else 0
        if current_epoch != self.last_displayed_epoch and current_epoch >= 0:
            self.update_topography_for_epoch(current_epoch, eeg_time)
            self.last_displayed_epoch = current_epoch
            
    def _update_time_display(self):
        """Update the time label and epoch label."""
        if self.cap is None:
            return
            
        current_time = self.current_frame_idx / self.video_fps
        
        pos_str = self._format_time(current_time)
        dur_str = self._format_time(self.video_duration)
        self.time_label.setText(f"{pos_str} / {dur_str}")
        
        # Calculate EEG time
        eeg_time = current_time - self.time_offset
        epoch = int(eeg_time / self.epoch_length) if self.epoch_length > 0 else 0
        
        self.epoch_label.setText(f"Epoch: {epoch} | EEG: {eeg_time:.2f}s | Video: {current_time:.2f}s")
        
    def _format_time(self, seconds):
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
        
    def toggle_playback(self):
        """Toggle play/pause."""
        if self.cap is None:
            return
            
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
            
    def start_playback(self):
        """Start video playback."""
        if self.cap is None:
            return
            
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        
        # Calculate timer interval from FPS
        interval_ms = int(1000 / self.video_fps)
        self.playback_timer.start(max(1, interval_ms))
        
    def pause_playback(self):
        """Pause video playback."""
        self.is_playing = False
        self.play_btn.setText("▶ Play")
        self.playback_timer.stop()
        
    def stop_playback(self):
        """Stop playback and reset to start."""
        self.pause_playback()
        self.current_frame_idx = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._display_current_frame()
        self._update_time_display()
        self.seek_slider.setValue(0)
        
    def seek_position(self, position_ms):
        """Seek to a position (in milliseconds)."""
        if self.cap is None:
            return
            
        target_time = position_ms / 1000.0
        target_frame = int(target_time * self.video_fps)
        target_frame = max(0, min(target_frame, self.video_frame_count - 1))
        
        self.current_frame_idx = target_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self._display_current_frame()
        self._update_time_display()
        self._sync_mosaic_and_topo()
        
    def _on_slider_pressed(self):
        """Handle slider press - pause updates while dragging."""
        self._slider_dragging = True
        
    def _on_slider_released(self):
        """Handle slider release - resume updates."""
        self._slider_dragging = False
        self.seek_position(self.seek_slider.value())
        
    def on_offset_changed(self, value):
        """Handle time offset change."""
        self.time_offset = value
        self._sync_mosaic_and_topo()
        self._update_time_display()
        
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
        if self.cap is not None and video_time >= 0:
            target_frame = int(video_time * self.video_fps)
            target_frame = max(0, min(target_frame, self.video_frame_count - 1))
            self.current_frame_idx = target_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self._display_current_frame()
            self._update_time_display()
            
            # Update slider
            self.seek_slider.setValue(int(video_time * 1000))
            
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
        
    def closeEvent(self, event):
        """Clean up on close."""
        self.playback_timer.stop()
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
