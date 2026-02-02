"""
Full View Video Dialog for MEEG Seizure Labeling.

This module provides a dialog for previewing and exporting combined video
with EEG mosaic plot and topography visualization.
"""

import numpy as np
import pandas as pd
import cv2
import tempfile
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressDialog, QFileDialog,
                             QMessageBox, QSplitter, QGroupBox, QSpinBox,
                             QFormLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for rendering
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class FullViewVideoDialog(QDialog):
    """Dialog for combined video preview and export."""
    
    def __init__(self, video_path, df, mosaic_relationships, electrode_positions,
                 sampling_rate, epoch_length, time_offset, theme_colors=None, 
                 segment_start_time=0.0, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.df = df
        self.mosaic_relationships = mosaic_relationships
        self.electrode_positions = electrode_positions
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.time_offset = time_offset
        self.segment_start_time = segment_start_time  # Offset for segment-extracted videos
        self.theme_colors = theme_colors or {'bg_primary': '#1a1a1a', 'fg_primary': '#ffffff'}
        
        # Video capture
        self.cap = None
        self.video_fps = 30
        self.video_width = 640
        self.video_height = 480
        self.video_duration = 0
        
        # Export settings
        self.output_fps = 30
        self.topo_update_interval = 0.5  # seconds
        
        # Display parameters for mosaic
        self.spacing = 200
        self.time_window = 10.0  # seconds to show in mosaic
        
        self.setWindowTitle("Full View Video Export")
        self.setMinimumSize(1000, 700)
        self.init_ui()
        self.load_video_info()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Combined Video Export")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Preview area
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Load video to see preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet("background-color: #333; color: white;")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # Settings
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        settings_layout.addRow("Output FPS:", self.fps_spinbox)
        
        # Info labels
        self.info_label = QLabel("Video info will appear here")
        settings_layout.addRow("Video Info:", self.info_label)
        
        layout.addWidget(settings_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("🔄 Generate Preview Frame")
        self.preview_btn.clicked.connect(self.generate_preview)
        btn_layout.addWidget(self.preview_btn)
        
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("💾 Export Video")
        self.save_btn.clicked.connect(self.export_video)
        btn_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
    def load_video_info(self):
        """Load video information."""
        if not self.video_path or not Path(self.video_path).exists():
            self.info_label.setText("Video file not found")
            return
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.info_label.setText("Failed to open video")
            return
            
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = frame_count / self.video_fps if self.video_fps > 0 else 0
        
        self.info_label.setText(
            f"{self.video_width}x{self.video_height} @ {self.video_fps:.1f}fps, "
            f"Duration: {self.video_duration:.1f}s"
        )
        
        # Set output FPS to match source
        self.fps_spinbox.setValue(int(self.video_fps))
        self.cap.release()
        
    def render_mosaic_frame(self, eeg_time):
        """Render mosaic plot at given EEG time and return as numpy array."""
        fig = Figure(figsize=(6, 4), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        
        if self.df.empty or not self.mosaic_relationships:
            ax.text(0.5, 0.5, 'No EEG data', ha='center', va='center')
        else:
            # Calculate time range
            half_window = self.time_window / 2
            start_time = max(0, eeg_time - half_window)
            end_time = start_time + self.time_window
            
            start_sample = int(start_time * self.sampling_rate)
            end_sample = int(end_time * self.sampling_rate)
            end_sample = min(end_sample, len(self.df))
            
            if start_sample < end_sample:
                df_slice = self.df.iloc[start_sample:end_sample]
                time_array = np.arange(len(df_slice)) / self.sampling_rate + start_time
                
                y_ticks = []
                y_labels = []
                total_pairs = len(self.mosaic_relationships)
                current_y_offset = total_pairs * self.spacing
                
                for mosaic_name, ch1, ch2 in self.mosaic_relationships:
                    if ch1 not in df_slice.columns or ch2 not in df_slice.columns:
                        continue
                    montage_data = df_slice[ch1].values - df_slice[ch2].values
                    ax.plot(time_array, montage_data + current_y_offset, 
                           color='black', linewidth=0.5)
                    y_ticks.append(current_y_offset)
                    y_labels.append(f'{ch1}-{ch2}')
                    current_y_offset -= self.spacing
                
                # Red time marker
                ax.axvline(x=eeg_time, color='red', linewidth=2, alpha=0.9)
                
                if y_ticks:
                    ax.set_ylim(min(y_ticks) - 200, max(y_ticks) + 200)
                ax.set_xlim(start_time, end_time)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=6)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Render to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        
        # Convert RGBA to BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
    def render_topography_frame(self, eeg_time):
        """Render topography at given EEG time and return as numpy array."""
        fig = Figure(figsize=(3, 3), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        
        # Simple topography visualization
        if not self.electrode_positions or self.df.empty:
            ax.text(0.5, 0.5, 'No topo data', ha='center', va='center', 
                   transform=ax.transAxes)
        else:
            # Get sample index for current time
            sample_idx = int(eeg_time * self.sampling_rate)
            sample_idx = max(0, min(sample_idx, len(self.df) - 1))
            
            # Extract electrode values for this sample
            x_coords = []
            y_coords = []
            values = []
            
            for electrode in self.electrode_positions:
                name = electrode.get('name', f"E{electrode['number']}")
                if name in self.df.columns:
                    x_coords.append(electrode.get('ml', 0))
                    y_coords.append(electrode.get('ap', 0))
                    values.append(self.df[name].iloc[sample_idx])
            
            if values:
                # Normalize values for coloring
                values = np.array(values)
                vmin, vmax = np.percentile(values, [5, 95])
                if vmax - vmin < 1:
                    vmax = vmin + 1
                
                sc = ax.scatter(x_coords, y_coords, c=values, cmap='RdBu_r',
                              s=100, vmin=vmin, vmax=vmax, edgecolors='black')
                ax.set_xlim(-6, 6)
                ax.set_ylim(-9, 6)
                ax.set_aspect('equal')
                ax.set_title(f't={eeg_time:.1f}s', fontsize=8)
                ax.axis('off')
        
        fig.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
    def generate_preview(self):
        """Generate and show a preview frame."""
        if not self.video_path:
            return
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
            
        # Get frame from middle of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, video_frame = cap.read()
        cap.release()
        
        if not ret:
            return
            
        # Get video time at this position (accounting for segment offset)
        segment_time = (total_frames // 2) / self.video_fps
        original_video_time = segment_time + self.segment_start_time
        eeg_time = original_video_time - self.time_offset
        
        # Render combined frame
        combined = self.render_combined_frame(video_frame, eeg_time)
        
        # Convert to QImage and display
        height, width, channel = combined.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit preview area
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.preview_label.size(), 
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        
    def render_combined_frame(self, video_frame, eeg_time):
        """Combine video, mosaic, and topography into single frame."""
        # Render mosaic
        mosaic_img = self.render_mosaic_frame(eeg_time)
        
        # Render topography
        topo_img = self.render_topography_frame(eeg_time)
        
        # Resize video frame to match layout
        video_resized = cv2.resize(video_frame, (640, 480))
        
        # Resize mosaic to match video height
        mosaic_resized = cv2.resize(mosaic_img, (600, 400))
        
        # Resize topo
        topo_resized = cv2.resize(topo_img, (300, 300))
        
        # Create combined frame: Video on left, Mosaic+Topo stacked on right
        # Total size: 640 + 600 = 1240 wide, 480 tall
        combined = np.zeros((480, 1240, 3), dtype=np.uint8)
        combined[:, :, :] = 255  # White background
        
        # Place video (left)
        combined[0:480, 0:640] = video_resized
        
        # Place mosaic (top right)
        combined[0:400, 640:1240] = mosaic_resized
        
        # Place topo (bottom right, centered)
        topo_x = 640 + (600 - 300) // 2
        combined[400:480, topo_x:topo_x+300] = topo_resized[0:80, :]  # Crop to fit
        
        # Convert BGR to RGB for display
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        return combined
        
    def export_video(self):
        """Export combined video to file."""
        if not self.video_path:
            QMessageBox.warning(self, "Error", "No video loaded")
            return
            
        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Combined Video", "", "MP4 Video (*.mp4);;AVI Video (*.avi)"
        )
        
        if not save_path:
            return
            
        # Open source video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Failed to open source video")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Use source video FPS to maintain original speed
        output_fps = self.video_fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, output_fps, (1240, 480))
        
        # Progress dialog
        progress = QProgressDialog("Exporting video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        frame_idx = 0
        while True:
            ret, video_frame = cap.read()
            if not ret:
                break
                
            # Calculate times (accounting for segment offset)
            segment_time = frame_idx / self.video_fps
            original_video_time = segment_time + self.segment_start_time
            eeg_time = original_video_time - self.time_offset
            
            # Render and write combined frame
            combined = self.render_combined_frame(video_frame, eeg_time)
            # Convert back to BGR for writing
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out.write(combined_bgr)
            
            frame_idx += 1
            progress.setValue(frame_idx)
            
            if progress.wasCanceled():
                break
                
        cap.release()
        out.release()
        progress.close()
        
        if not progress.wasCanceled():
            QMessageBox.information(self, "Success", f"Video exported to:\n{save_path}")
            
    def closeEvent(self, event):
        """Clean up on close."""
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
