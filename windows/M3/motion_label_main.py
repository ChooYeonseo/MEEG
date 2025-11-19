"""
Motion Label Main Window for MEEG Analysis.

This module contains the main window for movement detection and labeling.
"""

from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class MotionLabelWindow(QWidget):
    """Main window for motion/movement labeling."""
    
    def __init__(self, electrode_positions=None, current_data=None, 
                 channel_mapping=None, mosaic_relationships=None, parent=None):
        super().__init__(parent)
        self.electrode_positions = electrode_positions or []
        self.current_data = current_data
        self.channel_mapping = channel_mapping or {}
        self.mosaic_relationships = mosaic_relationships or []
        
        # Extract EEG data from cache
        self.eeg_data = self.extract_eeg_data()
        
        self.init_ui()
    
    def extract_eeg_data(self):
        """Extract EEG data from current_data loaded from cache."""
        extracted_data = {
            'amplifier_data': [],
            'aux_input_data': [],
            'frequency_parameters': None,
            'amplifier_channels': [],
            'filenames': []
        }
        
        if not self.current_data:
            return extracted_data
        
        try:
            for filename, result, data_present in self.current_data:
                if data_present and result:
                    extracted_data['filenames'].append(filename)
                    if 'amplifier_data' in result:
                        extracted_data['amplifier_data'].append(result['amplifier_data'])
                    if 'aux_input_data' in result:
                        extracted_data['aux_input_data'].append(result['aux_input_data'])
                    if 'frequency_parameters' in result and not extracted_data['frequency_parameters']:
                        extracted_data['frequency_parameters'] = result['frequency_parameters']
                    if 'amplifier_channels' in result and not extracted_data['amplifier_channels']:
                        extracted_data['amplifier_channels'] = result['amplifier_channels']
            
            if extracted_data['amplifier_data']:
                extracted_data['amplifier_data'] = np.concatenate(extracted_data['amplifier_data'], axis=1) if len(extracted_data['amplifier_data']) > 1 else extracted_data['amplifier_data'][0]
            if extracted_data['aux_input_data']:
                extracted_data['aux_input_data'] = np.concatenate(extracted_data['aux_input_data'], axis=1) if len(extracted_data['aux_input_data']) > 1 else extracted_data['aux_input_data'][0]
        except Exception as e:
            print(f"Error extracting EEG data: {e}")
        
        return extracted_data
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Movement Label - MEEG Analysis")
        self.setGeometry(200, 200, 1200, 800)
        
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Movement Detection & Labeling")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Info text
        info_label = QLabel("Movement detection and labeling interface will be implemented here.")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Data summary
        summary_label = QLabel(
            f"Electrodes: {len(self.electrode_positions)}\n"
            f"Mapped channels: {len(self.channel_mapping)}\n"
            f"Mosaic relationships: {len(self.mosaic_relationships)}"
        )
        summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(summary_label)
        
        layout.addStretch()
