"""
Output capture utility for redirecting stdout/stderr to GUI.

This module provides the OutputCapture class that captures print statements
and redirects them to the GUI terminal window using PyQt6 signals.
"""

import sys
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal


class OutputCapture(QObject):
    """Capture stdout/stderr and emit signals for GUI updates."""
    
    output_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        """Capture text and emit signal."""
        if text.strip():  # Only process non-empty text
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] {text}"
            self.output_received.emit(formatted_text)
            # Also write to original stdout
            self.original_stdout.write(text)
            
    def flush(self):
        """Flush method required for stdout replacement."""
        pass
