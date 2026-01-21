"""
Auto-Update Module for MEEG Application.

This module handles checking for updates from GitHub releases and
provides update notifications and auto-update functionality.

Usage:
    from utils.auto_updater import AutoUpdater
    
    updater = AutoUpdater()
    updater.check_for_updates(callback=on_update_available)
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Callable, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox, QPushButton

# Import app version from config
try:
    from config import APP_VERSION, GITHUB_REPO
except ImportError:
    APP_VERSION = "1.0.0"
    GITHUB_REPO = "ChooYeonseo/MEEG"


class UpdateChecker(QThread):
    """Background thread for checking GitHub releases."""
    
    update_available = pyqtSignal(str, str, str)  # new_version, release_notes, download_url
    no_update = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, current_version: str, repo: str):
        super().__init__()
        self.current_version = current_version
        self.repo = repo
        self.api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    
    def run(self):
        """Check GitHub API for latest release."""
        print(f"[Update Check] Checking for updates... (current: v{self.current_version})")
        
        if not HAS_REQUESTS:
            print("[Update Check] ERROR: requests library not installed")
            self.error.emit("requests library not installed. Cannot check for updates.")
            return
        
        try:
            print(f"[Update Check] Fetching: {self.api_url}")
            response = requests.get(self.api_url, timeout=10)
            
            if response.status_code == 404:
                # No releases yet
                print("[Update Check] No releases found on GitHub")
                self.no_update.emit()
                return
            
            response.raise_for_status()
            release_data = response.json()
            
            latest_version = release_data.get("tag_name", "").lstrip("v")
            release_notes = release_data.get("body", "No release notes available.")
            
            print(f"[Update Check] Latest version on GitHub: v{latest_version}")
            
            # Find the appropriate download asset (Windows exe)
            download_url = None
            assets = release_data.get("assets", [])
            for asset in assets:
                name = asset.get("name", "").lower()
                if name.endswith(".exe") or name.endswith(".zip"):
                    download_url = asset.get("browser_download_url")
                    break
            
            # If no exe/zip asset, use source code zipball
            if not download_url:
                download_url = release_data.get("zipball_url")
            
            # Compare versions
            if self._is_newer_version(latest_version, self.current_version):
                print(f"[Update Check] UPDATE AVAILABLE: v{self.current_version} -> v{latest_version}")
                self.update_available.emit(latest_version, release_notes, download_url or "")
            else:
                print(f"[Update Check] You are running the latest version (v{self.current_version})")
                self.no_update.emit()
                
        except requests.RequestException as e:
            print(f"[Update Check] Network error: {e}")
            self.error.emit(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            print("[Update Check] Invalid response from GitHub API")
            self.error.emit("Invalid response from GitHub API")
        except Exception as e:
            print(f"[Update Check] Unexpected error: {e}")
            self.error.emit(f"Unexpected error: {str(e)}")
    
    def _is_newer_version(self, latest: str, current: str) -> bool:
        """Compare version strings (e.g., '1.2.0' > '1.1.0')."""
        try:
            latest_parts = [int(x) for x in latest.split(".")]
            current_parts = [int(x) for x in current.split(".")]
            
            # Pad with zeros
            max_len = max(len(latest_parts), len(current_parts))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            current_parts.extend([0] * (max_len - len(current_parts)))
            
            return latest_parts > current_parts
        except ValueError:
            # If parsing fails, do string comparison
            return latest > current


class AutoUpdater(QObject):
    """
    Main auto-updater class for MEEG application.
    
    Features:
    - Background update checking
    - Update notification popup
    - Download and apply updates
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_version = APP_VERSION
        self.repo = GITHUB_REPO
        self._checker: Optional[UpdateChecker] = None
        self._update_callback: Optional[Callable] = None
    
    def check_for_updates(self, 
                          callback: Optional[Callable[[str, str, str], None]] = None,
                          silent: bool = True) -> None:
        """
        Start background check for updates.
        
        Args:
            callback: Optional callback(version, notes, url) when update found
            silent: If True, don't show "no update" messages
        """
        if self._checker is not None and self._checker.isRunning():
            return  # Already checking
        
        self._update_callback = callback
        self._silent = silent
        
        self._checker = UpdateChecker(self.current_version, self.repo)
        self._checker.update_available.connect(self._on_update_available)
        self._checker.no_update.connect(self._on_no_update)
        self._checker.error.connect(self._on_error)
        self._checker.start()
    
    def _on_update_available(self, version: str, notes: str, url: str):
        """Handle update available signal."""
        if self._update_callback:
            self._update_callback(version, notes, url)
        else:
            self._show_update_dialog(version, notes, url)
    
    def _on_no_update(self):
        """Handle no update available."""
        print(f"[Update Check] No update needed - running latest version")
        if not self._silent:
            QMessageBox.information(
                None, 
                "No Updates", 
                f"You are running the latest version ({self.current_version})."
            )
    
    def _on_error(self, error_msg: str):
        """Handle update check error."""
        print(f"[Update Check] Error: {error_msg}")
        if not self._silent:
            QMessageBox.warning(
                None,
                "Update Check Failed",
                f"Could not check for updates:\n{error_msg}"
            )
    
    def _show_update_dialog(self, version: str, notes: str, url: str):
        """Show update available dialog."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Update Available")
        msg_box.setText(f"A new version of MEEG is available!\n\n"
                       f"Current version: {self.current_version}\n"
                       f"New version: {version}")
        msg_box.setDetailedText(notes)
        
        update_btn = msg_box.addButton("Update Now", QMessageBox.ButtonRole.AcceptRole)
        later_btn = msg_box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == update_btn:
            self._apply_update(version)
    
    def _apply_update(self, version: str):
        """Apply update using git pull and restart the application."""
        try:
            # Get the project root directory
            project_root = Path(__file__).parent.parent
            
            print(f"[Update] Applying update to v{version}...")
            print(f"[Update] Project root: {project_root}")
            
            # Check if this is a git repository
            git_dir = project_root / ".git"
            if not git_dir.exists():
                QMessageBox.warning(
                    None,
                    "Update Failed",
                    "This is not a git repository. Cannot apply automatic updates.\n"
                    "Please manually download the latest version from GitHub."
                )
                return
            
            # Run git pull
            print("[Update] Running git pull...")
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                # Try with master branch if main fails
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            if result.returncode == 0:
                print(f"[Update] Git pull successful: {result.stdout}")
                
                # Ask user to restart
                reply = QMessageBox.information(
                    None,
                    "Update Complete",
                    f"Successfully updated to v{version}!\n\n"
                    "The application needs to restart to apply changes.\n"
                    "Click OK to restart now.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Ok:
                    # Restart the application
                    print("[Update] Restarting application...")
                    python = sys.executable
                    main_script = project_root / "meeg.py"
                    subprocess.Popen([python, str(main_script)])
                    sys.exit(0)
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                print(f"[Update] Git pull failed: {error_msg}")
                QMessageBox.warning(
                    None,
                    "Update Failed",
                    f"Could not apply update:\n{error_msg}\n\n"
                    "Please try running 'git pull' manually."
                )
                
        except subprocess.TimeoutExpired:
            print("[Update] Git pull timed out")
            QMessageBox.warning(
                None,
                "Update Failed",
                "Update timed out. Please try again or run 'git pull' manually."
            )
        except Exception as e:
            print(f"[Update] Error: {e}")
            QMessageBox.critical(
                None,
                "Update Failed",
                f"Could not apply update:\n{str(e)}"
            )


def check_for_updates_on_startup(parent=None, silent: bool = True):
    """
    Convenience function to check for updates on application startup.
    
    Call this from your main window's __init__ or show event.
    
    Args:
        parent: Parent widget (optional)
        silent: If True, only show notifications when update is available
    
    Returns:
        AutoUpdater instance (keep reference to prevent garbage collection)
    """
    updater = AutoUpdater(parent)
    updater.check_for_updates(silent=silent)
    return updater
