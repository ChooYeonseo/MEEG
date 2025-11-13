"""
Cache project selection dialog.

This module provides a dialog for selecting and managing cached EEG projects.
"""

from pathlib import Path
from datetime import datetime
import sys
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                            QListWidgetItem, QPushButton, QLabel, QMessageBox,
                            QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

current_dir = Path(__file__).parent.parent
utils_dir = current_dir / "utils"
sys.path.insert(0, str(utils_dir))

# Import theme system
sys.path.insert(0, str(current_dir))
from theme import preferences_manager

try:
    from cache_manager import cache_manager
except ImportError as e:
    print(f"Error importing cache_manager: {e}")


class CachedProjectDialog(QDialog):
    """Dialog for selecting cached projects."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_cache_key = None
        
        # Get current theme
        self.current_theme = preferences_manager.get_setting('theme', 'tokyo_night')
        
        self.setup_ui()
        self.load_projects()
        self.apply_theme()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Load Existing Project")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Select a Cached Project")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Instructions
        instructions = QLabel("Choose from previously processed EEG datasets:")
        instructions.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instructions)
        
        # Project list group
        list_group = QGroupBox("Available Projects")
        list_layout = QVBoxLayout(list_group)
        
        self.project_list = QListWidget()
        self.project_list.setMinimumHeight(250)
        self.project_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.project_list.itemDoubleClicked.connect(self.accept)
        list_layout.addWidget(self.project_list)
        
        layout.addWidget(list_group)
        
        # Project details group
        details_group = QGroupBox("Project Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(120)
        self.details_text.setPlainText("Select a project to view details...")
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Management buttons
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self.load_projects)
        button_layout.addWidget(self.refresh_button)
        
        self.delete_button = QPushButton("🗑️ Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected_project)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)
        
        self.clear_all_button = QPushButton("Clear All Cache")
        self.clear_all_button.clicked.connect(self.clear_all_cache)
        button_layout.addWidget(self.clear_all_button)
        
        button_layout.addStretch()
        
        # Dialog buttons
        self.load_button = QPushButton("📂 Load Project")
        self.load_button.setEnabled(False)
        self.load_button.setDefault(True)
        self.load_button.clicked.connect(self.accept)
        button_layout.addWidget(self.load_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def apply_theme(self):
        """Apply the current theme to the dialog."""
        # Import appropriate theme
        if self.current_theme == 'tokyo_night':
            from theme import TOKYO_NIGHT_STYLES as THEME_STYLES, TOKYO_NIGHT_COLORS as THEME_COLORS
        elif self.current_theme == 'dark':
            from theme import DARK_THEME_STYLES as THEME_STYLES, DARK_COLORS as THEME_COLORS
        else:  # normal
            from theme import NORMAL_THEME_STYLES as THEME_STYLES, NORMAL_COLORS as THEME_COLORS
        
        # Get hover color (fallback to bg_tertiary if hover not defined)
        hover_color = THEME_COLORS.get('hover', THEME_COLORS.get('bg_tertiary', '#e8e8e8'))
        selection_color = THEME_COLORS.get('accent_blue', THEME_COLORS.get('accent_primary', '#6b6b6b'))
        
        # Apply dialog background
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {THEME_COLORS['bg_primary']};
                color: {THEME_COLORS['fg_primary']};
            }}
            QLabel {{
                color: {THEME_COLORS['fg_primary']};
                background: transparent;
            }}
            {THEME_STYLES.get('form_layout', '')}
        """)
        
        # Apply group box styles
        for widget in self.findChildren(QGroupBox):
            widget.setStyleSheet(THEME_STYLES['group_box'])
        
        # Apply button styles
        for widget in self.findChildren(QPushButton):
            widget.setStyleSheet(THEME_STYLES['button_primary'])
        
        # Apply list widget styles
        if hasattr(self, 'project_list'):
            list_style = THEME_STYLES.get('list_widget', '')
            if not list_style:
                # Create custom list style from theme colors
                list_style = f"""
                    QListWidget {{
                        background-color: {THEME_COLORS['bg_secondary']};
                        color: {THEME_COLORS['fg_primary']};
                        border: 1px solid {THEME_COLORS['border']};
                        border-radius: 4px;
                        padding: 5px;
                    }}
                    QListWidget::item {{
                        padding: 8px;
                        border-radius: 4px;
                    }}
                    QListWidget::item:selected {{
                        background-color: {selection_color};
                        color: white;
                    }}
                    QListWidget::item:hover {{
                        background-color: {hover_color};
                    }}
                """
            self.project_list.setStyleSheet(list_style)
        
        # Apply text edit styles
        if hasattr(self, 'details_text'):
            text_style = THEME_STYLES.get('text_edit', '')
            if not text_style:
                # Create custom text edit style from theme colors
                text_style = f"""
                    QTextEdit {{
                        background-color: {THEME_COLORS['bg_secondary']};
                        color: {THEME_COLORS['fg_primary']};
                        border: 1px solid {THEME_COLORS['border']};
                        border-radius: 4px;
                        padding: 8px;
                    }}
                """
            self.details_text.setStyleSheet(text_style)
        
    def load_projects(self):
        """Load and display cached projects."""
        self.project_list.clear()
        self.details_text.setPlainText("Select a project to view details...")
        
        try:
            projects = cache_manager.list_cached_projects()
            
            if not projects:
                item = QListWidgetItem("No cached projects found")
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it unselectable
                item.setData(Qt.ItemDataRole.UserRole, None)
                self.project_list.addItem(item)
                return
            
            for project in projects:
                # Format the display text
                timestamp = datetime.fromisoformat(project["cache_timestamp"])
                time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                directory_name = Path(project["directory_path"]).name
                display_text = f"{directory_name} ({project['num_files']} files) - {time_str}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, project)
                self.project_list.addItem(item)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading cached projects: {str(e)}")
            
    def on_selection_changed(self):
        """Handle project selection changes."""
        current_item = self.project_list.currentItem()
        
        if current_item is None:
            self.load_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.details_text.setPlainText("Select a project to view details...")
            self.selected_cache_key = None
            return
            
        project_data = current_item.data(Qt.ItemDataRole.UserRole)
        
        if project_data is None:
            self.load_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.details_text.setPlainText("No project data available.")
            self.selected_cache_key = None
            return
        
        # Enable buttons
        self.load_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        self.selected_cache_key = project_data["cache_key"]
        
        # Display project details
        timestamp = datetime.fromisoformat(project_data["cache_timestamp"])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        details = f"""Project Details:
        
Directory: {project_data["directory_path"]}
Files: {project_data["num_files"]} files
Cached: {time_str}
Cache Key: {project_data["cache_key"][:16]}...

Double-click to load this project."""
        
        self.details_text.setPlainText(details)
        
    def delete_selected_project(self):
        """Delete the selected cached project."""
        if not self.selected_cache_key:
            return
            
        current_item = self.project_list.currentItem()
        if not current_item:
            return
            
        project_data = current_item.data(Qt.ItemDataRole.UserRole)
        if not project_data:
            return
            
        # Confirm deletion
        directory_name = Path(project_data["directory_path"]).name
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the cached project '{directory_name}'?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cache_manager.remove_cached_project(self.selected_cache_key)
                self.load_projects()  # Refresh the list
                QMessageBox.information(self, "Success", "Cached project deleted successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error deleting cached project: {str(e)}")
                
    def clear_all_cache(self):
        """Clear all cached projects."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to clear ALL cached projects?\n\n"
            "This action cannot be undone and will remove all cached data.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                cache_manager.clear_cache()
                self.load_projects()  # Refresh the list
                QMessageBox.information(self, "Success", "All cached projects cleared successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error clearing cache: {str(e)}")
                
    def get_selected_cache_key(self):
        """Get the selected cache key."""
        return self.selected_cache_key
