import os
import shutil
import uuid
from typing import List, Generator
from contextlib import contextmanager
from utils.config import PathsConfig

class FileManager:
    """Manages file storage, folder initialization, and temp directory lifecycles."""
    
    def __init__(self, config: PathsConfig):
        self.config = config
        self._initialize_directories()
        
    def _initialize_directories(self) -> None:
        """Ensures all standard project directories exist."""
        directories = [
            self.config.data_dir,
            self.config.json_dir,
            self.config.csv_dir,
            self.config.predictions_dir,
            self.config.heatmaps_dir,
            self.config.overlays_dir,
            self.config.reports_dir,
            self.config.pdf_dir,
            self.config.temp_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def get_unique_filename(self, original_filename: str) -> str:
        """Generates a secure, unique filename to avoid collision."""
        ext = os.path.splitext(original_filename)[1]
        return f"{uuid.uuid4().hex}{ext}"
        
    def save_temp_file(self, content: bytes, filename: str) -> str:
        """Saves bytes content to temp directory, returning absolute path."""
        unique_name = self.get_unique_filename(filename)
        path = os.path.abspath(os.path.join(self.config.temp_dir, unique_name))
        with open(path, "wb") as f:
            f.write(content)
        return path
        
    def copy_to_predictions(self, src_path: str, filename: str) -> str:
        """Copies file to predictions directory."""
        dest_path = os.path.abspath(os.path.join(self.config.predictions_dir, filename))
        shutil.copy2(src_path, dest_path)
        return dest_path

    def copy_to_heatmaps(self, src_path: str, filename: str) -> str:
        """Copies file to heatmaps directory."""
        dest_path = os.path.abspath(os.path.join(self.config.heatmaps_dir, filename))
        shutil.copy2(src_path, dest_path)
        return dest_path

    def copy_to_overlays(self, src_path: str, filename: str) -> str:
        """Copies file to overlays directory."""
        dest_path = os.path.abspath(os.path.join(self.config.overlays_dir, filename))
        shutil.copy2(src_path, dest_path)
        return dest_path

    def clean_temp_files(self) -> None:
        """Cleans all files in the temp directory."""
        for filename in os.listdir(self.config.temp_dir):
            file_path = os.path.join(self.config.temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass # Fail silently during cleanups to maintain service uptime
                
    @contextmanager
    def temp_session(self) -> Generator['FileManager', None, None]:
        """Context manager session that cleans up temp files on exit."""
        try:
            yield self
        finally:
            self.clean_temp_files()
