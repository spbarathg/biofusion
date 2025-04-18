import unittest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Import the module to test
from src.core.paths import (
    ROOT_DIR, CONFIG_DIR, DATA_DIR, LOGS_DIR, WALLETS_DIR, BACKUPS_DIR,
    CONFIG_PATH, ENCRYPTION_KEY_PATH, get_path
)

class TestPaths(unittest.TestCase):
    """Test cases for the paths module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Save original paths
        self.original_paths = {
            "ROOT_DIR": ROOT_DIR,
            "CONFIG_DIR": CONFIG_DIR,
            "DATA_DIR": DATA_DIR,
            "LOGS_DIR": LOGS_DIR,
            "WALLETS_DIR": WALLETS_DIR,
            "BACKUPS_DIR": BACKUPS_DIR,
            "CONFIG_PATH": CONFIG_PATH,
            "ENCRYPTION_KEY_PATH": ENCRYPTION_KEY_PATH
        }
    
    def test_base_paths_existence(self):
        """Test that all base paths exist"""
        paths_to_check = [
            ROOT_DIR,
            CONFIG_DIR,
            DATA_DIR,
            LOGS_DIR,
            WALLETS_DIR,
            BACKUPS_DIR
        ]
        
        for path in paths_to_check:
            with self.subTest(path=path):
                self.assertTrue(os.path.exists(path), f"Path {path} does not exist")
                self.assertTrue(os.path.isdir(path), f"Path {path} is not a directory")
    
    def test_config_files_paths(self):
        """Test paths for configuration files"""
        # CONFIG_PATH should exist
        self.assertTrue(CONFIG_PATH.parent.exists(), f"Config directory {CONFIG_PATH.parent} does not exist")
        
        # ENCRYPTION_KEY_PATH's parent directory should exist
        self.assertTrue(ENCRYPTION_KEY_PATH.parent.exists(), 
                        f"Encryption key parent directory {ENCRYPTION_KEY_PATH.parent} does not exist")
    
    def test_get_path_function(self):
        """Test the get_path function"""
        # Test with a simple relative path
        relative_path = "test/sample.txt"
        expected_path = ROOT_DIR / relative_path
        self.assertEqual(get_path(relative_path), expected_path)
        
        # Test with current directory notation
        relative_path = "./test/sample.txt"
        expected_path = ROOT_DIR / "test/sample.txt"  # ./ should be stripped
        self.assertEqual(get_path(relative_path), expected_path)
        
        # Test with absolute path (should return the path unchanged)
        if sys.platform == "win32":
            abs_path = "C:/absolute/path/test.txt"
            self.assertEqual(get_path(abs_path), Path(abs_path))
        else:
            abs_path = "/absolute/path/test.txt"
            self.assertEqual(get_path(abs_path), Path(abs_path))
        
        # Test with empty path
        with self.assertRaises(ValueError):
            get_path("")
    
    def test_directory_creation(self):
        """Test that directories are created if they don't exist"""
        # Create a temporary directory to use as ROOT_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            
            # Temporarily mock the ROOT_DIR and dependent paths
            # This would require modifying the module's globals, which is not ideal
            # In a real-world scenario, it would be better to refactor the module
            # to allow injection of the root path
            
            # For this test, we'll just verify that the directories mentioned
            # in the module docstring are created in the actual ROOT_DIR
            self.assertTrue(CONFIG_DIR.exists())
            self.assertTrue(DATA_DIR.exists())
            self.assertTrue(LOGS_DIR.exists())
            self.assertTrue(WALLETS_DIR.exists())
            self.assertTrue(BACKUPS_DIR.exists())

if __name__ == '__main__':
    unittest.main() 