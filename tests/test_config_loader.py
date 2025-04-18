import os
import unittest
import tempfile
import yaml
from pathlib import Path

# Import the module to test
from src.core.config_loader import ConfigLoader, get_config

class TestConfigLoader(unittest.TestCase):
    """Test cases for ConfigLoader class"""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a temporary config file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_settings.yaml"
        
        # Sample test configuration
        self.test_config = {
            "colony": {
                "initial_capital": 10.0,
                "max_workers": 5
            },
            "worker": {
                "max_trades_per_hour": 10,
                "min_profit_threshold": 0.01
            },
            "network": {
                "rpc_url": "https://api.mainnet-beta.solana.com",
                "timeout": 30
            }
        }
        
        # Write test config to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
            
        # Reset the singleton instance
        ConfigLoader._instance = None
        ConfigLoader._config = None

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.temp_dir.cleanup()
        
        # Reset the singleton instance
        ConfigLoader._instance = None
        ConfigLoader._config = None

    def test_singleton_pattern(self):
        """Test that ConfigLoader implements the singleton pattern correctly."""
        loader1 = ConfigLoader(str(self.config_path))
        loader2 = ConfigLoader(str(self.config_path))
        
        # Both instances should be the same object
        self.assertIs(loader1, loader2)

    def test_get_full_config(self):
        """Test getting the complete configuration."""
        loader = ConfigLoader(str(self.config_path))
        config = loader.get_full_config()
        
        self.assertEqual(config, self.test_config)

    def test_get_section_config(self):
        """Test getting specific configuration sections."""
        loader = ConfigLoader(str(self.config_path))
        
        # Test colony section
        colony_config = loader.get_colony_config()
        self.assertEqual(colony_config, self.test_config["colony"])
        
        # Test worker section
        worker_config = loader.get_worker_config()
        self.assertEqual(worker_config, self.test_config["worker"])

    def test_get_with_dot_notation(self):
        """Test getting config values using dot notation."""
        loader = ConfigLoader(str(self.config_path))
        
        # Test nested values
        self.assertEqual(loader.get("colony.initial_capital"), 10.0)
        self.assertEqual(loader.get("worker.max_trades_per_hour"), 10)
        self.assertEqual(loader.get("network.rpc_url"), "https://api.mainnet-beta.solana.com")
        
        # Test with default value for missing key
        self.assertEqual(loader.get("missing.key", "default"), "default")

    def test_get_config_helper(self):
        """Test the get_config helper function."""
        # Ensure ConfigLoader instance is created with our test config
        ConfigLoader(str(self.config_path))
        
        # Test the helper function
        self.assertEqual(get_config("colony.initial_capital"), 10.0)
        self.assertEqual(get_config("missing.key", "default"), "default")

    def test_reload_config(self):
        """Test that config can be reloaded when file changes."""
        loader = ConfigLoader(str(self.config_path))
        
        # Initial check
        self.assertEqual(loader.get("colony.initial_capital"), 10.0)
        
        # Update the config file
        updated_config = self.test_config.copy()
        updated_config["colony"]["initial_capital"] = 20.0
        
        with open(self.config_path, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Reload the config
        loader.reload()
        
        # Check updated value
        self.assertEqual(loader.get("colony.initial_capital"), 20.0)

if __name__ == '__main__':
    unittest.main() 