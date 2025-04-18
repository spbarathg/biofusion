import unittest
from unittest import mock
import os
import tempfile
import shutil
import logging
from pathlib import Path
import sys
import io

# Import the module to test
from src.logging.log_config import setup_logging
from loguru import logger

class TestLogConfig(unittest.TestCase):
    """Test cases for the log_config module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary logs directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self.temp_dir.name) / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Store original handlers to restore later
        self.original_handlers = logger._core.handlers.copy()
        
        # Create an environment where we can capture log output
        self.log_output = io.StringIO()
        
    def tearDown(self):
        """Tear down test fixtures"""
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Restore logger's original state
        logger.remove()
        for handler_id, handler in self.original_handlers.items():
            logger._core.handlers[handler_id] = handler
    
    def test_setup_logging_basic(self):
        """Test basic setup of logging"""
        # Patch stdout to capture output
        with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Call the setup_logging function
            component_logger = setup_logging("test_component")
            
            # Verify logger was properly configured
            self.assertIsNotNone(component_logger)
            
            # Verify we can log messages
            test_message = "Test log message"
            component_logger.info(test_message)
            
            # Check log output
            log_contents = mock_stdout.getvalue()
            self.assertIn(test_message, log_contents)
            self.assertIn("test_component", log_contents)
    
    def test_setup_logging_with_file(self):
        """Test logging setup with component log file"""
        # Create path for mock logs directory
        with mock.patch('src.logging.log_config.logger.add') as mock_add:
            # Call setup_logging with a log file
            component_logger = setup_logging("file_component", "file_component.log")
            
            # Check that logger.add was called with the correct file path
            file_log_call = False
            for call in mock_add.call_args_list:
                args, kwargs = call
                if kwargs.get('level') == "DEBUG" and "file_component.log" in str(args[0]):
                    file_log_call = True
                    break
            
            self.assertTrue(file_log_call, "Logger was not set up with the correct file path")
    
    def test_log_levels(self):
        """Test that different log levels work properly"""
        # Patch stdout to capture output
        with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Set up logging
            component_logger = setup_logging("level_test")
            
            # Log different levels
            component_logger.debug("Debug message")
            component_logger.info("Info message")
            component_logger.warning("Warning message")
            component_logger.error("Error message")
            component_logger.critical("Critical message")
            
            # Check log output
            log_contents = mock_stdout.getvalue()
            self.assertIn("Debug message", log_contents)
            self.assertIn("Info message", log_contents)
            self.assertIn("Warning message", log_contents)
            self.assertIn("Error message", log_contents)
            self.assertIn("Critical message", log_contents)
    
    def test_error_file_separation(self):
        """Test that errors are logged to a separate file"""
        # Mock logger.add to check how it's called
        with mock.patch('src.logging.log_config.logger.add') as mock_add:
            # Set up logging
            component_logger = setup_logging("error_test")
            
            # Check that error_log is created with ERROR level
            error_log_call = False
            for call in mock_add.call_args_list:
                args, kwargs = call
                if kwargs.get('level') == "ERROR" and "error.log" in str(args[0]):
                    error_log_call = True
                    break
            
            self.assertTrue(error_log_call, "Error log file not set up with ERROR level")

if __name__ == '__main__':
    unittest.main() 