import unittest
import logging
import os
from unittest.mock import patch
from log_config import configure_logging, get_log_level


class TestLogConfig(unittest.TestCase):
    """Test LOG_LEVEL environment variable support."""
    
    def setUp(self):
        """Reset logging configuration before each test."""
        # Reset logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.setLevel(logging.WARNING)
    
    def test_get_log_level_valid_values(self):
        """Test that valid log level strings return correct constants."""
        test_cases = [
            ('DEBUG', logging.DEBUG),
            ('debug', logging.DEBUG),
            ('INFO', logging.INFO),
            ('info', logging.INFO),
            ('WARNING', logging.WARNING),
            ('warning', logging.WARNING),
            ('WARN', logging.WARNING),
            ('warn', logging.WARNING),
            ('ERROR', logging.ERROR),
            ('error', logging.ERROR),
            ('CRITICAL', logging.CRITICAL),
            ('critical', logging.CRITICAL),
            ('FATAL', logging.CRITICAL),
            ('fatal', logging.CRITICAL),
        ]
        
        for level_string, expected_level in test_cases:
            with self.subTest(level_string=level_string):
                result = get_log_level(level_string)
                self.assertEqual(result, expected_level)
    
    def test_get_log_level_invalid_values(self):
        """Test that invalid log level strings fall back to INFO."""
        invalid_levels = ['INVALID', 'TRACE', '123', '', 'foo']
        
        for level_string in invalid_levels:
            with self.subTest(level_string=level_string):
                result = get_log_level(level_string)
                self.assertEqual(result, logging.INFO)
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_configure_logging_from_env_var(self):
        """Test that LOG_LEVEL environment variable is respected."""
        configure_logging()
        
        # Check that root logger level is set to DEBUG
        self.assertEqual(logging.root.level, logging.DEBUG)
        
        # Test that debug messages are actually logged
        with self.assertLogs(level='DEBUG') as log:
            logger = logging.getLogger('test')
            logger.debug('Test debug message')
            self.assertIn('Test debug message', log.output[0])
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'ERROR'})
    def test_configure_logging_from_env_var_error(self):
        """Test that LOG_LEVEL=ERROR sets appropriate level."""
        configure_logging()
        
        # Check that root logger level is set to ERROR
        self.assertEqual(logging.root.level, logging.ERROR)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_configure_logging_default_info(self):
        """Test that default log level is INFO when LOG_LEVEL is not set."""
        configure_logging()
        
        # Check that root logger level is set to INFO (default)
        self.assertEqual(logging.root.level, logging.INFO)
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'INVALID'})
    def test_configure_logging_invalid_env_var(self):
        """Test that invalid LOG_LEVEL falls back to INFO and logs warning."""
        with self.assertLogs('log_config', level='WARNING') as log:
            configure_logging()
            
            # Check that warning was logged about invalid value
            self.assertTrue(any('Invalid LOG_LEVEL value' in msg for msg in log.output))
        
        # Check that root logger level falls back to INFO
        self.assertEqual(logging.root.level, logging.INFO)
    
    def test_configure_logging_override_parameter(self):
        """Test that log_level parameter overrides environment variable."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'ERROR'}):
            configure_logging(log_level='DEBUG')
            
            # Check that parameter override works
            self.assertEqual(logging.root.level, logging.DEBUG)
    
    def test_debug_messages_visible_with_debug_level(self):
        """Test that debug messages are visible when LOG_LEVEL=DEBUG."""
        configure_logging(log_level='DEBUG')
        
        logger = logging.getLogger('test_logger')
        
        with self.assertLogs(logger, level='DEBUG') as log:
            logger.debug('This is a debug message')
            logger.info('This is an info message')
            logger.warning('This is a warning message')
            
            # All three messages should be captured
            self.assertEqual(len(log.output), 3)
            self.assertIn('debug message', log.output[0])
            self.assertIn('info message', log.output[1])
            self.assertIn('warning message', log.output[2])
    
    def test_debug_messages_hidden_with_info_level(self):
        """Test that debug messages are hidden when LOG_LEVEL=INFO."""
        configure_logging(log_level='INFO')
        
        logger = logging.getLogger('test_logger')
        
        with self.assertLogs(logger, level='INFO') as log:
            logger.debug('This is a debug message')
            logger.info('This is an info message')
            logger.warning('This is a warning message')
            
            # Only info and warning messages should be captured
            self.assertEqual(len(log.output), 2)
            self.assertIn('info message', log.output[0])
            self.assertIn('warning message', log.output[1])


if __name__ == '__main__':
    unittest.main()