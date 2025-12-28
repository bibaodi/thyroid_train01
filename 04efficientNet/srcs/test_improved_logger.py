#!/usr/bin/env python3
"""
Test script for the ImprovedLogger class
"""

import os
import sys

# Add the current directory to the path to import improved_logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_logger import ImprovedLogger


def test_improved_logger():
    """Test the ImprovedLogger class"""
    print("Testing ImprovedLogger...")
    
    # Create an instance of ImprovedLogger
    improved_logger = ImprovedLogger(
        name="test_logger",
        log_dir="test_logs",
        log_level="INFO"
    )
    
    # Set up the logger
    logger = improved_logger.setup()
    
    # Test logging
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    print("Test completed. Check the test_logs directory for log files.")


if __name__ == "__main__":
    test_improved_logger()