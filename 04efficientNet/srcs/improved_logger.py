import logging
import os
from datetime import datetime


class ImprovedLogger:
    """
    An improved logger class that handles both file and console logging
    with better formatting and organization.
    """
    
    def __init__(self, name, log_dir, log_level='INFO'):
        """
        Initialize the ImprovedLogger.
        
        Args:
            name (str): Name of the logger
            log_dir (str): Directory to save log files
            log_level (str): Logging level (default: 'INFO')
        """
        self.name = name
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create log directory if it doesn't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def setup(self):
        """
        Set up and configure the logger with both file and console handlers.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_dir is specified
        if self.log_dir:
            # Create log filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{self.name}_{timestamp}.log"
            log_filepath = os.path.join(self.log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Also create a latest log file that gets overwritten each time
            latest_log_filepath = os.path.join(self.log_dir, f"{self.name}_latest.log")
            latest_file_handler = logging.FileHandler(latest_log_filepath, mode='w', encoding='utf-8')
            latest_file_handler.setLevel(self.log_level)
            latest_file_handler.setFormatter(formatter)
            logger.addHandler(latest_file_handler)
        
        # Prevent propagation to avoid duplicate logs in parent loggers
        logger.propagate = False
        
        return logger


# Example usage (commented out):
# if __name__ == "__main__":
#     # Create an instance of ImprovedLogger
#     improved_logger = ImprovedLogger(
#         name="test_logger",
#     )
#     
#     # Set up the logger
#     logger = improved_logger.setup()
#     
#     # Test logging
#     logger.info("This is an info message")
#     logger.warning("This is a warning message")
#     logger.error("This is an error message")