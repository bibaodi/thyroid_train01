import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from  glog import logger

def perform_task():
    logger.debug("Performing a business task")
    # Your business logic code here
    logger.info("Task completed successfully")