import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from  glog import logger, initLogger, APP_NAME

import test_log_mod  # Import your business logic modules

def main():
    global APP_NAME
    APP_NAME = "logtest_log_main"
    initLogger("test_log_main")
    logger.info("Application started")
    # Call your business logic functions
    test_log_mod.perform_task()

if __name__ == "__main__":
    main()