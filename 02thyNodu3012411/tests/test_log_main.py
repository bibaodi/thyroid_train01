import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog

import test_log_mod  # Import your business logic modules

def main():
    glog.glogger = glog.initLogger("logtest_main")
    glog.glogger.info("Application started")
    # Call your business logic functions
    test_log_mod.perform_task()

if __name__ == "__main__":
    main()