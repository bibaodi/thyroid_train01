from typing import List
import sys
import datetime
import logging

"""
How to set up Logging for Python Projects

    1. Logger: Loggers are instantiated using logging.getLogger(), Use __name__ to automatically name your loggers
    2. Formatters: A logger needs a ‘formatter’ and ‘handler’ to specify the format and location of the log messages
    3. Handlers: If a handler is not defined, you will not see any log message outputs
"""

#logger = logging.getLogger(__name__)
glogger = None
APP_NAME = "pyglog"
def initLogger(log_file_name_prefix:str="pyglog"):
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string
    formatted_date_time = now.strftime("%y%m%dT%H%M%S")
    # Create the log file name
    log_file_name = f"{log_file_name_prefix}_{formatted_date_time}.log"
    _ver = sys.version_info
    kwargs = {
        'filename': log_file_name,
        'level': logging.DEBUG,
        'format': '%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
        'datefmt': '%Y%m%d %H:%M:%S'
    }
    
    if _ver.minor >= 10:
        kwargs['encoding'] = 'utf-8'
    else:
        print(f"WARNING: this Program develop in Python3.10.12, Current Version May has Problem in `pathlib.Path` to `str` convert.")
    logging.basicConfig(**kwargs)
    
    logger = logging.getLogger(log_file_name_prefix)

    global glogger
    if glogger is None:
         glogger = logger
         print(f"glogger is None, set to {glogger}")
    return logger

def get_logger():
    global glogger
    if glogger is None:
        glogger = initLogger(APP_NAME)
    return glogger

#logger = initLogger(APP_NAME)
