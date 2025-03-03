from typing import List
import sys
import datetime
import logging

from multimethod import multimethod


#logger = logging.getLogger(__name__)
logger = None
APP_NAME = "AppName"
def initLogger(log_file_name_prefix:str="python-glog"):
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

    # global logger
    # if logger is None:
    #     logger = logger0
    return logger

logger = initLogger(APP_NAME)
