from enum import Flag

class TaskTypes(Flag):
    """
    Enum class to represent the task types.
    """
    Classify = 61  # 
    Detect = 62  # 
    Segment = 63  #
    UNKNOWN = 127  # Unknown format