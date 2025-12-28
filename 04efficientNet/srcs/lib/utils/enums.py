"""
Enums for the project
"""
from enum import Enum


class ImageOrientation(Enum):
    """
    Enum for image orientation classification
    """
    TRANSVERSE = "transverse"
    LONGITUDINAL = "longitudinal"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"ImageOrientation.{self.name}"
