"""
Dynamic Case Data Filter
Supports configurable filter nodes for case validation
"""
import os
from typing import Dict, List, Any, Callable, Tuple

from ..utils.orientation_detector import OrientationDetector
from ..utils.thyroid_nodule_detector import ThyroidNoduleDetector
from ..utils.enums import ImageOrientation


class CaseDataFilter:
    """
    Dynamic case data filter with configurable filter nodes.
    Each filter node can be enabled/disabled and configured independently.
    """

    def __init__(self, filter_config: Dict[str, Any] = None, orientation_model_path: str = None, thyroid_nodule_model_path: str = None):
        """
        Initialize the case data filter.

        Args:
            filter_config: Dictionary containing filter configuration
            orientation_model_path: Path to orientation detection model (optional)
            thyroid_nodule_model_path: Path to thyroid nodule detection model (optional)
                Example:
                {
                    "min_images_check": {
                        "enabled": True,
                        "min_count": 2
                    },
                    "max_images_check": {
                        "enabled": False,
                        "max_count": 100
                    },
                    "orientation_check": {
                        "enabled": True,
                        "require_both_orientations": True
                    },
                    "thyroid_nodule_check": {
                        "enabled": True,
                        "require_nodules": True,
                        "min_nodule_count": 1
                    }
                }
        """
        self.m_filter_config = filter_config or {}
        self.m_results = {}
        self.m_orientation_detector = None
        self.m_thyroid_nodule_detector = None

        # Initialize orientation detector if model path is provided
        if orientation_model_path and os.path.exists(orientation_model_path):
            try:
                self.m_orientation_detector = OrientationDetector(orientation_model_path)
                print(f"✅ Orientation detector initialized with model: {orientation_model_path}")
            except Exception as e:
                print(f"⚠️  Could not initialize orientation detector: {e}")
                self.m_orientation_detector = None

        # Initialize thyroid nodule detector if model path is provided
        if thyroid_nodule_model_path and os.path.exists(thyroid_nodule_model_path):
            try:
                self.m_thyroid_nodule_detector = ThyroidNoduleDetector(thyroid_nodule_model_path)
                print(f"✅ Thyroid nodule detector initialized with model: {thyroid_nodule_model_path}")
            except Exception as e:
                print(f"⚠️  Could not initialize thyroid nodule detector: {e}")
                self.m_thyroid_nodule_detector = None

        # Register available filter nodes
        self.m_filter_nodes = {
            "min_images_check": self._check_min_images,
            "max_images_check": self._check_max_images,
            "file_extension_check": self._check_file_extensions,
            "case_name_pattern_check": self._check_case_name_pattern,
            "orientation_check": self._check_orientations,
            "thyroid_nodule_check": self._check_thyroid_nodules,
            "nodule_size_check": self._check_nodule_sizes
        }


    def _is_node_enabled(self, node_name: str) -> bool:
        """Check if a filter node is enabled in the configuration."""
        node_config = self.m_filter_config.get(node_name, {})
        return node_config.get("enabled", False)

    def _get_image_files(self, case_folder: str) -> List[str]:
        """Get all image files in the case folder."""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        image_files = []

        for file in os.listdir(case_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(case_folder, file))

        return image_files

    def _check_min_images(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if case folder has minimum required number of images.

        Args:
            case_folder: Path to case folder
            config: Node configuration with 'min_count' parameter

        Returns:
            Dictionary with check results
        """
        min_count = config.get("min_count", 2)
        image_files = self._get_image_files(case_folder)
        actual_count = len(image_files)

        passed = actual_count >= min_count

        return {
            "passed": passed,
            "actual_count": actual_count,
            "required_min": min_count,
            "message": f"Found {actual_count} images, required minimum: {min_count}"
        }

    def _check_max_images(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if case folder doesn't exceed maximum number of images.

        Args:
            case_folder: Path to case folder
            config: Node configuration with 'max_count' parameter

        Returns:
            Dictionary with check results
        """
        max_count = config.get("max_count", 100)
        image_files = self._get_image_files(case_folder)
        actual_count = len(image_files)

        passed = actual_count <= max_count

        return {
            "passed": passed,
            "actual_count": actual_count,
            "allowed_max": max_count,
            "message": f"Found {actual_count} images, maximum allowed: {max_count}"
        }

    def _check_file_extensions(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if all files have allowed extensions.

        Args:
            case_folder: Path to case folder
            config: Node configuration with 'allowed_extensions' parameter

        Returns:
            Dictionary with check results
        """
        allowed_extensions = config.get("allowed_extensions", ['.png', '.jpg', '.jpeg'])
        allowed_extensions = [ext.lower() for ext in allowed_extensions]

        all_files = os.listdir(case_folder)
        invalid_files = []

        for file in all_files:
            if os.path.isfile(os.path.join(case_folder, file)):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext and file_ext not in allowed_extensions:
                    invalid_files.append(file)

        passed = len(invalid_files) == 0

        return {
            "passed": passed,
            "invalid_files": invalid_files,
            "allowed_extensions": allowed_extensions,
            "message": f"Found {len(invalid_files)} files with invalid extensions"
        }

    def _check_case_name_pattern(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if case name matches required pattern.

        Args:
            case_folder: Path to case folder
            config: Node configuration with 'pattern' parameter (regex)

        Returns:
            Dictionary with check results
        """
        import re

        pattern = config.get("pattern", ".*")  # Default: match anything
        case_name = os.path.basename(case_folder)

        try:
            matched = bool(re.match(pattern, case_name))
        except re.error as e:
            return {
                "passed": False,
                "error": f"Invalid regex pattern: {e}",
                "case_name": case_name,
                "pattern": pattern
            }

        return {
            "passed": matched,
            "case_name": case_name,
            "pattern": pattern,
            "message": f"Case name '{case_name}' {'matches' if matched else 'does not match'} pattern '{pattern}'"
        }

    def _check_orientations(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if case has required image orientations.

        Args:
            case_folder: Path to case folder
            config: Node configuration with orientation requirements
                - require_both_orientations: bool (default True)
                - require_transverse: bool (default False)
                - require_longitudinal: bool (default False)

        Returns:
            Dictionary with check results
        """
        if self.m_orientation_detector is None:
            return {
                "passed": False,
                "error": "Orientation detector not initialized",
                "message": "Cannot check orientations without orientation model"
            }

        # Get configuration
        require_both = config.get("require_both_orientations", True)
        require_transverse = config.get("require_transverse", False)
        require_longitudinal = config.get("require_longitudinal", False)

        # Get image files
        image_files = self._get_image_files(case_folder)

        if not image_files:
            return {
                "passed": False,
                "message": "No images found for orientation check",
                "orientation_results": []
            }

        try:
            # Detect orientations
            orientation_results = self.m_orientation_detector.detect_orientations(image_files)
            summary = self.m_orientation_detector.get_orientation_summary(orientation_results)

            # Check requirements
            passed = True
            messages = []

            if require_both:
                if not summary['has_both_orientations']:
                    passed = False
                    messages.append("Case must have both transverse and longitudinal images")
                else:
                    messages.append("Case has both orientations ✓")

            if require_transverse:
                if not summary['has_transverse']:
                    passed = False
                    messages.append("Case must have at least one transverse image")
                else:
                    messages.append("Case has transverse images ✓")

            if require_longitudinal:
                if not summary['has_longitudinal']:
                    passed = False
                    messages.append("Case must have at least one longitudinal image")
                else:
                    messages.append("Case has longitudinal images ✓")

            # If no specific requirements, just check if we could detect orientations
            if not (require_both or require_transverse or require_longitudinal):
                if summary['unknown_count'] == summary['total_images']:
                    passed = False
                    messages.append("Could not detect orientation for any images")
                else:
                    messages.append("Orientation detection completed")

            return {
                "passed": passed,
                "message": "; ".join(messages),
                "orientation_summary": summary,
                "orientation_results": orientation_results,
                "transverse_count": summary['transverse_count'],
                "longitudinal_count": summary['longitudinal_count'],
                "unknown_count": summary['unknown_count']
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Error during orientation detection: {e}"
            }

    def _check_thyroid_nodules(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if case has required thyroid nodules.

        Args:
            case_folder: Path to case folder
            config: Node configuration with nodule requirements
                - require_nodules: bool (default True)
                - min_nodule_count: int (default 1)
                - confidence_threshold: float (default 0.5)

        Returns:
            Dictionary with check results
        """
        if self.m_thyroid_nodule_detector is None:
            return {
                "passed": False,
                "error": "Thyroid nodule detector not initialized",
                "message": "Cannot check thyroid nodules without nodule detection model"
            }

        # Get configuration
        require_nodules = config.get("require_nodules", True)
        min_nodule_count = config.get("min_nodule_count", 1)
        confidence_threshold = config.get("confidence_threshold", 0.5)

        # Update detector confidence threshold if specified
        if confidence_threshold != self.m_thyroid_nodule_detector.m_confidence_threshold:
            self.m_thyroid_nodule_detector.set_confidence_threshold(confidence_threshold)

        # Get image files
        image_files = self._get_image_files(case_folder)

        if not image_files:
            return {
                "passed": False,
                "message": "No images found for thyroid nodule check",
                "nodule_results": []
            }

        try:
            # Detect thyroid nodules
            nodule_results = self.m_thyroid_nodule_detector.detect_nodules(image_files)
            summary = self.m_thyroid_nodule_detector.get_nodule_summary(nodule_results)

            # Check requirements
            passed = True
            messages = []

            if require_nodules:
                if not summary['has_any_nodules']:
                    passed = False
                    messages.append("Case must have at least one image with thyroid nodules")
                else:
                    messages.append("Case has images with thyroid nodules ✓")

            if min_nodule_count > 0:
                total_nodules = summary['total_nodules_detected']
                if total_nodules < min_nodule_count:
                    passed = False
                    messages.append(f"Case must have at least {min_nodule_count} nodules (found {total_nodules})")
                else:
                    messages.append(f"Case has sufficient nodules ({total_nodules} >= {min_nodule_count}) ✓")

            # If no specific requirements, just check if we could detect anything
            if not require_nodules and min_nodule_count == 0:
                if summary['error_count'] == summary['total_images']:
                    passed = False
                    messages.append("Could not process any images for nodule detection")
                else:
                    messages.append("Nodule detection completed")

            return {
                "passed": passed,
                "message": "; ".join(messages),
                "nodule_summary": summary,
                "nodule_results": nodule_results,
                "images_with_nodules": summary['images_with_nodules'],
                "total_nodules_detected": summary['total_nodules_detected'],
                "detection_rate": summary['detection_rate']
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Error during thyroid nodule detection: {e}"
            }

    def _check_nodule_sizes(self, case_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if nodule sizes are within acceptable ranges.

        Args:
            case_folder: Path to case folder
            config: Node configuration with size requirements
                - size_ranges: List of tuples [(min_width_percent, min_height_percent), (max_width_percent, max_height_percent)]
                  Example: [(0.1, 0.1), (0.5, 0.5)] means nodules must be:
                  - width >= 10% and height >= 10% of image size (minimum)
                  - width <= 50% and height <= 50% of image size (maximum)
                - check_all_nodules: bool (default True) - if True, all nodules must meet criteria
                - confidence_threshold: float (default 0.5)

        Returns:
            Dictionary with check results
        """
        if self.m_thyroid_nodule_detector is None:
            return {
                "passed": False,
                "error": "Thyroid nodule detector not initialized",
                "message": "Cannot check nodule sizes without nodule detection model"
            }

        # Get configuration
        size_ranges = config.get("size_ranges", [(0.05, 0.05), (0.8, 0.8)])  # Default: 5% to 80%
        check_all_nodules = config.get("check_all_nodules", True)
        confidence_threshold = config.get("confidence_threshold", 0.5)

        # Validate size_ranges format
        if not isinstance(size_ranges, list) or len(size_ranges) != 2:
            return {
                "passed": False,
                "error": "Invalid size_ranges format",
                "message": "size_ranges must be a list of 2 tuples: [(min_w, min_h), (max_w, max_h)]"
            }

        try:
            min_range, max_range = size_ranges
            min_width_percent, min_height_percent = min_range
            max_width_percent, max_height_percent = max_range
        except (ValueError, TypeError):
            return {
                "passed": False,
                "error": "Invalid size_ranges values",
                "message": "size_ranges must contain numeric tuples: [(min_w, min_h), (max_w, max_h)]"
            }

        # Update detector confidence threshold if specified
        if confidence_threshold != self.m_thyroid_nodule_detector.m_confidence_threshold:
            self.m_thyroid_nodule_detector.set_confidence_threshold(confidence_threshold)

        # Get image files
        image_files = self._get_image_files(case_folder)

        if not image_files:
            return {
                "passed": False,
                "message": "No images found for nodule size check",
                "nodule_results": []
            }

        try:
            # Detect thyroid nodules with size information
            nodule_results = self.m_thyroid_nodule_detector.detect_nodules(image_files)

            # Analyze nodule sizes
            total_nodules = 0
            valid_size_nodules = 0
            invalid_size_nodules = 0
            size_violations = []

            for result in nodule_results:
                for detection in result.get('detections', []):
                    if 'size_info' in detection:
                        total_nodules += 1
                        size_info = detection['size_info']

                        width_percent = size_info['width_percent']
                        height_percent = size_info['height_percent']

                        # Check if nodule size is within acceptable range
                        width_valid = min_width_percent <= width_percent <= max_width_percent
                        height_valid = min_height_percent <= height_percent <= max_height_percent

                        if width_valid and height_valid:
                            valid_size_nodules += 1
                        else:
                            invalid_size_nodules += 1
                            violation = {
                                'image_name': result.get('image_name', 'unknown'),
                                'width_percent': width_percent,
                                'height_percent': height_percent,
                                'width_valid': width_valid,
                                'height_valid': height_valid,
                                'confidence': detection.get('confidence', 0)
                            }
                            size_violations.append(violation)

            # Determine if check passes
            passed = True
            messages = []

            if total_nodules == 0:
                passed = False
                messages.append("No nodules found for size checking")
            else:
                if check_all_nodules:
                    # All nodules must meet size criteria
                    if invalid_size_nodules > 0:
                        passed = False
                        messages.append(f"Found {invalid_size_nodules}/{total_nodules} nodules with invalid sizes")
                    else:
                        messages.append(f"All {total_nodules} nodules have valid sizes ✓")
                else:
                    # At least one nodule must meet size criteria
                    if valid_size_nodules == 0:
                        passed = False
                        messages.append("No nodules meet the size criteria")
                    else:
                        messages.append(f"{valid_size_nodules}/{total_nodules} nodules have valid sizes ✓")

                # Add size range info to message
                messages.append(f"Size range: {min_width_percent:.1f}%-{max_width_percent:.1f}% width, {min_height_percent:.1f}%-{max_height_percent:.1f}% height")

            return {
                "passed": passed,
                "message": "; ".join(messages),
                "total_nodules": total_nodules,
                "valid_size_nodules": valid_size_nodules,
                "invalid_size_nodules": invalid_size_nodules,
                "size_violations": size_violations,
                "size_ranges": size_ranges,
                "check_all_nodules": check_all_nodules,
                "nodule_results": nodule_results
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "message": f"Error during nodule size checking: {e}"
            }

    def add_custom_filter(self, node_name: str, filter_func: Callable) -> None:
        """
        Add a custom filter node.

        Args:
            node_name: Name of the filter node
            filter_func: Function that takes (case_folder, config) and returns dict with results
        """
        self.m_filter_nodes[node_name] = filter_func

    def get_available_filters(self) -> List[str]:
        """Get list of available filter node names."""
        return list(self.m_filter_nodes.keys())

    def get_filter_results(self) -> Dict[str, Any]:
        """Get the last filter results."""
        return self.m_results

    def filter_case(self, case_folder: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply all enabled filters to a case folder.

        Args:
            case_folder: Path to the case folder

        Returns:
            Tuple of (pass_filter: bool, results: dict)
            - pass_filter: True if case passes all enabled filters
            - results: Dictionary containing results from all filter nodes
        """
        if not os.path.exists(case_folder):
            return False, {"error": f"Case folder does not exist: {case_folder}"}

        if not os.path.isdir(case_folder):
            return False, {"error": f"Path is not a directory: {case_folder}"}

        self.m_results = {
            "case_folder": case_folder,
            "case_name": os.path.basename(case_folder),
            "filter_results": {}
        }

        overall_pass = True

        # Apply each enabled filter node
        for node_name, node_func in self.m_filter_nodes.items():
            if self._is_node_enabled(node_name):
                try:
                    node_result = node_func(case_folder, self.m_filter_config.get(node_name, {}))
                    self.m_results["filter_results"][node_name] = node_result

                    # If any filter fails, overall result is False
                    if not node_result.get("passed", False):
                        overall_pass = False

                except Exception as e:
                    self.m_results["filter_results"][node_name] = {
                        "passed": False,
                        "error": str(e)
                    }
                    overall_pass = False

        self.m_results["overall_passed"] = overall_pass
        return overall_pass, self.m_results


# Example usage and configuration
def create_default_filter_config() -> Dict[str, Any]:
    """Create a default filter configuration."""
    return {
        "min_images_check": {
            "enabled": True,
            "min_count": 2
        },
        "max_images_check": {
            "enabled": False,
            "max_count": 100
        },
        "file_extension_check": {
            "enabled": False,
            "allowed_extensions": ['.png', '.jpg', '.jpeg']
        },
        "case_name_pattern_check": {
            "enabled": False,
            "pattern": r"^[A-Za-z0-9_-]+$"  # Alphanumeric, underscore, hyphen only
        },
        "orientation_check": {
            "enabled": False,  # Disabled by default (requires orientation model)
            "require_both_orientations": True,
            "require_transverse": False,
            "require_longitudinal": False
        },
        "thyroid_nodule_check": {
            "enabled": False,  # Disabled by default (requires nodule detection model)
            "require_nodules": True,
            "min_nodule_count": 1,
            "confidence_threshold": 0.5
        },
        "nodule_size_check": {
            "enabled": False,  # Disabled by default (requires nodule detection model)
            "size_ranges": [(0.05, 0.05), (0.8, 0.8)],  # 5% to 80% of image size
            "check_all_nodules": True,  # All nodules must meet size criteria
            "confidence_threshold": 0.5
        }
    }
