import os
import hashlib
from pathlib import Path
from typing import Set, Union, List, Dict, Tuple

def safe_file_read(file_path: Union[str, Path], 
                  chunk_size: int = 8192) -> bytes:
    """Safe file read with error handling."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(chunk_size)
    except (IOError, PermissionError) as e:
        logging.error(f"File read error: {file_path} - {e}")
        raise

def validate_paths(*paths: Union[str, Path], 
                  check_exists: bool = True,
                  check_access: bool = True) -> bool:
    """Unified path validation for multiple directories/files."""
    for path in paths:
        path = Path(path).resolve()
        if check_exists and not path.exists():
            logging.error(f"Path not found: {path}")
            return False
        if check_access and not os.access(path, os.R_OK):
            logging.error(f"Path inaccessible: {path}")
            return False
    return True

def recursive_file_search(root_dir: Union[str, Path],
                         extensions: Set[str],
                         exclude_dirs: Set[str] = None) -> List[Path]:
    """Recursive file search with extension filtering."""
    root = Path(root_dir).resolve()
    found_files = []
    
    for entry in root.rglob('*'):
        if exclude_dirs and entry.is_dir() and entry.name in exclude_dirs:
            continue
        if entry.is_file() and entry.suffix.lower() in extensions:
            found_files.append(entry)
    return found_files

def standardized_json_handling(json_path: Union[str, Path],
                              mode: str = 'load') -> Union[Dict, bool]:
    """Unified JSON read/write operations."""
    try:
        json_path = Path(json_path).resolve()
        if mode == 'load':
            with json_path.open('r') as f:
                return json.load(f)
        elif mode == 'save':
            with json_path.open('w') as f:
                json.dump(data, f, indent=2)
            return True
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"JSON operation failed: {json_path} - {e}")
        return False
