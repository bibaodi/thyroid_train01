"""
File: findDuplicateFileByUIDBasedFilename.py
Author: eton.bi
Date: 2023-03-25
Version: 1.0
Description: Finds duplicate image files between two directories.
"""

#!/bin/env python
import argparse
import os
import hashlib
import logging
from typing import Dict, List, Union, Optional

# Configure logging
# Update logging format to include line numbers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s[LINE:%(lineno)d]-%(message)s'
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
HASH_CHUNK_SIZE = 8192  # For file hashing

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file's contents with error handling."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(HASH_CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, PermissionError) as error:
        logging.error(f"File read error: {file_path} - {error}")
        raise
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from core_utils import validate_paths, recursive_file_search

class DuplicateFileFinder:
    def __init__(self, source_dir: str, reference_dir: str, 
                 output_file: str, match_method: str = 'filename'):
        self.source_dir = os.path.abspath(source_dir)
        self.reference_dir = os.path.abspath(reference_dir)
        self.output_file = output_file
        self.match_method = match_method
        self.reference_index: Dict[Union[str, bytes], List[str]] = {}

    def run(self) -> None:
        """Orchestrate the duplicate finding workflow."""
        if not self._validate_directory_access() or not self._validate_output_path():
            return
        
        try:
            self._build_reference_index()
            matches = self._find_source_matches()
            self._write_match_results(matches)
        except Exception as error:
            logging.error(f"Processing failed: {error}")
            raise

    def _validate_directory_access(self) -> bool:
        """Verify directory accessibility with comprehensive checks."""
        dir_checks = [
            (self.source_dir, "Source"),
            (self.reference_dir, "Reference")
        ]
        
        for path, name in dir_checks:
            if not os.path.exists(path):
                logging.error(f"{name} directory not found: {path}")
                return False
            if not os.access(path, os.R_OK):
                logging.error(f"{name} directory inaccessible: {path}")
                return False
        return True

    def _validate_output_path(self) -> bool:
        """Validate output file path is writable"""
        output_dir = os.path.dirname(os.path.abspath(self.output_file)) or '.'
        if not os.access(output_dir, os.W_OK):
            logging.error(f"Output directory not writable: {output_dir}")
            return False
        return True

    def _build_reference_index(self):
        """Build search index for reference files"""
        self.reference_index = self._create_file_index(self.reference_dir)

    def _find_source_matches(self) -> List[str]:
        """Find matches in source directory against reference index"""
        return self._find_file_matches(self.source_dir)

    def _write_match_results(self, matches: List[str]):
        """Write results to output file"""
        try:
            with open(self.output_file, 'w') as f:
                f.writelines(matches)
            logging.info(f"Wrote {len(matches)} matches to {self.output_file}")
        except IOError as e:
            logging.error(f"Failed to write output: {e}")

    def _get_file_key(self, file_path: str, filename: str) -> Union[str, bytes]:
        """Unified method to get file key based on match method"""
        if self.match_method == "content":
            return compute_file_hash(file_path)
        return os.path.splitext(filename)[0].lower()

    def _create_file_index(self, directory_path: str) -> Dict[Union[str, bytes], List[str]]:
        """Build file index for a directory"""
        file_index: Dict[Union[str, bytes], List[str]] = {}
        abs_dir_path = os.path.abspath(directory_path)
        parent_dir = os.path.dirname(abs_dir_path)
        
        logging.info(f"Building index for: {directory_path} ({self.match_method} method)")

        for root, _, filenames in os.walk(abs_dir_path):
            for filename in filenames:
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension not in IMAGE_EXTENSIONS:
                    continue

                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, parent_dir)
                
                try:
                    index_key = self._get_file_key(file_path, filename)
                    file_index.setdefault(index_key, []).append(relative_path)
                except Exception as error:
                    logging.warning(f"Skipping unreadable file {file_path}: {error}")
        return file_index

    def _find_file_matches(self, search_dir: str) -> List[str]:
        """Identify matching files in a directory"""
        matches = []
        abs_search_path = os.path.abspath(search_dir)
        parent_dir = os.path.dirname(abs_search_path)
        
        logging.info(f"Searching for matches in: {search_dir}")

        for root, _, filenames in os.walk(abs_search_path):
            for filename in filenames:
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension not in IMAGE_EXTENSIONS:
                    continue

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, parent_dir)
                
                try:
                    search_key = self._get_file_key(file_path, filename)
                    
                    if search_key in self.reference_index:
                        matches.extend(
                            f"{rel_path},{ref_path}\n"
                            for ref_path in self.reference_index[search_key]
                        )
                except Exception as error:
                    logging.warning(f"Skipping unreadable file {file_path}: {error}")
        return matches


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Find duplicate images between two directories.')
    parser.add_argument('source_dir', help='Primary directory containing images to check')
    parser.add_argument('reference_dir', help='Comparison directory containing reference images')
    parser.add_argument('-o', '--output', default='duplicate-file-matches.csv', help='Output file path for results')
    parser.add_argument('--match-method', choices=['content', 'filename'], default='filename',
                      help='Comparison method: file contents (MD5) or base filename')
    return parser.parse_args()


def main():
    args = parse_arguments()

    finder = DuplicateFileFinder(
        args.source_dir,
        args.reference_dir,
        args.output,
        args.match_method
    )
    finder.run()
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
    except Exception as error:
        logging.error(f"Critical failure: {error}")
        raise SystemExit(1) from error
