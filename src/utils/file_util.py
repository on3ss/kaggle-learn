"""
Module for handling file paths relative to the root directory of the project.
"""

import os


def root_dir() -> str:
    """
    Get the root directory of the project.

    Returns:
        str: Absolute path to the root directory of the project.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def file_path(directory: str, filename: str) -> str:
    """
    Construct a file path relative to the root directory of the project.

    Args:
        directory (str): The name of the directory where the file is located.
        filename (str): The name of the file.

    Returns:
        str: Absolute path to the file.
    """
    return os.path.join(root_dir(), directory, filename)
