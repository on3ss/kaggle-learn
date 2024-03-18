import os


def root_dir() -> str:
    """Get the root directory of the project.

    Returns:
        str: The root directory of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def file_path(directory: str, filename: str) -> str:
    """Get the path of a file within the project's root directory.

    Args:
        directory (str): The directory within the root directory where the file is located.
        filename (str): The name of the file.

    Returns:
        str: The path of the file within the project's root directory.
    """
    return os.path.join(root_dir(), directory, filename)
