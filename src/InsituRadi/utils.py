import os
from pathlib import Path


def del_files_in_folder(folder: Path):
    """
    Delete all files and the folder itself from the specified directory.

    Parameters:
    folder (Path): Path to the folder to be deleted.
    """

    # Get a list of all files in the folder
    files = list(folder.glob("*"))

    # Remove each file in the folder
    for file in files:
        os.remove(file)

    # Remove the empty folder itself
    os.rmdir(folder)

