import os
from pathlib import Path
import numpy as np
from insituradi.tls_pcd import TLS_PCDs


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


def cut_pcd_series(input_folder: Path, output_folders: list[Path], bounds: list[tuple[np.ndarray, float, float]]):
    """
    Processes .e57 point cloud files by cropping them into cylindrical regions defined in bounds,
    and saves results to corresponding output folders as .ply files.

    Parameters:
        input_folder (Path): Folder containing .e57 point cloud files to process
        output_folders (list[Path]): Output directories for each bound region
        bounds (list[tuple]): List of region specifications where each tuple contains:
            - np.ndarray[3]: Base point [x, y, z] defining:
                - XY center for radial selection
                - Z base for vertical range
            - float: Height (vertical range added to base Z)
            - float: Radius (max horizontal distance from base XY)

    Process:
        1. Filters points vertically (Z-axis) first for efficiency
        2. Filters remaining points radially (XY-plane)
        3. Saves results to corresponding output folder for each region

    Requirements:
        - len(output_folders) must equal len(bounds)
    """

    # Get all .e57 files in the input folder
    path_pcds = list(input_folder.glob("*.e57"))

    for file in path_pcds:
        # Load point cloud data from E57 file
        pcd = TLS_PCDs.load_single_pcd_e57(file)

        # Process each region individually
        for bound, output_folder in zip(bounds, output_folders):
            # First filter: Vertical (Z-axis) slice [base_z, base_z + height]
            selection = (pcd.coord[:, 2] >= bound[0][2]) & (pcd.coord[:, 2] <= (bound[0][2] + bound[1]))
            pcd_selected = pcd.select_by_index(selection)

            # Second filter: Radial (XY-plane) selection around base_point's XY coordinates
            translated_coord = pcd_selected.coord - bound[0]    # Center coordinates at base_point
            distances = np.linalg.norm(translated_coord[:,0:2], axis=1) # Calculate XY distances
            pcd_final = pcd_selected.select_by_index(distances<=bound[2])

            # Save filtered point cloud to corresponding output folder
            pcd_final.write_pcds_in_separate_files_ply(output_folder)
            
        print("Finished pcd: "+file.stem)

    return
