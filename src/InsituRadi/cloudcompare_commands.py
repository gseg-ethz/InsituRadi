import subprocess
from pathlib import Path
from InsituRadi.utils import del_files_in_folder
from InsituRadi.config import CLOUDCOMPARE_PATH


def spatial_ss_pcds_individually(input_folderpath: Path, output_folderpath: Path, radius: float = 0.1,
                                 input_filetype: str = "*",
                                 output_filetype: str = "ply"):
    """
    Perform spatial subsampling on point cloud files individually.

    Args:
        input_folderpath (Path): Path to the folder containing input point cloud files.
        output_folderpath (Path): Path to the folder where subsampled point clouds will be saved.
        radius (float): The radius for spatial subsampling. Default is 0.1.
        input_filetype (str): File type of the input point clouds. Default is "*" (all files).
        output_filetype (str): File type for the output point clouds. Default is "ply".
    """
    print("Start spatial subsampling.")
    output_folderpath.mkdir(parents=True, exist_ok=True)

    # Get all point cloud files in the input folder
    path_pcds = list(input_folderpath.glob("*." + input_filetype))

    for pcd_file in path_pcds:
        # Construct the CloudCompare command for spatial subsampling
        cc_command = "cd '" + str(CLOUDCOMPARE_PATH) + "'\n.\CloudCompare -SILENT -AUTO_SAVE OFF -o '" + str(
            pcd_file) + "' -SS SPATIAL " + str(
            radius) + " -C_EXPORT_FMT " + output_filetype.upper() + " -SAVE_CLOUDS FILE '" + str(
            output_folderpath / pcd_file.stem) + "." + output_filetype + "'"

        # Execute the command using PowerShell
        completed = subprocess.run(["powershell", "-Command", cc_command], capture_output=True)
        if completed.returncode != 0:
            print("An error occured: %s", completed.stderr)
        else:
            print("Point cloud " + pcd_file.stem + " has been sucessfully downsampled.")

    return


def merge_pcds(input_folderpath: Path, output_folderpath: Path, input_filetype: str = "*",
               output_name: str = "merged_pcd", output_filetype: str = "ply"):
    """
    Merge multiple point cloud files into a single file.

    Args:
        input_folderpath (Path): Path to the folder containing input point cloud files.
        output_folderpath (Path): Path to the folder where the merged point cloud will be saved.
        input_filetype (str): File type of the input point clouds. Default is "*" (all files).
        output_name (str): Name of the merged output file. Default is "merged_pcd".
        output_filetype (str): File type for the output point cloud. Default is "ply".
    """
    print("Start merging.")
    output_folderpath.mkdir(parents=True, exist_ok=True)

    # Get all point cloud files in the input folder
    path_pcds = list(input_folderpath.glob("*." + input_filetype))

    # Construct the command to open all point clouds
    open_pcds = ""
    for file in path_pcds:
        open_pcds = open_pcds + "-o '" + str(file) + "' "

    # Construct the CloudCompare command to merge the point clouds
    cc_command = "cd '" + str(
        CLOUDCOMPARE_PATH) + "'\n.\CloudCompare -SILENT -AUTO_SAVE OFF " + open_pcds + "-MERGE_CLOUDS -C_EXPORT_FMT " + output_filetype.upper() + " -SAVE_CLOUDS FILE '" + str(
        output_folderpath / output_name) + "." + output_filetype + "'"

    # Execute the command using PowerShell
    completed = subprocess.run(["powershell", "-Command", cc_command], capture_output=True)
    if completed.returncode != 0:
        print("An error occured: %s", completed.stderr)
    else:
        print("All point clouds successfully merged.")


def estimate_normals(input_folderpath: Path, output_folderpath: Path, radius: float, input_filetype: str = "*",
                     output_filetype: str = "ply"):
    """
    Estimate normals for point cloud files.

    Args:
        input_folderpath (Path): Path to the folder containing input point cloud files.
        output_folderpath (Path): Path to the folder where point clouds with normals will be saved.
        radius (float): The radius for normal estimation.
        input_filetype (str): File type of the input point clouds. Default is "*" (all files).
        output_filetype (str): File type for the output point clouds. Default is "ply".
    """
    print("Start calculation of point cloud normals.")
    output_folderpath.mkdir(parents=True, exist_ok=True)

    # Get all point cloud files in the input folder
    path_pcds = list(input_folderpath.glob("*." + input_filetype))

    for pcd_file in path_pcds:
        # Construct the CloudCompare command for normal estimation
        cc_command = "cd '" + str(CLOUDCOMPARE_PATH) + "'\n.\CloudCompare -SILENT -AUTO_SAVE OFF -o '" + str(
            pcd_file) + "' -OCTREE_NORMALS " + str(
            radius) + " -MODEL LS -C_EXPORT_FMT " + output_filetype.upper() + " -SAVE_CLOUDS FILE '" + str(
            output_folderpath / pcd_file.stem) + "." + output_filetype + "'"

        # Execute the command using PowerShell
        completed = subprocess.run(["powershell", "-Command", cc_command], capture_output=True)
        if completed.returncode != 0:
            print("An error occured: %s", completed.stderr)
        else:
            print("Normals of point cloud " + pcd_file.stem + " has been sucessfully calculated.")


def estimate_surface_variation(input_folderpath: Path, output_folderpath: Path, radius: float,
                               input_filetype: str = "*",
                               output_filetype: str = "ply"):
    """
    Estimate surface variation for point cloud files.

    Args:
        input_folderpath (Path): Path to the folder containing input point cloud files.
        output_folderpath (Path): Path to the folder where point clouds with surface variation will be saved.
        radius (float): The radius for surface variation estimation.
        input_filetype (str): File type of the input point clouds. Default is "*" (all files).
        output_filetype (str): File type for the output point clouds. Default is "ply".
    """
    print("Start calculation of point cloud surface variation.")
    output_folderpath.mkdir(parents=True, exist_ok=True)

    # Get all point cloud files in the input folder
    path_pcds = list(input_folderpath.glob("*." + input_filetype))

    for pcd_file in path_pcds:
        # Construct the CloudCompare command for surface variation estimation
        cc_command = "cd '" + str(CLOUDCOMPARE_PATH) + "'\n.\CloudCompare -SILENT -AUTO_SAVE OFF -o '" + str(
            pcd_file) + "' -FEATURE SURFACE_VARIATION " + str(
            radius) + " -C_EXPORT_FMT " + output_filetype.upper() + " -SAVE_CLOUDS FILE '" + str(
            output_folderpath / pcd_file.stem) + "." + output_filetype + "'"

        # Execute the command using PowerShell
        completed = subprocess.run(["powershell", "-Command", cc_command], capture_output=True)
        if completed.returncode != 0:
            print("An error occured: %s", completed.stderr)
        else:
            print("Surface variation of point cloud " + pcd_file.stem + " has been sucessfully calculated.")


def multistage_spatial_ss(input_folderpath: Path, output_folderpath: Path, radius: float, input_filetype: str = "*",
                          output_filetype: str = "ply"):
    """
    Perform multi-stage spatial subsampling on point cloud files.

    Args:
        input_folderpath (Path): Path to the folder containing input point cloud files.
        output_folderpath (Path): Path to the folder where the final subsampled point clouds will be saved.
        radius (float): The radius for spatial subsampling.
        input_filetype (str): File type of the input point clouds. Default is "*" (all files).
        output_filetype (str): File type for the output point clouds. Default is "ply".
    """
    print("Start multi-stage spatial subsampling.")

    # First stage: Subsample point clouds individually
    spatial_ss_pcds_individually(input_folderpath,
                                 output_folderpath / "subsampled_pcds_10cm", radius, input_filetype, "ply")

    # Second stage: Merge the subsampled point clouds
    merge_pcds(output_folderpath / "subsampled_pcds_10cm", output_folderpath / "merged_pcd")

    # Clean up: Delete the intermediate subsampled files
    del_files_in_folder(output_folderpath / "subsampled_pcds_10cm")

    # Third stage: Subsample the merged point cloud
    spatial_ss_pcds_individually(output_folderpath / "merged_pcd",
                                 output_folderpath, radius=radius, output_filetype=output_filetype)

    # Clean up: Delete the intermediate merged file
    del_files_in_folder(output_folderpath / "merged_pcd")
    print("Finish multi-stage spatial subsampling.")
    return
