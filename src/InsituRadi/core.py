from pathlib import Path
import numpy as np
from InsituRadi.tls_pcd import TLS_PCDs
from InsituRadi import cloudcompare_commands as cc
from InsituRadi.utils import del_files_in_folder


def preprocessing_radiometric_compensation(input_folder_pcd: Path, output_folder: Path, fileformat_input: str,
                                           input_folder_e57: Path = None, radius: float = 0.05,
                                           ascii_delimiter: str = " ", ascii_header=None, range_min: float = -np.inf,
                                           range_max: float = np.inf, intensity_min: float = -np.inf,
                                           intensity_max: float = np.inf, surf_var_min: float = -np.inf,
                                           surf_var_max: float = np.inf, aoi_min: float = -np.inf,
                                           aoi_max: float = np.inf, pcds_number_patch_min: int = -np.inf,
                                           pcds_number_patch_max: int = np.inf):
    """
    Preprocess radiometric compensation by filtering and calculating point cloud properties.

    Parameters:
    input_folder_pcd (Path): Path to the input point cloud folder.
    output_folder (Path): Path to store intermediate and final results.
    fileformat_input (str): Format of input files (e.g., 'asc', 'e57').
    input_folder_e57 (Path): Path to E57 files for scanner center.
    radius (float): Radius for neighborhood search, normals, surface variation.
    ascii_delimiter (str): Delimiter for ASCII files.
    ascii_header (list): Header for ASCII files.
    range_min, range_max (float): Range bounds for filtering.
    intensity_min, intensity_max (float): Intensity bounds for filtering.
    surf_var_min, surf_var_max (float): Surface variation bounds for filtering.
    aoi_min, aoi_max (float): Angle of incidence bounds for filtering.
    pcds_number_patch_min, pcds_number_patch_max (int): Number of different scans in patch bounds for filtering.
    """

    if ascii_header is None:
        ascii_header = ["x", "y", "z", "intensity", "red", "green", "blue", "reflectance", "name"]

    # Convert ASCII files to PLY format if needed
    if fileformat_input == 'asc':
        pcd_ascii = TLS_PCDs.load_pcds_from_ascii(input_folder_pcd, header=ascii_header, delimiter=ascii_delimiter)
        pcd_ascii.write_pcds_in_separate_files_ply(output_folder / "ply_files")
        input_folder_pcd = output_folder / "ply_files"
    elif fileformat_input == 'e57':
        input_folder_e57 = input_folder_pcd

    # Subsample point clouds for patch calculation (using CloudCompare commands)
    cc.multistage_spatial_ss(input_folder_pcd, output_folder / "subsampled_merged",
                             radius=radius * 2, input_filetype=fileformat_input)

    # Estimate normals for each point in the cloud
    cc.estimate_normals(input_folder_pcd, output_folder / "pcd_normals", radius=radius, input_filetype=fileformat_input)
    if fileformat_input == 'asc':
        del_files_in_folder(output_folder / "ply_files")

    # Calculate surface variation
    cc.estimate_surface_variation(output_folder / "pcd_normals", output_folder / "surface_variation", radius=radius)
    del_files_in_folder(output_folder / "pcd_normals")

    # Load point cloud data and calculate additional properties (Distance, AOI, neighbors)
    pcd = TLS_PCDs.load_pcds_from_ply(output_folder / "surface_variation")
    pcd.set_scanner_center_from_e57(input_folder_e57)
    pcd.estimate_range()
    pcd.estimate_aoi()
    pcd.find_patch_number(output_folder / "subsampled_merged", radius=radius)
    pcd.count_unique_pcds_in_patch()
    pcd.write_pcds_in_separate_files_ply(output_folder / "information_pcds")
    del_files_in_folder(output_folder / "surface_variation")

    # Filter point clouds based on given scalar field bounds for radiometric calibration estimation
    filtered_pcd = pcd.filter_pcd(
        sf_names=["scalar_Range", "scalar_Intensity", "scalar_AOI", "scalar_Different_pcds_patch",
                  "scalar_Surface_variation_(" + str(radius) + ")"],
        sf_bounds=[(range_min, range_max), (intensity_min, intensity_max), (aoi_min, aoi_max),
                   (pcds_number_patch_min, pcds_number_patch_max), (surf_var_min, surf_var_max)]
    )

    # Save filtered point clouds to output folder
    filtered_pcd.write_pcds_in_separate_files_ply(output_folder / "filtered_pcds")

    return
