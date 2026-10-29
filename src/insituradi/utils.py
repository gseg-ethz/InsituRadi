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


def cut_pcd_series(
    input_folder: Path,
    output_folder: Path,
    bounds: list[tuple[str, np.ndarray, float, float]],
    format: str = "e57",
):
    """
    Processes .e57/.ply point cloud files by cropping them into cylindrical regions defined in bounds,
    and saves results to corresponding output folders as .ply files.

    Parameters:
        input_folder (Path): Folder containing .e57 point cloud files to process
        output_folder (Path): Output directory
        bounds (list[tuple]): List of region specifications where each tuple contains:
            - str: Subfolder name for output (e.g., "region1")
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
    path_pcds = list(input_folder.glob(f"*.{format}"))

    for file in path_pcds:
        # Load point cloud data from E57 file
        if format == "e57":
            pcd = TLS_PCDs.load_single_pcd_e57(file)
        elif format == "ply":
            pcd = TLS_PCDs.load_single_pcd_ply(file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Process each region individually
        for bound in bounds:
            # First filter: Vertical (Z-axis) slice [base_z, base_z + height]
            selection = (pcd.coord[:, 2] >= bound[1][2]) & (
                pcd.coord[:, 2] <= (bound[1][2] + bound[2])
            )
            pcd_selected = pcd.select_by_index(selection)

            # Second filter: Radial (XY-plane) selection around base_point's XY coordinates
            translated_coord = (
                pcd_selected.coord - bound[1]
            )  # Center coordinates at base_point
            distances = np.linalg.norm(
                translated_coord[:, 0:2], axis=1
            )  # Calculate XY distances
            pcd_final = pcd_selected.select_by_index(distances <= bound[3])

            # Save filtered point cloud to corresponding output folder
            pcd_final.write_pcds_in_separate_files_ply(output_folder / bound[0])

        print("Finished pcd: " + file.stem)

    return


def find_offset_interferometer(
    file: Path,
    direction: float,
    z_bounds: list[float, float],
    dist_place: int = 1,
    format: str = "ply",
):
    interferometer_r = int(file.stem.split("_")[dist_place]) / 1000000
    # Load point cloud data
    if format == "asc":
        pcd = TLS_PCDs.load_single_pcd_ascii(
            file, header=["x", "y", "z", "Intensity", "Reflectance"], delimiter=","
        )
    elif format == "ply":
        pcd = TLS_PCDs.load_single_pcd_ply(file)

    pcd.estimate_pol_coords()

    filtered_pcd = pcd.filter_pcd(
        ["scalar_Pol_hz"], [(direction - 0.001, direction + 0.001)]
    )
    filtered_pcd = filtered_pcd.select_by_index(
        (filtered_pcd.coord[:, 2] >= z_bounds[0])
        & (filtered_pcd.coord[:, 2] < z_bounds[1])
    )
    filtered_pcd.scalar_fields["scalar_2D_range"] = np.sqrt(
        filtered_pcd.coord[:, 0] ** 2 + filtered_pcd.coord[:, 1] ** 2
    )

    range = np.median(filtered_pcd.scalar_fields["scalar_2D_range"])
    offset = range - interferometer_r

    return offset


def cut_comp_data(
    input_folder: Path,
    output_folder: Path,
    direction: float,
    z_bounds: list[float, float],
    target_info: list[tuple[str, str, float, float, float, float]],
    format: str = "ply",
    dist_place: int = 1,
    aoi_place: int = 2,
):
    path_pcds = list(input_folder.glob(f"*.{format}"))

    inter_offset = find_offset_interferometer(
        path_pcds[0], direction, z_bounds, dist_place, format
    )

    for file in path_pcds:
        inter_r = int(file.stem.split("_")[dist_place]) / 1000000
        inter_aoi = int(file.stem.split("_")[aoi_place]) / 10 / 180 * np.pi

        inter_distance = inter_r + inter_offset

        # Load point cloud data
        if format == "asc":
            pcd = TLS_PCDs.load_single_pcd_ascii(
                file, header=["x", "y", "z", "Intensity", "Reflectance"], delimiter=","
            )
        elif format == "ply":
            pcd = TLS_PCDs.load_single_pcd_ply(file)

        # pcd.scalar_fields["scalar_Intensity"] = pcd.scalar_fields[
        #     "scalar_Intensity"
        # ].astype(float)

        center_point = [
            inter_distance * np.cos(direction),
            inter_distance * np.sin(direction),
        ]

        # Rotate coordinates so that x axis is in direction
        cos_theta = np.cos(-direction)
        sin_theta = np.sin(-direction)
        R = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
        rotate_coord = pcd.coord @ R.T
        # Rotate coordinates so that x axis is perpendicular to surface normals
        trans_coord = rotate_coord.copy()
        trans_coord[:, 0:2] = trans_coord[:, 0:2] - ((center_point + [1]) @ R.T)[0:2]
        cos_theta = np.cos(-inter_aoi)
        sin_theta = np.sin(-inter_aoi)
        R = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
        final_coord = trans_coord @ R.T

        pre_filtered_pcd = pcd.select_by_index(
            (final_coord[:, 0] <= 0.03) & (final_coord[:, 0] >= -0.03)
        )

        for target in target_info:
            filtered_pcd = pre_filtered_pcd.select_by_index(
                (pre_filtered_pcd.coord[:, 2] >= target[4])
                & (pre_filtered_pcd.coord[:, 2] < target[5])
            )
            filtered_pcd.estimate_pol_coords()

            radius = np.linalg.norm(filtered_pcd.coord[:, 0:2] - center_point, axis=1)
            if len(radius) > 0:
                center_hz = filtered_pcd.scalar_fields["scalar_Pol_hz"][
                    np.argmin(radius)
                ]

                filtered_pcd = filtered_pcd.select_by_index(
                    (radius >= target[2]) & (radius <= target[3])
                )

                if target[1] == "l":
                    filtered_pcd = filtered_pcd.filter_pcd(
                        ["scalar_Pol_hz"], [(center_hz, np.inf)]
                    )
                elif target[1] == "r":
                    filtered_pcd = filtered_pcd.filter_pcd(
                        ["scalar_Pol_hz"], [(-np.inf, center_hz)]
                    )

                if len(filtered_pcd.coord) > 0:
                    filtered_pcd.write_pcds_in_separate_files_ply(
                        output_folder / target[0]
                    )

    print("Finished")
    return
