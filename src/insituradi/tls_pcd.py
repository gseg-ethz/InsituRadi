# tls_pcd.py
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Optional

import colorcet
import datashader as ds
import datashader.transfer_functions as tf
import laspy
import numpy as np
import pandas as pd
import pye57
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plyfile import PlyData, PlyElement
from scipy import spatial


@dataclass
class TLS_PCDs:
    """
    Class representing a collection of point clouds from terrestrial laser scanning (TLS).

    Attributes:
        coord (np.ndarray): Coordinates (x, y, z) of the points.
        normals (np.ndarray): Normal vectors (nx, ny, nz) of the points.
        colors (np.ndarray): RGB colors of the points.
        scalar_fields (dict[str, np.ndarray]): Additional scalar fields (e.g., intensity, reflectance).
        pcd_filenames (list[str]): Filenames of the loaded point cloud data.
        scanner_center (Optional[np.ndarray]): Scanner center positions.
    """

    coord: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    scalar_fields: dict[str, np.ndarray]
    pcd_filenames: list[str]
    scanner_center: Optional[np.ndarray] = None

    @classmethod
    def load_pcds_from_ply(cls, folder_path: Path):
        """
        Load point cloud data from .ply files.

        Args:
            folder_path (Path): Path to the folder containing .ply files.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing loaded point clouds.
        """
        print("Loading point clouds.")

        all_coord, all_normals, all_colors = [], [], []
        scalar_fields = {}

        # Get all .ply files in the folder
        path_pcds = list(folder_path.rglob("*.ply"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Iterate over each .ply file
        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))
            plydata = PlyData.read(path_pcds[i])

            # Extract coordinates (x, y, z)
            coord = np.empty(
                (
                    plydata["vertex"].count,
                    3,
                ),
                dtype=float,
            )
            coord[:, 0] = plydata["vertex"]["x"]
            coord[:, 1] = plydata["vertex"]["y"]
            coord[:, 2] = plydata["vertex"]["z"]
            all_coord.append(coord)
            del coord

            # Get scalar fields names (e.g., intensity, reflectance)
            scalar_fields_names = [pe.name for pe in plydata["vertex"].properties]

            # Initialize scalar fields on the first iteration
            if i == 0:
                if "scalar_PCD_ID" not in scalar_fields_names:
                    scalar_fields_names = scalar_fields_names + ["scalar_PCD_ID"]
                for sf in scalar_fields_names:
                    if sf not in [
                        "x",
                        "y",
                        "z",
                        "r",
                        "g",
                        "b",
                        "red",
                        "green",
                        "blue",
                        "nx",
                        "ny",
                        "nz",
                    ]:
                        scalar_fields[sf] = []

            # Extract normals if present
            if "nx" in scalar_fields_names:
                normals = np.empty(
                    (
                        plydata["vertex"].count,
                        3,
                    ),
                    dtype=float,
                )
                normals[:, 0] = plydata["vertex"]["nx"]
                normals[:, 1] = plydata["vertex"]["ny"]
                normals[:, 2] = plydata["vertex"]["nz"]
                all_normals.append(normals)
                del normals

            # Extract colors if present
            if ("red" in scalar_fields_names) or ("r" in scalar_fields_names):
                colors = np.empty(
                    (
                        plydata["vertex"].count,
                        3,
                    ),
                    dtype=np.uint8,
                )
                colors[:, 0] = (
                    plydata["vertex"]["r"]
                    if "r" in scalar_fields_names
                    else plydata["vertex"]["red"]
                )
                colors[:, 1] = (
                    plydata["vertex"]["g"]
                    if "g" in scalar_fields_names
                    else plydata["vertex"]["green"]
                )
                colors[:, 2] = (
                    plydata["vertex"]["b"]
                    if "b" in scalar_fields_names
                    else plydata["vertex"]["blue"]
                )
                all_colors.append(colors)
                del colors

            # Collect scalar fields data
            for sf in scalar_fields_names:
                if sf not in [
                    "x",
                    "y",
                    "z",
                    "r",
                    "g",
                    "b",
                    "red",
                    "green",
                    "blue",
                    "nx",
                    "ny",
                    "nz",
                    "scalar_PCD_ID",
                ]:
                    scalar_fields[sf].append(np.array(plydata["vertex"][sf]).squeeze())

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(
                np.ones((plydata["vertex"].count), dtype=np.uint8) * i
            )

            del plydata

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = (
                np.concatenate(scalar_fields[sf])
                if len(scalar_fields[sf]) > 0
                else None
            )

        print("Finished loading point clouds.")
        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=pcd_filenames,
        )

    @classmethod
    def load_pcds_from_e57(cls, folder_path: Path, registered: bool = True):
        """
        Load point cloud data from .ply files.

        Args:
            folder_path (Path): Path to the folder containing .ply files.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing loaded point clouds.
        """
        print("Loading point clouds.")

        all_coord, all_normals, all_colors = [], [], []
        scalar_fields = {}

        # Get all .e57 files in the folder
        path_pcds = list(folder_path.rglob("*.e57"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Iterate over each .e57 file
        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))

            # Read E57 file contents using pye57 library
            e57 = pye57.E57(str(path_pcds[i]))

            # Extract raw scan data with intensity values
            data = e57.read_scan(0, intensity=True)

            # Create coordinate matrix from Cartesian components (N x 3 array)
            coord = np.transpose(
                np.vstack(
                    (
                        np.array([data["cartesianX"]]),
                        np.array([data["cartesianY"]]),
                        np.array([data["cartesianZ"]]),
                    )
                )
            )

            # Get scanner transformation metadata from file header
            header = e57.get_header(0)
            if not registered:
                rot = header.rotation_matrix
                transl = header.translation
                transm = np.concatenate(
                    (
                        np.concatenate((rot, np.transpose(np.array([transl]))), axis=1),
                        np.array([[0, 0, 0, 1]]),
                    )
                )

            # Transform coordinates to target system using inverse scanner transform
            coord = np.matmul(
                np.linalg.inv(transm),
                np.concatenate((coord, np.ones((coord.shape[0], 1))), axis=1).T,
            )
            coord = coord[0:3, :].T

            all_coord.append(coord)
            del coord

            # Get scalar fields names (e.g., intensity, reflectance)
            scalar_fields_names = header.point_fields

            # Initialize scalar fields on the first iteration
            if i == 0:
                if "scalar_PCD_ID" not in scalar_fields_names:
                    scalar_fields_names = scalar_fields_names + ["scalar_PCD_ID"]
                for sf in scalar_fields_names:
                    if sf not in [
                        "cartesianX",
                        "cartesianY",
                        "cartesianZ",
                        "cartesianInvalidState",
                        "rowIndex",
                        "columnIndex",
                    ]:
                        if sf[0:7] != "scalar_":
                            sf = "scalar_" + sf
                        scalar_fields[sf] = []

            # Collect scalar fields data
            for sf in scalar_fields_names:
                if sf not in [
                    "scalar_PCD_ID",
                    "cartesianX",
                    "cartesianY",
                    "cartesianZ",
                    "cartesianInvalidState",
                    "rowIndex",
                    "columnIndex",
                ]:
                    # Prepare intensity data (convert to column vector and remove singleton dimension)
                    sf_data = np.transpose(np.array([data[sf]]))
                    if sf[0:7] != "scalar_":
                        sf = "scalar_" + sf
                    scalar_fields[sf].append(sf_data.squeeze())

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(
                np.ones((np.size(sf_data, 0)), dtype=np.uint8) * i
            )

            del data

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = (
                np.concatenate(scalar_fields[sf])
                if len(scalar_fields[sf]) > 0
                else None
            )

        print("Finished loading point clouds.")
        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=pcd_filenames,
        )

    @classmethod
    def load_pcds_from_las(cls, folder_path: Path):
        print("Loading point clouds.")

        all_coord, all_normals, all_colors = [], [], []
        scalar_fields = {}

        # Get all .ply files in the folder
        path_pcds = list(folder_path.rglob("*.las"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Iterate over each .ply file
        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))
            pcd_las = laspy.open(path_pcds[i]).read()

            # Extract coordinates (x, y, z)
            coord = pcd_las.xyz
            all_coord.append(coord)
            del coord

            # Get scalar fields names (e.g., intensity, reflectance)
            scalar_fields_names = list(pcd_las.point_format.dimension_names)
            for j in np.arange(len(scalar_fields_names)):
                if not scalar_fields_names[j].startswith("scalar_"):
                    if scalar_fields_names[j] not in [
                        "x",
                        "y",
                        "z",
                        "r",
                        "g",
                        "b",
                        "red",
                        "green",
                        "blue",
                        "nx",
                        "ny",
                        "nz",
                        "NormalX",
                        "NormalY",
                        "NormalZ",
                        "X",
                        "Y",
                        "Z",
                    ]:
                        scalar_fields_names[j] = "scalar_" + scalar_fields_names[j]

            # Initialize scalar fields on the first iteration
            if i == 0:
                if "scalar_PCD_ID" not in scalar_fields_names:
                    scalar_fields_names = scalar_fields_names + ["scalar_PCD_ID"]
                for sf in scalar_fields_names:
                    if sf not in [
                        "x",
                        "y",
                        "z",
                        "r",
                        "g",
                        "b",
                        "red",
                        "green",
                        "blue",
                        "nx",
                        "ny",
                        "nz",
                        "NormalX",
                        "NormalY",
                        "NormalZ",
                        "X",
                        "Y",
                        "Z",
                    ]:
                        scalar_fields[sf] = []

            # Extract normals if present
            if ("nx" in scalar_fields_names) or ("NormalX" in scalar_fields_names):
                normals = np.empty(
                    (
                        len(pcd_las.points),
                        3,
                    ),
                    dtype=float,
                )
                normals[:, 0] = (
                    pcd_las["nx"] if "nx" in scalar_fields_names else pcd_las["NormalX"]
                )
                normals[:, 1] = (
                    pcd_las["ny"] if "ny" in scalar_fields_names else pcd_las["NormalY"]
                )
                normals[:, 2] = (
                    pcd_las["nz"] if "nz" in scalar_fields_names else pcd_las["NormalZ"]
                )
                all_normals.append(normals)
                del normals

            # Extract colors if present
            if ("red" in scalar_fields_names) or ("r" in scalar_fields_names):
                colors = np.empty(
                    (
                        len(pcd_las.points),
                        3,
                    ),
                    dtype=np.uint8,
                )
                colors[:, 0] = (
                    pcd_las["r"] if "r" in scalar_fields_names else pcd_las["red"]
                )
                colors[:, 1] = (
                    pcd_las["g"] if "g" in scalar_fields_names else pcd_las["green"]
                )
                colors[:, 2] = (
                    pcd_las["b"] if "b" in scalar_fields_names else pcd_las["blue"]
                )
                all_colors.append(colors)
                del colors

            # Collect scalar fields data
            for sf in scalar_fields_names:
                if sf not in [
                    "x",
                    "y",
                    "z",
                    "r",
                    "g",
                    "b",
                    "red",
                    "green",
                    "blue",
                    "nx",
                    "ny",
                    "nz",
                    "scalar_PCD_ID",
                    "NormalX",
                    "NormalY",
                    "NormalZ",
                    "X",
                    "Y",
                    "Z",
                ]:
                    scalar_fields[sf].append(np.array(pcd_las[sf[7:]]).squeeze())

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(
                np.ones((len(pcd_las.points)), dtype=np.uint8) * i
            )

            del pcd_las

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = (
                np.concatenate(scalar_fields[sf])
                if len(scalar_fields[sf]) > 0
                else None
            )

        print("Finished loading point clouds.")
        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=pcd_filenames,
        )

    @classmethod
    def load_pcds_from_ascii(
        cls, folder_path: Path, header: list[str] = None, delimiter: str = " "
    ):
        """
        Load point cloud data from ASCII files.

        Args:
            folder_path (Path): Path to the folder containing ASCII files.
            header (list[str]): List of column names in the ASCII files.
            delimiter (str): Delimiter used in the ASCII files.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing loaded point clouds.
        """

        if header is None:
            header = [
                "x",
                "y",
                "z",
                "intensity",
                "red",
                "green",
                "blue",
                "reflectance",
                "name",
            ]
        print("Loading pcd from ascii files.")

        all_coord = []
        all_normals = []
        all_colors = []
        scalar_fields = {}

        # Get all .asc files in the folder
        path_pcds = list(folder_path.rglob("*.asc"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Capitalize header names to match the data
        header_capi = [head.capitalize() for head in header]

        # Initialize scalar fields
        for sf in header_capi:
            if sf not in [
                "X",
                "Y",
                "Z",
                "R",
                "G",
                "B",
                "Red",
                "Green",
                "Blue",
                "Nx",
                "Ny",
                "Nz",
                "Name",
            ]:
                scalar_fields["scalar_" + sf] = []
        scalar_fields["scalar_PCD_ID"] = []

        # Iterate over each .asc file
        for i in np.arange(len(path_pcds)):
            print("Started " + path_pcds[i].stem)
            ascii_data = pd.read_csv(
                path_pcds[i], delimiter=delimiter, names=header_capi, index_col=False
            )

            # Extract coordinates (x, y, z)
            coord = np.empty(
                (
                    len(ascii_data.X),
                    3,
                ),
                dtype=float,
            )
            coord[:, 0] = ascii_data.X
            coord[:, 1] = ascii_data.Y
            coord[:, 2] = ascii_data.Z
            all_coord.append(coord)
            del coord

            # Extract normals if present
            if "Nx" in header_capi:
                normals = np.empty(
                    (
                        len(ascii_data.X),
                        3,
                    ),
                    dtype=float,
                )
                normals[:, 0] = ascii_data.Nx
                normals[:, 1] = ascii_data.Ny
                normals[:, 2] = ascii_data.Nz
                all_normals.append(normals)
                del normals

            # Extract colors if present
            if ("Red" in header_capi) or ("R" in header_capi):
                colors = np.empty(
                    (
                        len(ascii_data.X),
                        3,
                    ),
                    dtype=np.uint8,
                )
                colors[:, 0] = ascii_data.R if "R" in header_capi else ascii_data.Red
                colors[:, 1] = ascii_data.G if "G" in header_capi else ascii_data.Green
                colors[:, 2] = ascii_data.B if "B" in header_capi else ascii_data.Blue
                all_colors.append(colors)
                del colors

            # Collect scalar fields data
            for sf in header_capi:
                if sf not in [
                    "X",
                    "Y",
                    "Z",
                    "R",
                    "G",
                    "B",
                    "Red",
                    "Green",
                    "Blue",
                    "Nx",
                    "Ny",
                    "Nz",
                    "Name",
                    "scalar_PCD_ID",
                ]:
                    scalar_fields["scalar_" + sf].append(
                        np.array(ascii_data[sf]).squeeze()
                    )

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(
                np.ones((len(ascii_data.X)), dtype=np.uint8) * i
            )

            del ascii_data

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = (
                np.concatenate(scalar_fields[sf])
                if len(scalar_fields[sf]) > 0
                else None
            )

        print("Finished loading point clouds.")
        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=pcd_filenames,
        )

    @classmethod
    def load_single_pcd_ascii(
        cls, filename: Path, header: list[str] = None, delimiter: str = " "
    ):
        if header is None:
            header = [
                "x",
                "y",
                "z",
                "intensity",
                "red",
                "green",
                "blue",
                "reflectance",
                "name",
            ]
        print("Loading pcd from ascii file: " + filename.stem)

        header_capi = [head.capitalize() for head in header]
        ascii_data = pd.read_csv(
            filename, delimiter=delimiter, names=header_capi, index_col=False
        )
        coord = np.empty(
            (
                len(ascii_data.X),
                3,
            ),
            dtype=float,
        )
        coord[:, 0] = ascii_data.X
        coord[:, 1] = ascii_data.Y
        coord[:, 2] = ascii_data.Z

        if "Nx" in header_capi:
            normals = np.empty(
                (
                    len(ascii_data.X),
                    3,
                ),
                dtype=float,
            )
            normals[:, 0] = ascii_data.Nx
            normals[:, 1] = ascii_data.Ny
            normals[:, 2] = ascii_data.Nz
        else:
            normals = None

        if ("Red" in header_capi) or ("R" in header_capi):
            colors = np.empty(
                (
                    len(ascii_data.X),
                    3,
                ),
                dtype=np.uint8,
            )
            colors[:, 0] = ascii_data.R if "R" in header_capi else ascii_data.Red
            colors[:, 1] = ascii_data.G if "G" in header_capi else ascii_data.Green
            colors[:, 2] = ascii_data.B if "B" in header_capi else ascii_data.Blue
        else:
            colors = None

        # Collect scalar fields data
        scalar_fields = {}
        for sf in header_capi:
            if sf not in [
                "X",
                "Y",
                "Z",
                "R",
                "G",
                "B",
                "Red",
                "Green",
                "Blue",
                "Nx",
                "Ny",
                "Nz",
                "Name",
                "scalar_PCD_ID",
            ]:
                scalar_fields["scalar_" + sf] = np.array(ascii_data[sf]).squeeze()
        scalar_fields["scalar_PCD_ID"] = np.zeros((len(ascii_data.X)), dtype=np.uint8)

        print("Finished loading point cloud.")
        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=[filename.stem],
        )

    @classmethod
    def load_single_pcd_ply(cls, path: Path):
        """
        Load a single point cloud from a .ply file.

        Args:
            path (Path): Path to the .ply file.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing the loaded point cloud.
        """
        print("Started loading pcd from ply: " + path.stem)
        plydata = PlyData.read(path)
        coord = np.empty(
            (
                plydata["vertex"].count,
                3,
            ),
            dtype=float,
        )
        coord[:, 0] = plydata["vertex"]["x"]
        coord[:, 1] = plydata["vertex"]["y"]
        coord[:, 2] = plydata["vertex"]["z"]

        # Get scalar fields names
        scalar_fields_names = [pe.name for pe in plydata["vertex"].properties]

        # Extract normals if present
        if "nx" in scalar_fields_names:
            normals = np.empty(
                (
                    plydata["vertex"].count,
                    3,
                ),
                dtype=float,
            )
            normals[:, 0] = plydata["vertex"]["nx"]
            normals[:, 1] = plydata["vertex"]["ny"]
            normals[:, 2] = plydata["vertex"]["nz"]
        else:
            normals = None

        # Extract colors if present
        if ("red" in scalar_fields_names) or ("r" in scalar_fields_names):
            colors = np.empty(
                (
                    plydata["vertex"].count,
                    3,
                ),
                dtype=np.uint8,
            )
            colors[:, 0] = (
                plydata["vertex"]["r"]
                if "r" in scalar_fields_names
                else plydata["vertex"]["red"]
            )
            colors[:, 1] = (
                plydata["vertex"]["g"]
                if "g" in scalar_fields_names
                else plydata["vertex"]["green"]
            )
            colors[:, 2] = (
                plydata["vertex"]["b"]
                if "b" in scalar_fields_names
                else plydata["vertex"]["blue"]
            )
        else:
            colors = None

        # Collect scalar fields data
        scalar_fields = {}
        for sf in scalar_fields_names:
            if sf not in [
                "x",
                "y",
                "z",
                "r",
                "g",
                "b",
                "red",
                "green",
                "blue",
                "nx",
                "ny",
                "nz",
                "scalar_PCD_ID",
            ]:
                scalar_fields[sf] = np.array(plydata["vertex"][sf]).squeeze()

        scalar_fields["scalar_PCD_ID"] = np.zeros((len(coord[:, 0])), dtype=np.uint8)

        return cls(
            coord=coord,
            normals=normals,
            colors=colors,
            scalar_fields=scalar_fields,
            pcd_filenames=[path.stem],
        )

    @classmethod
    def load_single_pcd_e57(cls, path: Path):
        """
        Load a single point cloud from a .e57 file.

        Args:
            path (Path): Path to the .e57 file.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing the loaded point cloud.
        """
        print("Started loading pcd from e57: " + path.stem)

        # Read E57 file contents using pye57 library
        e57 = pye57.E57(str(path))

        # Extract raw scan data with intensity values
        data = e57.read_scan(0, intensity=True)

        # Get scanner transformation metadata from file header
        header = e57.get_header(0)
        rot = header.rotation_matrix
        transl = header.translation
        transm = np.concatenate(
            (
                np.concatenate((rot, np.transpose(np.array([transl]))), axis=1),
                np.array([[0, 0, 0, 1]]),
            )
        )

        # Prepare intensity data (convert to column vector and remove singleton dimension)
        intensity = np.transpose(np.array([data["intensity"]]))
        scalar_fields = {"scalar_Intensity": intensity.squeeze()}

        # Create coordinate matrix from Cartesian components (N x 3 array)
        coord = np.transpose(
            np.vstack(
                (
                    np.array([data["cartesianX"]]),
                    np.array([data["cartesianY"]]),
                    np.array([data["cartesianZ"]]),
                )
            )
        )

        # Transform coordinates to target system using inverse scanner transform
        coord = np.matmul(
            np.linalg.inv(transm),
            np.concatenate((coord, np.ones((coord.shape[0], 1))), axis=1).T,
        )
        coord = coord[0:3, :].T

        return cls(
            coord=coord,
            normals=None,
            colors=None,
            scalar_fields=scalar_fields,
            pcd_filenames=[path.stem],
        )

    @classmethod
    def load_single_pcd_las(cls, path: Path):
        print("Started loading pcd from las: " + path.stem)
        # Read laspy
        pcd_las = laspy.open(path).read()

        # Prepare intensity data (convert to column vector and remove singleton dimension)
        intensity = pcd_las.intensity
        classification = np.asarray(pcd_las.classification)
        scalar_fields = {
            "scalar_Intensity": intensity.squeeze(),
            "scalar_Classification": classification.squeeze(),
        }

        # Create coordinate matrix from Cartesian components (N x 3 array)
        coord = pcd_las.xyz

        return cls(
            coord=coord,
            normals=None,
            colors=None,
            scalar_fields=scalar_fields,
            pcd_filenames=[path.stem],
        )

    def set_scanner_center_from_e57(self, folder_e57: Path):
        """
        Set the scanner center coordinates using E57 files.

        Args:
            folder_e57 (Path): Path to folder containing .e57 files.
        """
        print("Set scanner center.")
        scanner_centers = []
        for file in self.pcd_filenames:
            e57 = pye57.E57(str(folder_e57 / file) + ".e57")
            scanner_centers.append(e57.scan_position(0))

        self.scanner_center = np.concatenate(scanner_centers)
        print("Scanner center has been set.")
        return

    def estimate_range(self):
        """
        Calculate the range (distance) from the scanner center to each point.
        """
        print("Calculate ranges.")
        position_scanner = self.scanner_center[self.scalar_fields["scalar_PCD_ID"]]
        self.scalar_fields["scalar_Range"] = np.linalg.norm(
            self.coord - position_scanner, axis=1
        )
        print("Finished calculating ranges.")
        return

    def estimate_aoi(self):
        """
        Calculate the angle of incidence (AOI) for each point.
        """
        print("Calculate AOI.")
        incident_vector = (
            self.coord - self.scanner_center[self.scalar_fields["scalar_PCD_ID"]]
        )
        aoi_rad = np.arccos(
            np.sum(self.normals * incident_vector, axis=1)
            / (
                np.linalg.norm(self.normals, axis=1)
                * np.linalg.norm(incident_vector, axis=1)
            )
        )
        aoi_rad_toolarge = aoi_rad >= np.pi / 2
        aoi_rad[aoi_rad_toolarge] = np.pi - aoi_rad[aoi_rad_toolarge]
        self.scalar_fields["scalar_AOI"] = aoi_rad
        print("Finished calculating AOI.")
        return

    def estimate_pol_coords(self):
        self.scalar_fields["scalar_Pol_hz"] = np.arctan2(
            self.coord[:, 1], self.coord[:, 0]
        )
        self.scalar_fields["scalar_Pol_r"] = np.sqrt(
            self.coord[:, 0] ** 2 + self.coord[:, 1] ** 2 + self.coord[:, 2] ** 2
        )
        self.scalar_fields["scalar_Pol_vz"] = np.arcsin(
            self.coord[:, 2] / self.scalar_fields["scalar_Pol_r"]
        )
        return

    def find_patch_number(self, sparse_pcd, radius: float):
        """
        Find the patch number for each point based on a subsampled point cloud.

        Args:
            sparse_path (Path): Path to the folder containing the subsampled point cloud.
            radius (float): Radius to search for neighboring points (ideally half of the subsampling rate).
        """
        print("Find patch number.")
        tree = spatial.cKDTree(sparse_pcd.coord)
        neighbours = tree.query_ball_point(self.coord, radius, workers=-1, p=2)
        self.scalar_fields["scalar_Patch_number"] = np.asarray(
            [sublist[0] if sublist else np.nan for sublist in neighbours], dtype=float
        )
        print("Finished finding patch number.")
        return

    def find_patch_number_mem_opt(
        self,
        sparse_path: Path,
        pointcloud_path_input: Path,
        pointcloud_path_output: Path,
        radius: float,
    ):
        """
        Find the patch number for each point based on a subsampled point cloud.

        Args:
            sparse_path (Path): Path to the folder containing the subsampled point cloud.
            radius (float): Radius to search for neighboring points (ideally half of the subsampling rate).
        """
        print("Find patch number.")
        sparse_pcd = TLS_PCDs.load_pcds_from_ply(sparse_path)
        tree = spatial.cKDTree(sparse_pcd.coord)

        path_pcds = list(pointcloud_path_input.rglob("*.ply"))
        pcd_filenames = [path.stem for path in path_pcds]

        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))
            pcd = TLS_PCDs.load_single_pcd_ply(path_pcds[i])
            neighbours = tree.query_ball_point(pcd.coord, radius, workers=-1, p=2)
            pcd.scalar_fields["scalar_Patch_number"] = np.asarray(
                [sublist[0] if sublist else np.nan for sublist in neighbours],
                dtype=float,
            )
            pcd.write_pcds_in_separate_files_ply(pointcloud_path_output)

        print("Finished finding patch number.")
        return

    def count_unique_pcds_in_patch(self):
        """
        Count the number of unique point clouds in each patch.
        """
        print("Start counting unique pcds in patch.")
        df = pd.DataFrame(
            {
                "Neighbors": self.scalar_fields["scalar_Patch_number"],
                "PCD_ID": self.scalar_fields["scalar_PCD_ID"],
            }
        )
        unique_counts = df.groupby("Neighbors")["PCD_ID"].nunique()
        self.scalar_fields["scalar_Different_pcds_patch"] = (
            df["Neighbors"].map(unique_counts).to_numpy()
        )
        print("Finished counting unique pcds in patch.")
        return

    def classify_folder_pcd(
        self, input_folder: Path, radius: float, output_folder: Path
    ):
        print("Start classifying pcd with other pcd.")
        tree = spatial.cKDTree(self.coord)
        print("Tree calculated :)")

        path_pcds = list(input_folder.rglob("*.ply"))
        pcd_filenames = [path.stem for path in path_pcds]

        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))
            pcd = TLS_PCDs.load_single_pcd_ply(path_pcds[i])
            neighbours = tree.query_ball_point(pcd.coord, radius, workers=-1, p=2)
            neighbours_list = np.asarray(
                [sublist[0] if sublist else np.nan for sublist in neighbours],
                dtype=float,
            )
            has_neighbour = ~np.isnan(neighbours_list)

            classifications = np.full(len(pcd.coord), 9999, dtype=int)

            if np.any(has_neighbour):
                valid_indices = neighbours_list[has_neighbour].astype(int)
                classifications[has_neighbour] = self.scalar_fields[
                    "scalar_Classification"
                ][valid_indices]

            pcd.scalar_fields["scalar_Classification"] = classifications
            pcd.write_pcds_in_separate_files_ply(output_folder)

        print("Finished classifying point clouds.")
        return

    def transform(self, matrix: np.ndarray):
        hom_pcd = np.hstack((self.coord, np.ones((np.shape(self.coord)[0], 1))))
        hom_pcd_trans = hom_pcd @ matrix.T
        self.coord = hom_pcd_trans[:, 0:3]
        print("PCD transformed")
        return

    def fuse_modalities_to_ref_epoch(
        self,
        ref_epoch_name: str,
        max_distance: float,
        sf_mapped: str,
        pcd2modality: list[(str, str)],
    ):
        print("Start fusing modalities to reference epoch.")
        ref_epoch_indices = np.where(
            self.scalar_fields["scalar_PCD_ID"]
            == self.pcd_filenames.index(ref_epoch_name)
        )[0]
        tree = spatial.cKDTree(self.coord[ref_epoch_indices])
        # Boolean mask over all points, initially False. We'll mark True
        # for points belonging to each processed pcd_name.
        fused_mask = np.ones(self.coord.shape[0], dtype=bool)
        for pcd_name, modality_name in pcd2modality:
            print("Processing modality " + modality_name + " from PCD " + pcd_name)
            pcd_indices = np.where(
                self.scalar_fields["scalar_PCD_ID"]
                == self.pcd_filenames.index(pcd_name)
            )[0]
            tree = spatial.cKDTree(self.coord[pcd_indices])
            distances, indices = tree.query(
                self.coord[ref_epoch_indices],
                distance_upper_bound=max_distance,
                workers=-1,
                p=2,
            )
            # Initialize target modality field with NaNs (one value per point in the full cloud)
            self.scalar_fields[modality_name] = np.full(self.coord.shape[0], np.nan)

            # tree.query returns index == n and distance == inf when no neighbor is found within the bound.
            # Filter out those invalid matches to avoid out-of-bounds access.
            valid = np.isfinite(distances) & (indices < len(pcd_indices))
            if np.any(valid):
                target_idx = pcd_indices[indices[valid]]
                source_idx = ref_epoch_indices[valid]
                self.scalar_fields[modality_name][source_idx] = self.scalar_fields[
                    sf_mapped
                ][target_idx]

            # Mark points from the current PCD as True in the fused mask
            fused_mask[pcd_indices] = False

        pcd_new = self.select_by_index(fused_mask)
        return pcd_new

    def calculate_polarization(self, sf_0, sf_45, sf_90, sf_135, prename):
        print("Start calculating polarization features.")
        I0 = self.scalar_fields[sf_0]
        I45 = self.scalar_fields[sf_45]
        I90 = self.scalar_fields[sf_90]
        I135 = self.scalar_fields[sf_135]
        S0 = (I0 + I45 + I90 + I135) / 2
        S1 = I0 - I90
        S2 = I45 - I135

        DoLP = np.sqrt(S1**2 + S2**2) / S0
        AoLP = 0.5 * np.arctan2(S2, S1)

        self.scalar_fields[prename + "DoLP"] = DoLP
        self.scalar_fields[prename + "AoLP"] = AoLP

        I_pol = np.sqrt(S1**2 + S2**2)
        I_unpol = S0 - I_pol

        self.scalar_fields[prename + "Iunpol"] = I_unpol
        self.scalar_fields[prename + "Ipol"] = I_pol

        print("Finished calculating polarization features.")
        return

    def write_pcds_in_separate_files_ply(self, folder_path: Path):
        """
        Write point cloud data to separate .ply files.

        Args:
            folder_path (Path): Path to the folder where the .ply files will be saved.
        """
        print("Start writing pcd.")
        dtype_list = [("x", "float"), ("y", "float"), ("z", "float")]

        for key in self.scalar_fields.keys():
            if not key == "scalar_PCD_ID":
                self.scalar_fields[key] = self.scalar_fields[key].astype(float)
            else:
                self.scalar_fields[key] = self.scalar_fields[key].astype(int)

        # Add color fields to the dtype list if colors are present
        if self.colors is not None:
            dtype_list.extend(
                [
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
        # Add normal fields to the dtype list if normals are present
        if self.normals is not None:
            dtype_list.extend(
                [
                    ("nx", "f8"),
                    ("ny", "f8"),
                    ("nz", "f8"),
                ]
            )

        # Add scalar fields to the dtype list
        for sf in self.scalar_fields.keys():
            dtype_list.append((sf, self.scalar_fields[sf].dtype.str))

        # Check if multiple point clouds are present
        if "scalar_PCD_ID" in self.scalar_fields.keys():
            unique_pcd_id = np.unique(self.scalar_fields["scalar_PCD_ID"])
            multi_pcd = True
        else:
            multi_pcd = False
            unique_pcd_id = [0]

        # Write each point cloud to a separate .ply file
        for pcd_id in unique_pcd_id:
            # Select points belonging to the current point cloud (if multiple point clouds exist)
            if multi_pcd:
                selection = self.scalar_fields["scalar_PCD_ID"] == pcd_id
            else:
                selection = np.ones(np.shape(self.coord[:, 0]), dtype=bool)

            # Create a structured array to hold all point cloud data
            pcd_all_info = np.empty(
                (self.coord[selection, 0].shape[0],), dtype=dtype_list
            )

            # Populate the structured array with coordinates
            pcd_all_info["x"] = self.coord[selection, 0]
            pcd_all_info["y"] = self.coord[selection, 1]
            pcd_all_info["z"] = self.coord[selection, 2]

            # Populate the structured array with colors (if present)
            if self.colors is not None:
                pcd_all_info["red"] = self.colors[selection, 0]
                pcd_all_info["green"] = self.colors[selection, 1]
                pcd_all_info["blue"] = self.colors[selection, 2]

            # Populate the structured array with normals (if present)
            if self.normals is not None:
                pcd_all_info["nx"] = self.normals[selection, 0]
                pcd_all_info["ny"] = self.normals[selection, 1]
                pcd_all_info["nz"] = self.normals[selection, 2]

            # Populate the structured array with scalar fields
            for sf in self.scalar_fields.keys():
                data = self.scalar_fields[sf][selection]

                # Check if the data is a 64-bit integer and downcast to 32-bit integer
                if data.dtype == "int64" or data.dtype == "i8":
                    data = data.astype("int32")
                # Optional: PLY files also prefer float32 ('f4') over float64 ('f8')
                elif data.dtype == "float64" or data.dtype == "f8":
                    data = data.astype("float32")

                pcd_all_info[sf] = data

            # Create a PlyElement from the structured array
            el = PlyElement.describe(pcd_all_info, "vertex")

            # Ensure the output folder exists
            folder_path.mkdir(parents=True, exist_ok=True)

            # Write the PlyElement to a .ply file
            PlyData([el]).write(str(folder_path / self.pcd_filenames[pcd_id]) + ".ply")
            print("Finished writing pcd: " + str(self.pcd_filenames[pcd_id]))
        return

    def write_pcds_in_separate_files_las(self, folder_path: Path):

        print("Start writing pcd.")
        for key in self.scalar_fields.keys():
            if not key == "scalar_PCD_ID":
                self.scalar_fields[key] = self.scalar_fields[key].astype(float)
            else:
                self.scalar_fields[key] = self.scalar_fields[key].astype(int)

        # Check if multiple point clouds are present
        if "scalar_PCD_ID" in self.scalar_fields.keys():
            unique_pcd_id = np.unique(self.scalar_fields["scalar_PCD_ID"])
            multi_pcd = True
        else:
            multi_pcd = False
            unique_pcd_id = [0]

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(self.coord, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])
        for sf in self.scalar_fields.keys():
            header.add_extra_dim(laspy.ExtraBytesParams(name=sf, type=np.float32))
        if self.normals is not None:
            header.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
            header.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
            header.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))

        # Write each point cloud to a separate .ply file
        for pcd_id in unique_pcd_id:
            # Select points belonging to the current point cloud (if multiple point clouds exist)
            if multi_pcd:
                selection = self.scalar_fields["scalar_PCD_ID"] == pcd_id
            else:
                selection = np.ones(np.shape(self.coord[:, 0]), dtype=bool)

            las = laspy.LasData(header)

            # Populate the structured array with coordinates
            las.x = self.coord[selection, 0]
            las.y = self.coord[selection, 1]
            las.z = self.coord[selection, 2]

            # Populate the structured array with colors (if present)
            if self.colors is not None:
                las.red = self.colors[selection, 0]
                las.green = self.colors[selection, 1]
                las.blue = self.colors[selection, 2]

            # Populate the structured array with normals (if present)
            if self.normals is not None:
                las.nx = self.normals[selection, 0]
                las.ny = self.normals[selection, 1]
                las.nz = self.normals[selection, 2]

            # Populate the structured array with scalar fields
            for sf in self.scalar_fields.keys():
                las[sf] = self.scalar_fields[sf][selection]

            las.write(str(folder_path / self.pcd_filenames[pcd_id]) + ".las")
            print("Finished writing pcd: " + str(self.pcd_filenames[pcd_id]))

        print("Finished writing all pcds.")
        return

    def filter_pcd(
        self, sf_names: list[str] = [], sf_bounds: list[tuple[float, float]] = []
    ):
        """
        Filter points within a given range for specified scalar fields.

        Args:
            sf_names (list[str]): List of scalar fields that should be filtered.
            sf_bounds (list[tuple[float, float]]): Bounds for each scalar field.

        Returns:
            TLS_PCDs: Filtered point cloud data.
        """
        print("Start filtering pcd.")

        # Start with all points selected
        selection = np.ones(len(self.coord), dtype=bool)

        # Iteratively apply filters based on scalar fields and their bounds
        for i in np.arange(len(sf_names)):
            selection &= (self.scalar_fields[sf_names[i]] >= sf_bounds[i][0]) & (
                self.scalar_fields[sf_names[i]] <= sf_bounds[i][1]
            )
            print(sf_names[i])

        # Apply selection to all available scalar fields
        selected_scalar_fields = {
            sf: values[selection] for sf, values in self.scalar_fields.items()
        }

        # Create a new TLS_PCDs instance with the filtered data
        filtered_pcd = TLS_PCDs(
            coord=self.coord[selection, :],
            colors=self.colors[selection, :] if self.colors is not None else None,
            normals=self.normals[selection, :] if self.normals is not None else None,
            scalar_fields=selected_scalar_fields,
            pcd_filenames=self.pcd_filenames,
            scanner_center=self.scanner_center,
        )

        print("Finished filtering pcd.")
        return filtered_pcd

    def select_by_index(self, index: np.ndarray):
        """
        Create a subset of the point cloud by selecting points using a boolean/index mask array.

        Parameters:
            index (np.ndarray): Boolean mask or integer index array for point selection.
                                Shape should match the number of points (self.coord.shape[0]).

        Returns:
            TLS_PCDs: New point cloud instance containing only the selected points.
        """
        # Filter scalar fields using the index mask/indices
        selected_scalar_fields = {
            sf: values[index] for sf, values in self.scalar_fields.items()
        }
        # Create new point cloud instance with selected points
        selected_pcd = TLS_PCDs(
            coord=self.coord[index, :],
            colors=self.colors[index, :] if self.colors is not None else None,
            normals=self.normals[index, :] if self.normals is not None else None,
            scalar_fields=selected_scalar_fields,
            pcd_filenames=self.pcd_filenames,
            scanner_center=self.scanner_center,
        )
        return selected_pcd

    def subsample_patches(self):
        pcd_ids = np.unique(self.scalar_fields["scalar_PCD_ID"])

        coord = []
        scalar_fields = {k: [] for k in self.scalar_fields}
        for pcd_id in pcd_ids:
            mask = self.scalar_fields["scalar_PCD_ID"] == pcd_id
            bins = np.arange(
                np.min(self.scalar_fields["scalar_Patch_number"][mask]) - 0.5,
                np.max(self.scalar_fields["scalar_Patch_number"][mask]) + 0.6,
                1,
            )

            binned_coord = np.column_stack(
                (
                    scipy.stats.binned_statistic(
                        self.scalar_fields["scalar_Patch_number"][mask],
                        values=self.coord[mask, 0],
                        statistic="median",
                        bins=bins,
                    )[0],
                    scipy.stats.binned_statistic(
                        self.scalar_fields["scalar_Patch_number"][mask],
                        values=self.coord[mask, 1],
                        statistic="median",
                        bins=bins,
                    )[0],
                    scipy.stats.binned_statistic(
                        self.scalar_fields["scalar_Patch_number"][mask],
                        values=self.coord[mask, 2],
                        statistic="median",
                        bins=bins,
                    )[0],
                )
            )

            nan_val = ~np.isnan(binned_coord[:, 0])
            coord.append(binned_coord[nan_val, :])

            for k in self.scalar_fields.keys():
                scalar_fields[k].append(
                    scipy.stats.binned_statistic(
                        self.scalar_fields["scalar_Patch_number"][mask],
                        values=self.scalar_fields[k][mask],
                        statistic="median",
                        bins=bins,
                    )[0][nan_val]
                )

        coord = np.concatenate(coord)
        for k in scalar_fields:
            scalar_fields[k] = np.concatenate(scalar_fields[k])

        merged_pcd = TLS_PCDs(
            coord=coord,
            normals=None,
            colors=None,
            scalar_fields=scalar_fields,
            pcd_filenames=self.pcd_filenames,
            scanner_center=self.scanner_center,
        )
        return merged_pcd

    def delete_scalar_field(self, scalar_field):
        """
        Delete a scalar field from the point cloud data.

        Args:
            scalar_field (str): Name of the scalar field to delete.
        """
        if scalar_field in self.scalar_fields.keys():
            del self.scalar_fields[scalar_field]
        return

    def evaluation_rsme(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]
        rmse = np.sqrt(1 / len(resid) * np.sum(resid**2))
        print("RSME: " + str(rmse))
        return rmse

    def evaluation_mae(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]
        rmse = 1 / len(resid) * np.sum(np.abs(resid))
        print("MAE: " + str(rmse))
        return rmse

    def evaluation_mape(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
    ):
        # Avoid division by zero
        mask = self.scalar_fields[sf_intensity] != 0
        mape = (
            np.mean(
                np.abs(
                    (
                        self.scalar_fields[est_intensity][mask]
                        - self.scalar_fields[sf_intensity][mask]
                    )
                    / self.scalar_fields[sf_intensity][mask]
                )
            )
            * 100
        )

        print("MAPE: " + str(mape) + "%")
        return mape

    def evaluation_cv_error(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]
        cv_error = np.std(resid) / np.mean(self.scalar_fields[sf_intensity])
        print("CV error: " + str(cv_error))
        return cv_error

    def evaluation_r2(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]
        r2 = 1 - (np.sum(resid**2)) / (
            np.sum(
                (
                    self.scalar_fields[sf_intensity]
                    - np.mean(self.scalar_fields[sf_intensity])
                )
                ** 2
            )
        )
        print("R2: " + str(r2))
        return r2

    def evaluation_vis_res3d(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
        path_plot: str = None,
        model_name: str = None,
        range_bin: float = 1,
        aoi_bin: float = 0.01,
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]
        df = pd.DataFrame(
            {
                "distance": self.scalar_fields["scalar_Range"],
                "aoi": self.scalar_fields["scalar_AOI"],
                "residual": resid,
            }
        )

        # Create Canvas — resolution can be adjusted
        canvas = ds.Canvas(
            plot_width=int((df["distance"].max() - df["distance"].min()) / range_bin),
            plot_height=int((df["aoi"].max() - df["aoi"].min()) / aoi_bin),
            x_range=(df["distance"].min(), df["distance"].max()),
            y_range=(df["aoi"].min(), df["aoi"].max()),
        )

        # Aggregate the data: mean of residuals per pixel
        agg = canvas.points(df, "distance", "aoi", ds.mean("residual"))

        # Clip to remove extreme outliers for better colormap contrast
        vmin, vmax = np.nanpercentile(agg.data, [1, 99])
        if np.abs(vmin) > vmax:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax
        clipped = agg.where((agg >= vmin) & (agg <= vmax))

        # Turn aggregation into an image
        img = tf.shade(clipped, cmap=colorcet.coolwarm, how="linear", span=[vmin, vmax])
        img_np = np.array(img.to_pil())

        # Display it using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(
            img_np,
            origin="upper",
            aspect="auto",
            extent=[
                df["distance"].min(),
                df["distance"].max(),
                df["aoi"].min(),
                df["aoi"].max(),
            ],
        )
        plt.xlabel("Distance [m]")
        plt.ylabel("AOI [rad]")
        plt.title("Heatmap of Residuals (Distance vs AOI)")

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="coolwarm")
        sm.set_array([])
        cbar = plt.colorbar(sm, label="Residual (mean per pixel)")
        plt.tight_layout()
        if path_plot is not None:
            path_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                str(path_plot / ("Heatmap_Residuals" + str(model_name) + ".png")),
                dpi=300,
            )
        plt.show()
        print("wuhu")
        return

    def evaluation_vis_res2d(
        self,
        est_intensity: str = "scalar_Estimated_intensity",
        sf_intensity: str = "scalar_Intensity",
        sf_x_axis: str = "scalar_AOI",
        path_plot: str = None,
        bin_size: float = 0.1,
        model_name: str = None,
    ):
        resid = self.scalar_fields[est_intensity] - self.scalar_fields[sf_intensity]

        bin_edges = np.arange(
            np.min(self.scalar_fields[sf_x_axis]),
            np.max(self.scalar_fields[sf_x_axis]),
            bin_size,
        )
        bin_labels = [
            f"{(bin_edges[i] + bin_edges[i + 1]) / 2:.2f}"
            for i in range(len(bin_edges) - 1)
        ]
        binned_data = pd.cut(
            self.scalar_fields[sf_x_axis],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            ordered=False,
        )
        df = pd.DataFrame({"Distance Bin": binned_data, "residuals": resid})

        boxplot_stats = (
            df.groupby("Distance Bin")["residuals"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .unstack()
        )
        boxplot_stats.columns = ["min", "q1", "median", "q3", "max"]

        bin_centers = (
            np.asarray(boxplot_stats.index.categories, dtype=float) + bin_size / 2
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bxp(
            [
                {
                    "med": row["median"],
                    "q1": row["q1"],
                    "q3": row["q3"],
                    "whislo": row["min"],
                    "whishi": row["max"],
                    "fliers": [],
                }
                for _, row in boxplot_stats.iterrows()
            ],
            positions=bin_centers,
            widths=bin_size * 0.8,
            showfliers=False,
        )

        # Add title and labels
        plt.title("Intensity Residual Distribution")
        plt.xlabel(sf_x_axis)
        plt.xlim((np.min(bin_edges), np.max(bin_edges) + bin_size / 2))
        plt.ylabel("Intensity Residuals")
        ax.set_xticks(bin_centers[::1])
        ax.set_xticklabels([f"{x:.1f}" for x in bin_centers[::1]])

        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        if path_plot is not None:
            path_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                str(
                    path_plot
                    / ("Residuals_" + sf_x_axis + "_" + str(model_name) + ".png")
                ),
                dpi=300,
            )
        plt.show()
        return

    def evaluation_vis_dist_AOI_Range(
        self, path_plot: str = None, range_bin: float = 1, aoi_bin: float = 0.01
    ):
        # Create a DataFrame
        df = pd.DataFrame(
            {
                "Range": self.scalar_fields["scalar_Range"],
                "AOI": self.scalar_fields["scalar_AOI"],
            }
        )

        # Create Canvas — resolution can be adjusted
        canvas = ds.Canvas(
            plot_width=int((df["Range"].max() - df["Range"].min()) / range_bin),
            plot_height=int((df["AOI"].max() - df["AOI"].min()) / aoi_bin),
            x_range=(df["Range"].min(), df["Range"].max()),
            y_range=(df["AOI"].min(), df["AOI"].max()),
        )

        # Aggregate the data: mean of residuals per pixel
        agg = canvas.points(df, "Range", "AOI")

        # Clip to remove extreme outliers for better colormap contrast
        vmin, vmax = np.nanpercentile(agg.data, [1, 99])
        clipped = agg.where((agg >= vmin) & (agg <= vmax))

        # Turn aggregation into an image
        img = tf.shade(clipped, cmap=colorcet.fire, how="linear", span=[vmin, vmax])
        img_np = np.array(img.to_pil())

        # Display it using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(
            img_np,
            origin="upper",
            aspect="auto",
            extent=[
                df["Range"].min(),
                df["Range"].max(),
                df["AOI"].min(),
                df["AOI"].max(),
            ],
        )
        plt.xlabel("Range [m]")
        plt.ylabel("AOI [rad]")
        plt.title("Heatmap of Number of Points (Range vs AOI)")

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="cet_CET_L3")
        sm.set_array([])
        cbar = plt.colorbar(sm, label="Number of Points")
        plt.tight_layout()
        if path_plot is not None:
            path_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path_plot / "AOI_Range_Dist.png"), dpi=300)
        plt.show()
        return

    def evaluation_vis_correlation(self, path_plot: str = None):
        df = pd.DataFrame(self.scalar_fields)
        # Compute per (PCD_ID, Patch_number) means:
        patch_stats = df.groupby(
            ["scalar_PCD_ID", "scalar_Patch_number"], as_index=False
        ).agg(
            scalar_Comp_intensity=("scalar_Comp_intensity", "mean"),
            scalar_Intensity=("scalar_Intensity", "mean"),
            scalar_AOI=("scalar_AOI", "mean"),
            scalar_Range=("scalar_Range", "mean"),
        )
        variables = [
            "scalar_Comp_intensity",
            "scalar_Intensity",
            "scalar_AOI",
            "scalar_Range",
        ]
        series_dicts = {
            var: {
                pcd: grp.set_index("scalar_Patch_number")[var]
                for pcd, grp in patch_stats.groupby("scalar_PCD_ID")
            }
            for var in variables
        }

        df_comp = patch_stats.pivot(
            index="scalar_Patch_number",
            columns="scalar_PCD_ID",
            values="scalar_Comp_intensity",
        )
        df_raw = patch_stats.pivot(
            index="scalar_Patch_number",
            columns="scalar_PCD_ID",
            values="scalar_Intensity",
        )
        df_aoi = patch_stats.pivot(
            index="scalar_Patch_number", columns="scalar_PCD_ID", values="scalar_AOI"
        )
        df_rng = patch_stats.pivot(
            index="scalar_Patch_number", columns="scalar_PCD_ID", values="scalar_Range"
        )

        clouds = df_comp.columns.tolist()

        def plot_heatmap(mat, title, path_plot, vmin=None, vmax=None):
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                mat,
                cmap="viridis",
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar_kws=dict(shrink=0.8),
            )
            plt.title(title)
            plt.xlabel("PCD_ID")
            plt.ylabel("PCD_ID")
            plt.tight_layout()
            plt.savefig(path_plot, dpi=300)
            plt.show()

        corr_comp = df_comp.corr(method="pearson")
        plot_heatmap(
            corr_comp,
            "Pearson corr — Compensated Intensity",
            path_plot / "Corr_comp_intensity.png",
        )

        corr_raw = df_raw.corr(method="pearson")
        plot_heatmap(
            corr_raw,
            "Pearson corr — Raw Intensity",
            path_plot / "Corr_raw_intensity.png",
        )

        corr_aoi = df_aoi.corr(method="pearson")
        plot_heatmap(corr_aoi, "Pearson corr — AOI", path_plot / "Corr_aoi.png")

        corr_rng = df_rng.corr(method="pearson")
        plot_heatmap(corr_rng, "Pearson corr — Range", path_plot / "Corr_range.png")

        presence = (~df_comp.isna()).astype(int)
        count_mat = presence.T.dot(presence)
        plot_heatmap(count_mat, "# Common patches", path_plot / "Common_patches.png")

        def make_spread_matrix(pivot_df, clouds):
            # spreads[i,j] = 99%ile(avg_means) - 1%ile(avg_means)
            spreads = pd.DataFrame(np.nan, index=clouds, columns=clouds)
            for i, j in combinations_with_replacement(clouds, 2):
                # average the two clouds' per-patch means
                avg = (pivot_df[i] + pivot_df[j]) / 2
                avg = avg.dropna()
                if len(avg) >= 2:
                    q99 = np.nanpercentile(avg, 99)
                    q01 = np.nanpercentile(avg, 1)
                    val = q99 - q01
                else:
                    val = np.nan
                spreads.at[i, j] = spreads.at[j, i] = val
            return spreads

        spread_aoi = make_spread_matrix(df_aoi, clouds)
        plot_heatmap(
            spread_aoi, "6) Spread of mean AOI (99% – 1%)", path_plot / "AOI_spread.png"
        )

        spread_rng = make_spread_matrix(df_rng, clouds)
        plot_heatmap(
            spread_rng,
            "7) Spread of mean Range (99% – 1%)",
            path_plot / "Range_spread.png",
        )
        return
