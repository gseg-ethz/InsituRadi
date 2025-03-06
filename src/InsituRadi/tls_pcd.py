# tls_pcd.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pye57
from plyfile import PlyData, PlyElement
from pathlib import Path
from typing import Optional
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
        path_pcds = list(folder_path.glob("*.ply"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Iterate over each .ply file
        for i in np.arange(len(path_pcds)):
            print("Started " + str(pcd_filenames[i]))
            plydata = PlyData.read(path_pcds[i])

            # Extract coordinates (x, y, z)
            coord = np.empty((plydata['vertex'].count, 3,), dtype=float)
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
                    if sf not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz"]:
                        scalar_fields[sf] = []

            # Extract normals if present
            if 'nx' in scalar_fields_names:
                normals = np.empty((plydata['vertex'].count, 3,), dtype=float)
                normals[:, 0] = plydata["vertex"]["nx"]
                normals[:, 1] = plydata["vertex"]["ny"]
                normals[:, 2] = plydata["vertex"]["nz"]
                all_normals.append(normals)
                del normals

            # Extract colors if present
            if ('red' in scalar_fields_names) or ('r' in scalar_fields_names):
                colors = np.empty((plydata['vertex'].count, 3,), dtype=np.uint8)
                colors[:, 0] = plydata["vertex"]["r"] if "r" in scalar_fields_names else plydata["vertex"]["red"]
                colors[:, 1] = plydata["vertex"]["g"] if "g" in scalar_fields_names else plydata["vertex"]["green"]
                colors[:, 2] = plydata["vertex"]["b"] if "b" in scalar_fields_names else plydata["vertex"]["blue"]
                all_colors.append(colors)
                del colors

            # Collect scalar fields data
            for sf in scalar_fields_names:
                if (sf not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz",
                               "scalar_PCD_ID"]):
                    scalar_fields[sf].append(np.array(plydata["vertex"][sf]).squeeze())

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(np.ones((plydata['vertex'].count), dtype=np.uint8) * i)

            del plydata

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = np.concatenate(scalar_fields[sf]) if len(scalar_fields[sf]) > 0 else None

        print("Finished loading point clouds.")
        return cls(coord=coord, normals=normals, colors=colors, scalar_fields=scalar_fields,
                   pcd_filenames=pcd_filenames)

    @classmethod
    def load_pcds_from_ascii(cls, folder_path: Path, header: list[str] = None, delimiter: str = " "):
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
            header = ["x", "y", "z", "intensity", "red", "green", "blue", "reflectance", "name"]
        print("Loading pcd from ascii files.")

        all_coord = []
        all_normals = []
        all_colors = []
        scalar_fields = {}

        # Get all .asc files in the folder
        path_pcds = list(folder_path.glob("*.asc"))
        pcd_filenames = [path.stem for path in path_pcds]

        # Capitalize header names to match the data
        header_capi = [head.capitalize() for head in header]

        # Initialize scalar fields
        for sf in header_capi:
            if sf not in ["X", "Y", "Z", "R", "G", "B", "Red", "Green", "Blue", "Nx", "Ny", "Nz", "Name"]:
                scalar_fields["scalar_" + sf] = []
        scalar_fields["scalar_PCD_ID"] = []

        # Iterate over each .asc file
        for i in np.arange(len(path_pcds)):
            print("Started " + path_pcds[i].stem)
            ascii_data = pd.read_csv(path_pcds[i], delimiter=delimiter,
                                     names=header_capi, index_col=False)

            # Extract coordinates (x, y, z)
            coord = np.empty((len(ascii_data.X), 3,), dtype=float)
            coord[:, 0] = ascii_data.X
            coord[:, 1] = ascii_data.Y
            coord[:, 2] = ascii_data.Z
            all_coord.append(coord)
            del coord

            # Extract normals if present
            if 'Nx' in header_capi:
                normals = np.empty((len(ascii_data.X), 3,), dtype=float)
                normals[:, 0] = ascii_data.Nx
                normals[:, 1] = ascii_data.Ny
                normals[:, 2] = ascii_data.Nz
                all_normals.append(normals)
                del normals

            # Extract colors if present
            if ('Red' in header_capi) or ('R' in header_capi):
                colors = np.empty((len(ascii_data.X), 3,), dtype=np.uint8)
                colors[:, 0] = ascii_data.R if "R" in header_capi else ascii_data.Red
                colors[:, 1] = ascii_data.G if "G" in header_capi else ascii_data.Green
                colors[:, 2] = ascii_data.B if "B" in header_capi else ascii_data.Blue
                all_colors.append(colors)
                del colors

            # Collect scalar fields data
            for sf in header_capi:
                if (sf not in ["X", "Y", "Z", "R", "G", "B", "Red", "Green", "Blue", "Nx", "Ny", "Nz", "Name",
                               "scalar_PCD_ID"]):
                    scalar_fields["scalar_" + sf].append(np.array(ascii_data[sf]).squeeze())

            # Add point cloud ID
            scalar_fields["scalar_PCD_ID"].append(np.ones((len(ascii_data.X)), dtype=np.uint8) * i)

            del ascii_data

        # Concatenate all collected data
        coord = np.concatenate(all_coord) if len(all_coord) > 0 else None
        normals = np.concatenate(all_normals) if len(all_normals) > 0 else None
        colors = np.concatenate(all_colors) if len(all_colors) > 0 else None
        for sf in scalar_fields.keys():
            scalar_fields[sf] = np.concatenate(scalar_fields[sf]) if len(scalar_fields[sf]) > 0 else None

        print("Finished loading point clouds.")
        return cls(coord=coord, normals=normals, colors=colors, scalar_fields=scalar_fields,
                   pcd_filenames=pcd_filenames)

    @classmethod
    def load_single_pcd_ply(cls, path: Path):
        """
        Load a single point cloud from a .ply file.

        Args:
            path (Path): Path to the .ply file.

        Returns:
            TLS_PCDs: Instance of TLS_PCDs containing the loaded point cloud.
        """
        print("Started loading pcd from ply.")
        plydata = PlyData.read(path)
        coord = np.empty((plydata['vertex'].count, 3,), dtype=float)
        coord[:, 0] = plydata["vertex"]["x"]
        coord[:, 1] = plydata["vertex"]["y"]
        coord[:, 2] = plydata["vertex"]["z"]

        # Get scalar fields names
        scalar_fields_names = [pe.name for pe in plydata["vertex"].properties]

        # Extract normals if present
        if 'nx' in scalar_fields_names:
            normals = np.empty((plydata['vertex'].count, 3,), dtype=float)
            normals[:, 0] = plydata["vertex"]["nx"]
            normals[:, 1] = plydata["vertex"]["ny"]
            normals[:, 2] = plydata["vertex"]["nz"]

        # Extract colors if present
        if ('red' in scalar_fields_names) or ('r' in scalar_fields_names):
            colors = np.empty((plydata['vertex'].count, 3,), dtype=np.uint8)
            colors[:, 0] = plydata["vertex"]["r"] if "r" in scalar_fields_names else plydata["vertex"]["red"]
            colors[:, 1] = plydata["vertex"]["g"] if "g" in scalar_fields_names else plydata["vertex"]["green"]
            colors[:, 2] = plydata["vertex"]["b"] if "b" in scalar_fields_names else plydata["vertex"]["blue"]

        # Collect scalar fields data
        scalar_fields = {}
        for sf in scalar_fields_names:
            if (sf not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz",
                           "scalar_PCD_ID"]):
                scalar_fields[sf] = np.array(plydata["vertex"][sf]).squeeze()

        return cls(coord=coord, normals=normals, colors=colors, scalar_fields=scalar_fields, pcd_filenames=[path.stem])

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
        self.scalar_fields["scalar_Range"] = np.linalg.norm(self.coord - position_scanner, axis=1)
        print("Finished calculating ranges.")
        return

    def estimate_aoi(self):
        """
        Calculate the angle of incidence (AOI) for each point.
        """
        print("Calculate AOI.")
        incident_vector = self.coord - self.scanner_center[self.scalar_fields["scalar_PCD_ID"]]
        aoi_rad = np.arccos(np.sum(self.normals * incident_vector, axis=1) / (
                np.linalg.norm(self.normals, axis=1) * np.linalg.norm(incident_vector, axis=1)))
        aoi_rad_toolarge = aoi_rad >= np.pi / 2
        aoi_rad[aoi_rad_toolarge] = np.pi - aoi_rad[aoi_rad_toolarge]
        self.scalar_fields["scalar_AOI"] = aoi_rad
        print("Finished calculating AOI.")
        return

    def find_patch_number(self, sparse_path: Path, radius: float):
        """
        Find the patch number for each point based on a subsampled point cloud.

        Args:
            sparse_path (Path): Path to the folder containing the subsampled point cloud.
            radius (float): Radius to search for neighboring points (ideally half of the subsampling rate).
        """
        print("Find patch number.")
        sparse_pcd = TLS_PCDs.load_pcds_from_ply(sparse_path)
        tree = spatial.cKDTree(sparse_pcd.coord)
        neighbours = tree.query_ball_point(self.coord, radius, workers=-1, p=2)
        self.scalar_fields["scalar_Patch_number"] = np.asarray(
            [sublist[0] if sublist else np.nan for sublist in neighbours], dtype=float)
        print("Finished finding patch number.")
        return

    def count_unique_pcds_in_patch(self):
        """
        Count the number of unique point clouds in each patch.
        """
        print("Start counting unique pcds in patch.")
        df = pd.DataFrame(
            {"Neighbors": self.scalar_fields["scalar_Patch_number"], "PCD_ID": self.scalar_fields["scalar_PCD_ID"]})
        unique_counts = df.groupby('Neighbors')['PCD_ID'].nunique()
        self.scalar_fields["scalar_Different_pcds_patch"] = df['Neighbors'].map(unique_counts).to_numpy()
        print("Finished counting unique pcds in patch.")
        return

    def write_pcds_in_separate_files_ply(self, folder_path: Path):
        """
        Write point cloud data to separate .ply files.

        Args:
            folder_path (Path): Path to the folder where the .ply files will be saved.
        """
        print("Start writing pcd.")
        dtype_list = [('x', 'float'), ('y', 'float'), ('z', 'float')]

        # Add color fields to the dtype list if colors are present
        if self.colors is not None:
            dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1"), ])
        # Add normal fields to the dtype list if normals are present
        if self.normals is not None:
            dtype_list.extend([("nx", "f8"), ("ny", "f8"), ("nz", "f8"), ])

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
            pcd_all_info = np.empty((self.coord[selection, 0].shape[0],), dtype=dtype_list)

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
                pcd_all_info[sf] = self.scalar_fields[sf][selection]

            # Create a PlyElement from the structured array
            el = PlyElement.describe(pcd_all_info, "vertex")

            # Ensure the output folder exists
            folder_path.mkdir(parents=True, exist_ok=True)

            # Write the PlyElement to a .ply file
            PlyData([el]).write(str(folder_path / self.pcd_filenames[pcd_id]) + ".ply")
            print("Finished writing pcd: " + self.pcd_filenames[pcd_id])
        return

    def filter_pcd(self, sf_names: list[str] = [],
                   sf_bounds: list[tuple[float, float]] = []):
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
                    self.scalar_fields[sf_names[i]] <= sf_bounds[i][1])
            print(sf_names[i])

        # Apply selection to all available scalar fields
        selected_scalar_fields = {sf: values[selection] for sf, values in self.scalar_fields.items()}

        # Create a new TLS_PCDs instance with the filtered data
        filtered_pcd = TLS_PCDs(coord=self.coord[selection, :],
                                colors=self.colors[selection, :] if self.colors is not None else None,
                                normals=self.normals[selection, :] if self.normals is not None else None,
                                scalar_fields=selected_scalar_fields, pcd_filenames=self.pcd_filenames,
                                scanner_center=self.scanner_center)

        print("Finished filtering pcd.")
        return filtered_pcd

    def delete_scalar_field(self, scalar_field):
        """
        Delete a scalar field from the point cloud data.

        Args:
            scalar_field (str): Name of the scalar field to delete.
        """
        if scalar_field in self.scalar_fields.keys():
            del self.scalar_fields[scalar_field]
        return
