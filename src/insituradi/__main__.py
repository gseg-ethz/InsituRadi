import argparse
from pathlib import Path

import numpy as np
from insituradi.core import preprocessing_radiometric_compensation
from insituradi.radiometric_compensation import RadiometricCompensation
from insituradi.radiometric_models import SmoothingSplines, AdaptedLambertianModel
from insituradi.tls_pcd import TLS_PCDs

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project_path', type=Path,
                    help='Path to the project folder.', required=True)
args = parser.parse_args()

# Define the path to the project directory
project_path = args.project_path

# Preprocess radiometric compensation: filters and prepares input data for calibration
preprocessing_radiometric_compensation(
    project_path / "input_files",                   # Path to the input point cloud data
    project_path / "intermediate_steps",            # Path to store intermediate processing results
    fileformat_input="e57",                         # Input file format of point clouds
    input_folder_e57=project_path / "input_files",  # Path to e57 data to extract the scanner position
    radius=0.05,                                    # Search radius for neighbors, surface variation, normals
    range_max=100,                                  # Maximum range to process (typical max. range of laser scanner)
    intensity_max=5.2 * (10 ** 6),                  # Maximum intensity value
    surf_var_max=0.005,                             # Maximum surface variance
    aoi_max=np.pi / 2,                              # Maximum angle of incidence (AOI)
    pcds_number_patch_min=3                         # Minimum number of different scans in patch
)

# Load the filtered point cloud data from preprocessing step
filtered_pcd = TLS_PCDs.load_pcds_from_ply(project_path / "intermediate_steps" / "filtered_pcds")

# Initialize radiometric compensation with specific models for Angle of Incidence (AOI) and Range
radiometric_compensation = RadiometricCompensation(
    AOI_model=AdaptedLambertianModel(norm0=0.3),    # AOI model using Adapted Lambertian Model
    Range_model=SmoothingSplines(norm0=12.5)        # Range model using Smoothing Splines
)

# Cycle 1 & 2: Estimate reflectance normalization value, range and AOI compensation function
radiometric_compensation.estimate_refl_range_aoi_iterative(
    filtered_pcd,
    show_plot=False,
    path_plot=project_path / "results" / "AL+SS"
)

# Switch AOI model to Smoothing Splines and re-run estimation
radiometric_compensation.AOI_model = SmoothingSplines(norm0=0.3)
radiometric_compensation.estimate_refl_range_aoi_iterative(
    filtered_pcd,
    show_plot=False,
    path_plot=project_path / "results" / "AL+SS"
)

# Save the trained radiometric model to a file for later use
radiometric_compensation.save_to_file(project_path / "results" / "AL+SS" / "Radiometric_Model.joblib")

print("Radiometric Calibration done :)")


# Dataprocessing: Applying radiometric calibration to point clouds
# Load the pre-trained radiometric model
radiometric_compensation = RadiometricCompensation.load_from_file(
    project_path / "results" / "AL+SS" / "Radiometric_Model.joblib"
)

# Apply radiometric compensation to a folder of point clouds (batch processing)
radiometric_compensation.compensate_pcd_folder(
    project_path / "intermediate_steps" / "information_pcds",  # Input folder with point clouds
    project_path / "results" / "AL+SS",  # Output folder for compensated data
    input_format="ply",  # Input file format
    sf_intensity="scalar_Intensity",  # Scalar field for intensity (source)
    sf_comp_intensity="scalar_Comp_intensity"  # Scalar field for compensated intensity (target)
)
