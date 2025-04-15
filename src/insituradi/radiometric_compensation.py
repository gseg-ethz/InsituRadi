import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from insituradi.tls_pcd import TLS_PCDs
from insituradi.radiometric_data import RadiometricData
from joblib import dump, load
from matplotlib import pyplot as plt


class RadiometricCompensation:
    """
    Class for performing radiometric compensation on point cloud data.
    This includes correcting for Angle of Incidence (AOI), range, and reflectance normalization values.
    """

    def __init__(self, AOI_model=None, Range_model=None):
        """
        Initialize the RadiometricCompensation class.

        Args:
            AOI_model: Model for Angle of Incidence (AOI) compensation.
            Range_model: Model for range compensation.
        """
        self.AOI_model = AOI_model
        self.Range_model = Range_model
        self.

    @classmethod
    def load_from_file(cls, path: Path):
        """
        Load a pre-trained RadiometricCompensation instance from a file.

        Args:
            path (Path): Path to the file containing the saved instance.

        Returns:
            RadiometricCompensation: Loaded instance.
        """
        return load(path)

    def estimate_AOI_compensation_function(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity",
                                           path_plot: Path = None, show_plot: bool = False):
        """
        Estimate the Angle of Incidence (AOI) compensation function.

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.
            path_plot (Path): Path to save the plot (optional).
            show_plot (bool): Whether to display the plot.
        """
        print("Start AOI compensation function estimation.")
        # Bin the data based on AOI values
        binned_data = RadiometricData.from_tls_pcd_and_binned(pcd, int(np.round(np.pi / 2 * 1000)), "scalar_AOI",
                                                              sf_intensity)
        # Fit the AOI model to the binned data
        self.AOI_model.fit(binned_data)

        # Plot the results
        plt.figure(figsize=(6, 5))
        plt.plot(binned_data.x_value * 180 / np.pi, binned_data.intensity, c='b', label='Intensity mean')
        synthetic_AOI = np.arange(0, np.pi / 2, step=0.01)
        plt.plot(synthetic_AOI * 180 / np.pi, self.AOI_model.model(synthetic_AOI, *self.AOI_model.params), c='r',
                 label='Fitted function')
        plt.xlim([0, 90])
        plt.ylim([binned_data.intensity.min(), binned_data.intensity.max()])
        plt.xlabel(r'AOI [deg]', fontsize=12)
        plt.ylabel(sf_intensity, fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        if path_plot is not None:
            path_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path_plot / "AOI_correction_") + self.AOI_model.name + ".png", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return

    def estimate_range_compensation_function(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity",
                                             path_plot: Path = None, show_plot: bool = False):
        """
        Estimate the range compensation function.

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.
            path_plot (Path): Path to save the plot (optional).
            show_plot (bool): Whether to display the plot.
        """
        print("Start range compensation function estimation.")
        # Bin the data based on range values
        binned_data = RadiometricData.from_tls_pcd_and_binned(pcd, int(np.round(
            pcd.scalar_fields["scalar_Range"].max() * 100)), "scalar_Range", sf_intensity)
        # Fit the range model to the binned data
        self.Range_model.fit(binned_data)

        # Plot the results
        plt.figure(figsize=(6, 5))
        plt.plot(binned_data.x_value, binned_data.intensity, c='b', label='Intensity mean')
        synthetic_range = np.arange(0, pcd.scalar_fields["scalar_Range"].max(), step=0.01)
        plt.plot(synthetic_range, self.Range_model.model(synthetic_range, *self.Range_model.params), c='r',
                 label='Fitted function')
        plt.xlim([pcd.scalar_fields["scalar_Range"].min(), pcd.scalar_fields["scalar_Range"].max()])
        plt.ylim([binned_data.intensity.min(), binned_data.intensity.max()])
        plt.xlabel(r'Range [m]', fontsize=12)
        plt.ylabel(sf_intensity, fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        if path_plot is not None:
            path_plot.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path_plot / "Range_correction_") + self.AOI_model.name + ".png", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return

    def reflectance_normalization_value(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity"):
        """
        Calculate reflectance normalization values for each patch.

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.

        Returns:
            np.ndarray: Reflectance correction values.
        """
        # Calculate the sum and count of intensities for each patch
        sum_patches = np.bincount(pcd.scalar_fields["scalar_Patch_number"], weights=pcd.scalar_fields[sf_intensity])
        counts_patches = np.bincount(pcd.scalar_fields["scalar_Patch_number"])
        # Calculate the mean reflectance for each patch
        patch_reflectance = sum_patches / counts_patches

        # Calculate the correction factor for each point
        mean_refl = np.mean(pcd.scalar_fields[sf_intensity])
        corr_refl = mean_refl / patch_reflectance[pcd.scalar_fields["scalar_Patch_number"]]
        print("Reflectance Correction done")
        return corr_refl

    def estimate_range_aoi_iterative(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity",
                                     path_plot: Path = None, show_plot: bool = False,
                                     estimate_aoi_function: bool = True, estimate_range_function: bool = True,
                                     max_it: int = 10, conv_criteria: float = 0.01):
        """
        Iteratively estimate the range and AOI compensation functions (Cycle 1).

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.
            path_plot (Path): Path to save the plot (optional).
            show_plot (bool): Whether to display the plot.
            estimate_aoi_function (bool): Whether to estimate the AOI function.
            estimate_range_function (bool): Whether to estimate the range function.
            max_it (int): Maximum number of iterations.
            conv_criteria (float): Convergence criteria for stopping the iterations.
        """
        # Initialize correction fields if they don't exist
        if "scalar_CorrRange" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrRange"] = np.ones(np.shape(pcd.scalar_fields["scalar_Range"]))
        if "scalar_CorrAOI" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrAOI"] = np.ones(np.shape(pcd.scalar_fields["scalar_AOI"]))
        if "scalar_CorrReflectance" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrReflectance"] = np.ones(np.shape(pcd.scalar_fields["scalar_AOI"]))

        # Initialize correction and iteration variables
        correction = pcd.scalar_fields["scalar_CorrRange"] * pcd.scalar_fields["scalar_CorrAOI"]
        diff_iterations = conv_criteria
        repetitions = 1

        # Iterate until convergence or maximum iterations are reached
        while diff_iterations >= conv_criteria and repetitions <= max_it:
            repetitions = repetitions + 1

            # AOI Correction
            if estimate_aoi_function:
                pcd.scalar_fields["scalar_TMP_comp_intens"] = pcd.scalar_fields[sf_intensity] * pcd.scalar_fields[
                    "scalar_CorrRange"] * pcd.scalar_fields["scalar_CorrReflectance"]
                self.estimate_AOI_compensation_function(pcd, "scalar_TMP_comp_intens", path_plot=path_plot,
                                                        show_plot=show_plot)
                pcd.scalar_fields["scalar_CorrAOI"] = self.AOI_model.predict(pcd.scalar_fields["scalar_AOI"])

            # Range Correction
            if estimate_range_function:
                pcd.scalar_fields["scalar_TMP_comp_intens"] = pcd.scalar_fields[sf_intensity] * pcd.scalar_fields[
                    "scalar_CorrAOI"] * pcd.scalar_fields["scalar_CorrReflectance"]
                self.estimate_range_compensation_function(pcd, "scalar_TMP_comp_intens", path_plot=path_plot,
                                                          show_plot=show_plot)
                pcd.scalar_fields["scalar_CorrRange"] = self.Range_model.predict(pcd.scalar_fields["scalar_Range"])

            # Calculate the difference between iterations
            diff_iterations = np.median(
                np.abs(correction - pcd.scalar_fields["scalar_CorrAOI"] * pcd.scalar_fields["scalar_CorrRange"]))
            correction = pcd.scalar_fields["scalar_CorrAOI"] * pcd.scalar_fields["scalar_CorrRange"]
            print("Distance, AOI fit: " + str(diff_iterations))

        # Clean up temporary scalar field
        pcd.delete_scalar_field("scalar_TMP_comp_intens")
        return

    def estimate_refl_range_aoi_iterative(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity",
                                          path_plot: Path = None, show_plot: bool = False,
                                          estimate_aoi_function: bool = True, estimate_range_function: bool = True,
                                          max_it_range_aoi: int = 10, conv_criteria_range_aoi: float = 0.01,
                                          max_it_refl: int = 10, conv_criteria_refl: float = 0.01):
        """
        Iteratively estimate reflectance normalization values, range, and AOI compensation functions (Cycle 2).

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.
            path_plot (Path): Path to save the plot (optional).
            show_plot (bool): Whether to display the plot.
            estimate_aoi_function (bool): Whether to estimate the AOI function.
            estimate_range_function (bool): Whether to estimate the range function.
            max_it_range_aoi (int): Maximum number of iterations for range and AOI estimation (Cycle 1).
            conv_criteria_range_aoi (float): Convergence criteria for range and AOI estimation (Cycle 1).
            max_it_refl (int): Maximum number of iterations for reflectance estimation (Cycle 2).
            conv_criteria_refl (float): Convergence criteria for reflectance estimation (Cycle 2).
        """
        # Initialize correction fields if they don't exist
        if "scalar_CorrRange" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrRange"] = np.ones(np.shape(pcd.scalar_fields["scalar_Range"]))
        if "scalar_CorrAOI" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrAOI"] = np.ones(np.shape(pcd.scalar_fields["scalar_AOI"]))
        if "scalar_CorrReflectance" not in pcd.scalar_fields.keys():
            pcd.scalar_fields["scalar_CorrReflectance"] = np.ones(np.shape(pcd.scalar_fields["scalar_AOI"]))

        pcd.scalar_fields["scalar_Patch_number"] = pcd.scalar_fields["scalar_Patch_number"].astype(int)

        # Initialize correction and iteration variables
        correction = pcd.scalar_fields["scalar_CorrRange"] * pcd.scalar_fields["scalar_CorrAOI"] * pcd.scalar_fields[
            "scalar_CorrReflectance"]
        diff_iterations = conv_criteria_refl
        repetitions = 1

        # Iterate until convergence or maximum iterations are reached
        while diff_iterations >= conv_criteria_refl and repetitions <= max_it_refl:
            # Estimate range and AOI compensation functions
            self.estimate_range_aoi_iterative(pcd, sf_intensity, estimate_aoi_function=estimate_aoi_function,
                                              estimate_range_function=estimate_range_function, max_it=max_it_range_aoi,
                                              conv_criteria=conv_criteria_range_aoi, path_plot=path_plot,
                                              show_plot=show_plot)

            # Calculate reflectance normalization values
            pcd.scalar_fields["scalar_TMP_comp_intens"] = pcd.scalar_fields[sf_intensity] * pcd.scalar_fields[
                "scalar_CorrAOI"] * pcd.scalar_fields["scalar_CorrRange"]
            pcd.scalar_fields["scalar_CorrReflectance"] = self.reflectance_normalization_value(pcd,
                                                                                               "scalar_TMP_comp_intens")
            # Calculate the difference between iterations
            diff_iterations = np.median(np.abs(
                correction - pcd.scalar_fields["scalar_CorrRange"] * pcd.scalar_fields["scalar_CorrAOI"] *
                pcd.scalar_fields["scalar_CorrReflectance"]))
            correction = pcd.scalar_fields["scalar_CorrRange"] * pcd.scalar_fields["scalar_CorrAOI"] * \
                         pcd.scalar_fields["scalar_CorrReflectance"]
            print("Reflectance fit: " + str(diff_iterations))

        # Clean up temporary scalar field
        pcd.delete_scalar_field("scalar_TMP_comp_intens")
        return

    def compensate_intensities(self, pcd: TLS_PCDs, sf_intensity: str = "scalar_Intensity",
                               sf_comp_intensity: str = "scalar_Comp_intensity", compAOI: bool = True,
                               compRange: bool = True):
        """
        Compensate intensities based on AOI and range corrections.

        Args:
            pcd (TLS_PCDs): Point cloud data.
            sf_intensity (str): Name of the scalar field representing intensity.
            sf_comp_intensity (str): Name of the scalar field to store compensated intensity.
            compAOI (bool): Whether to apply AOI compensation.
            compRange (bool): Whether to apply range compensation.
        """
        # Apply AOI correction if enabled
        if compAOI:
            corr_AOI = self.AOI_model.predict(pcd.scalar_fields["scalar_AOI"])
        else:
            corr_AOI = np.ones(np.shape(pcd.scalar_fields["scalar_AOI"]))

        # Apply range correction if enabled
        if compRange:
            corr_Range = self.Range_model.predict(pcd.scalar_fields["scalar_Range"])
        else:
            corr_Range = np.ones(np.shape(pcd.scalar_fields["scalar_Range"]))

        # Calculate the compensated intensity
        pcd.scalar_fields[sf_comp_intensity] = pcd.scalar_fields[sf_intensity] * corr_AOI * corr_Range

        return

    def save_to_file(self, path: Path):
        """
        Save the RadiometricCompensation instance to a file.

        Args:
            path (Path): Path to save the file.
        """
        path.parents[0].mkdir(parents=True, exist_ok=True)
        dump((self), path)
        return

    def compensate_pcd_folder(self, input_folder_path: Path, output_folder_path: Path,
                              sf_intensity: str = "scalar_Intensity", sf_comp_intensity: str = "scalar_Comp_intensity",
                              compRange: bool = True, compAOI: bool = True, input_format: str = "*"):
        """
        Compensate intensities for all point clouds in a folder.

        Args:
            input_folder_path (Path): Path to the folder containing input point clouds.
            output_folder_path (Path): Path to the folder to save compensated point clouds.
            sf_intensity (str): Name of the scalar field representing intensity (source).
            sf_comp_intensity (str): Name of the scalar field to store compensated intensity (target).
            compRange (bool): Whether to apply range compensation.
            compAOI (bool): Whether to apply AOI compensation.
            input_format (str): File format of the input point clouds.
        """
        # Get all point cloud files in the input folder
        path_pcds = list(input_folder_path.glob("*." + input_format))
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Process each point cloud file
        for file in path_pcds:
            pcd = TLS_PCDs.load_single_pcd_ply(file)
            self.compensate_intensities(pcd, sf_intensity=sf_intensity, sf_comp_intensity=sf_comp_intensity,
                                        compRange=compRange, compAOI=compAOI)
            pcd.write_pcds_in_separate_files_ply(output_folder_path)
        return
