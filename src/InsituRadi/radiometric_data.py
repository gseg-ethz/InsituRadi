from dataclasses import dataclass

import numpy as np
import scipy
from InsituRadi.tls_pcd import TLS_PCDs


@dataclass
class RadiometricData:
    """
    A dataclass to store radiometric data, including x-values (e.g., angle of incidence or range)
    and corresponding intensity values.
    """
    x_value: np.ndarray  # Array of x-values (e.g., angle of incidence or range)
    intensity: np.ndarray  # Array of intensity values corresponding to the x-values

    @classmethod
    def from_tls_pcd_and_binned(cls, tls_pcd: TLS_PCDs, bins: [int, np.ndarray], x_value_sf: str, intensity_sf: str):
        """
        Create a RadiometricData instance by binning data from a TLS_PCDs object.

        Args:
            tls_pcd (TLS_PCDs): Point cloud data containing scalar fields.
            bins (int or np.ndarray): Number of bins or bin edges for binning the data.
            x_value_sf (str): Name of the scalar field representing the x-values (e.g., angle of incidence or range).
            intensity_sf (str): Name of the scalar field representing the intensity values.

        Returns:
            RadiometricData: An instance of RadiometricData containing binned x-values and intensity values.
        """
        # Bin the intensity values based on the x-values
        mean_intens = scipy.stats.binned_statistic(
            tls_pcd.scalar_fields[x_value_sf],  # The x-values to bin by
            values=tls_pcd.scalar_fields[intensity_sf],  # The intensity values to bin
            statistic='mean',  # Use the mean as the statistic for binning
            bins=bins  # Number of bins or bin edges
        )

        # Calculate the middle value of each bin
        middle_x_value = (mean_intens[1][:-1] + mean_intens[1][1:]) / 2

        # Filter out bins with NaN values in either the x-values or the intensity values
        filter_data = (~np.isnan(middle_x_value)) & (~np.isnan(mean_intens[0]))

        # Extract the filtered x-values and intensity values
        binned_x_value = middle_x_value[filter_data]
        intensity_binned = mean_intens[0][filter_data]

        # Return a new RadiometricData instance with the binned data
        return cls(x_value=binned_x_value, intensity=intensity_binned)
