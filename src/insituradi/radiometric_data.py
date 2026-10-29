from dataclasses import dataclass

import numpy as np
from insituradi.tls_pcd import TLS_PCDs


@dataclass
class RadiometricData:
    """
    A dataclass to store radiometric data, including x-values (e.g., angle of incidence or range)
    and corresponding intensity values.
    """

    x_value: np.ndarray  # Array of x-values (e.g., angle of incidence or range)
    intensity: np.ndarray  # Array of intensity values corresponding to the x-values

    @classmethod
    def from_tls_pcd_and_binned(
        cls,
        tls_pcd: TLS_PCDs,
        bins: [int, np.ndarray],
        x_value_sf: str,
        intensity_sf: str,
    ):
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
        # median_intens = scipy.stats.binned_statistic(
        #     tls_pcd.scalar_fields[x_value_sf],  # The x-values to bin by
        #     values=tls_pcd.scalar_fields[intensity_sf],  # The intensity values to bin
        #     statistic="median",  # Use the median as the statistic for binning
        #     bins=bins,  # Number of bins or bin edges
        # )

        x_vals = tls_pcd.scalar_fields[x_value_sf]
        intensities = tls_pcd.scalar_fields[intensity_sf]

        # 1. Determine the exact bin edges
        if isinstance(bins, int):
            # Calculate bin edges if 'bins' is just an integer (number of bins)
            bin_edges = np.histogram_bin_edges(x_vals, bins=bins)
        else:
            # If 'bins' is already an array of edges, just ensure it's a NumPy array
            bin_edges = np.asarray(bins)

        # 2. Sort the data by x-values to avoid slow grouping iterations
        sort_idx = np.argsort(x_vals)
        sorted_x = x_vals[sort_idx]
        sorted_intensities = intensities[sort_idx]

        # 3. Find exactly where the sorted x-values cross the bin edges
        split_indices = np.searchsorted(sorted_x, bin_edges)

        # 4. Calculate the median for each bin slice
        medians = np.zeros(len(bin_edges) - 1)
        for i in range(len(medians)):
            start, end = split_indices[i], split_indices[i + 1]
            if start == end:
                medians[i] = np.nan  # Handle empty bins just like SciPy does
            else:
                medians[i] = np.median(sorted_intensities[start:end])

        # 5. Package as a tuple to mimic SciPy's BinnedStatisticResult
        # median_intens[0] gives the medians, median_intens[1] gives the bin edges
        median_intens = (medians, bin_edges)

        # Calculate the middle value of each bin
        middle_x_value = (median_intens[1][:-1] + median_intens[1][1:]) / 2

        # Filter out bins with NaN values in either the x-values or the intensity values
        # Filter out NaN and infinite values in both arrays
        mask_x = np.isfinite(middle_x_value)
        mask_i = np.isfinite(median_intens[0])
        filter_data = mask_x & mask_i

        # Extract the filtered x-values and intensity values
        binned_x_value = middle_x_value[filter_data]
        intensity_binned = median_intens[0][filter_data]

        # Return a new RadiometricData instance with the binned data
        return cls(x_value=binned_x_value, intensity=intensity_binned)
