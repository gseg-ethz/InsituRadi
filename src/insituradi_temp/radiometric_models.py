from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, BSpline
from scipy.optimize import curve_fit
from insituradi.radiometric_data import RadiometricData


class RadiometricModel(ABC):
    """
    Abstract base class for radiometric models.
    Defines the structure for models used in radiometric compensation.
    """

    def __init__(self, name: str, params=None):
        """
        Initialize the RadiometricModel.

        Args:
            name (str): Name of the model.
            params: Initial parameters for the model.
        """
        self.params = params
        self.name = name

    @abstractmethod
    def model(self):
        """
        Abstract method to define the mathematical model.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Abstract method to fit the model to data.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Abstract method to predict values using the fitted model.
        """
        pass


class OrenNayarModel(RadiometricModel):
    """
    Oren-Nayar model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Oren-Nayar model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("OrenNayar", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, b: float, sigma: float):
        """
        Define the Oren-Nayar model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            b (float): Scaling parameter.
            sigma (float): Surface roughness parameter.

        Returns:
            np.ndarray: Model output.
        """
        return b * np.cos(data) * (
                (1 - 0.5 * (sigma ** 2) / (sigma ** 2 + 0.33)) + (0.45 * (sigma ** 2) / (sigma ** 2 + 0.09)) * np.sin(
            data) * np.tan(data))

    def fit(self, data: RadiometricData):
        """
        Fit the Oren-Nayar model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted OrenNayarModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class BlinnPhongModel(RadiometricModel):
    """
    Blinn-Phong model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Blinn-Phong model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("BlinnPhong", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float, n: float):
        """
        Define the Blinn-Phong model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): Diffuse reflection parameter.
            b (float): Specular reflection parameter.
            n (float): Shininess parameter.

        Returns:
            np.ndarray: Model output.
        """
        return b * (a * np.cos(data) + (1 - a) * (np.cos(data) ** n))

    def fit(self, data: RadiometricData):
        """
        Fit the Blinn-Phong model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted BlinnPhongModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class LommelSeeligerModel(RadiometricModel):
    """
    Lommel-Seeliger model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Lommel-Seeliger model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("LommelSeeliger", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float):
        """
        Define the Lommel-Seeliger model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): First parameter.
            b (float): Second parameter.

        Returns:
            np.ndarray: Model output.
        """
        return (a * np.cos(data) + b * ((np.cos(data)) ** 2))

    def fit(self, data: RadiometricData):
        """
        Fit the Lommel-Seeliger model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted LommelSeeligerModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class LambertianModel(RadiometricModel):
    """
    Lambertian model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Lambertian model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("Lambertian", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float):
        """
        Define the Lambertian model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): Scaling parameter.

        Returns:
            np.ndarray: Model output.
        """
        return a * np.cos(data)

    def fit(self, data: RadiometricData):
        """
        Fit the Lambertian model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted LambertianModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class AdaptedLambertianModel(RadiometricModel):
    """
    Adapted Lambertian model for radiometric compensation.
    This model extends the Lambertian model with an additional parameter.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Adapted Lambertian model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("AdaptedLambertian", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float):
        """
        Define the Adapted Lambertian model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): First parameter.
            b (float): Second parameter.

        Returns:
            np.ndarray: Model output.
        """
        return a + b * np.cos(data)

    def fit(self, data: RadiometricData):
        """
        Fit the Adapted Lambertian model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted AdaptedLambertianModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class LambertBeckmannModel(RadiometricModel):
    """
    Lambert-Beckmann model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Lambert-Beckmann model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("LambertBeckmann", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float, c: float):
        """
        Define the Lambert-Beckmann model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): First parameter.
            b (float): Second parameter.
            c (float): Third parameter.

        Returns:
            np.ndarray: Model output.
        """
        return a * (b * np.cos(data) + ((1 - b) * (np.exp(-(np.tan(data) ** 2) / (c ** 2)) / (np.cos(data) ** 5))))

    def fit(self, data: RadiometricData):
        """
        Fit the Lambert-Beckmann model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params)
        self.params = popt_cos
        print("Fitted LambertBeckmannModel")
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(aoi_data, *self.params)
        return AOI_corr


class SmoothingSplines(RadiometricModel):
    """
    Smoothing splines model for radiometric compensation.
    """
    def __init__(self, norm0: float, params=None):
        """
        Initialize the Smoothing Splines model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("SmoothingSplines", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, *params):
        """
        Define the Smoothing Splines model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            *params: Spline parameters.

        Returns:
            np.ndarray: Model output.
        """
        return BSpline(*params)(data)

    def fit(self, data: RadiometricData):
        """
        Fit the Smoothing Splines model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        noise = np.nanmean(pd.Series(data.intensity).rolling(window=20).std())
        popt_cos = splrep(data.x_value, data.intensity, s=len(data.x_value) * noise ** 2)
        self.params = popt_cos
        print("Fitted SmoothingSplines")
        return

    def predict(self, data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Correction factor.
        """
        corr = self.model(self.norm0, *self.params) / self.model(data, *self.params)
        return corr

