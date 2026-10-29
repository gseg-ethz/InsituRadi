from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from insituradi.radiometric_data import RadiometricData
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit


class RadiometricModel(ABC):
    """
    Abstract base class for radiometric models.
    Defines the structure for models used in radiometric compensation.
    """

    def __init__(self, name: str, params=None, smoothing=None):
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

    @abstractmethod
    def fit(self):
        """
        Abstract method to fit the model to data.
        """

    @abstractmethod
    def predict(self):
        """
        Abstract method to predict values using the fitted model.
        """


class RadiometricModelGroup(UserDict):
    _hyperparams: dict[str, Any]
    _model: Callable[[Any, ...], RadiometricModel]
    # AOI_model: dict[str, RadiometricModel]

    def __init__(self, model: RadiometricModel, **kwargs):
        super().__init__()
        self._hyperparams = kwargs
        self._model = model

    def _create_for_class(self, cl: str):
        self[cl] = self._model(**self._hyperparams)

    def fit(self, cl: str, data: RadiometricData):
        if cl not in self.keys():
            self._create_for_class(cl)
        self[cl].fit(data)

    def predict(self, cl: str, aoi_data: np.ndarray):
        AOI_corr = self[cl].predict(aoi_data)
        return AOI_corr


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

    def model(self, data: np.ndarray, b: float, sigma: float, c: float):
        """
        Define the Oren-Nayar model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            b (float): Scaling parameter.
            sigma (float): Surface roughness parameter.

        Returns:
            np.ndarray: Model output.
        """
        return (
            b
            * np.cos(data)
            * (
                (1 - 0.5 * (sigma**2) / (sigma**2 + 0.33))
                + (0.45 * (sigma**2) / (sigma**2 + 0.09)) * np.sin(data) * np.tan(data)
            )
            + c
        )

    def fit(self, data: RadiometricData):
        """
        Fit the Oren-Nayar model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted OrenNayarModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
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

    def model(self, data: np.ndarray, a: float, b: float, n: float, c: float):
        """
        Definition of the Blinn-Phong model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): Diffuse reflection parameter.
            b (float): Specular reflection parameter.
            n (float): Shininess parameter.
            c (float): Offset parameter.

        Returns:
            np.ndarray: Model output.
        """
        return b * (a * np.cos(data) + (1 - a) * (np.abs(np.cos(data)) ** n)) + c

    def fit(self, data: RadiometricData):
        """
        Fit the Blinn-Phong model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted BlinnPhongModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
        return AOI_corr


class PhongModel_Helios(RadiometricModel):
    """
    Phong model for radiometric compensation.
    """

    def __init__(self, norm0: float, params=None):
        """
        Initialize the Phong model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("Phong", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float, n: float):
        """
        Define the Phong model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): Diffuse reflection parameter.
            b (float): Specular reflection parameter.
            n (float): Shininess parameter.

        Returns:
            np.ndarray: Model output.
        """
        return b * (a * np.cos(data) + (1 - a) * (np.abs(np.cos(data)) ** n))

    def fit(self, data: RadiometricData):
        """
        Fit the Phong model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted PhongModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
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

    def model(self, data: np.ndarray, a: float, b: float, c: float):
        """
        Define the Lommel-Seeliger model.

        Args:
            data (np.ndarray): Input data (e.g., angle of incidence).
            a (float): First parameter.
            b (float): Second parameter.

        Returns:
            np.ndarray: Model output.
        """
        return (a * np.cos(data) + b * ((np.cos(data)) ** 2)) + c

    def fit(self, data: RadiometricData):
        """
        Fit the Lommel-Seeliger model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted LommelSeeligerModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
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
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted LambertianModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
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
        return a * np.cos(data) + b

    def fit(self, data: RadiometricData):
        """
        Fit the Adapted Lambertian model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted AdaptedLambertianModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
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

    def model(self, data: np.ndarray, a: float, b: float, c: float, d: float):
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
        return (
            a
            * (
                b * np.cos(data)
                + (
                    (1 - b)
                    * (np.exp(-(np.tan(data) ** 2) / (c**2)) / (np.cos(data) ** 5))
                )
            )
            + d
        )

    def fit(self, data: RadiometricData):
        """
        Fit the Lambert-Beckmann model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_cos, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_cos
        print("Fitted LambertBeckmannModel")

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Correction factor.
        """
        AOI_corr = self.model(self.norm0, *self.params) / self.model(
            aoi_data, *self.params
        )
        AOI_corr[AOI_corr < 0] = 1
        return AOI_corr


class SmoothingSplines(RadiometricModel):
    """
    Smoothing splines model for radiometric compensation.
    """

    def __init__(self, norm0: float, params=None, smoothing=1):
        """
        Initialize the Smoothing Splines model.

        Args:
            norm0 (float): Normalization value (phi_0).
            params: Initial parameters for the model.
        """
        super().__init__("SmoothingSplines", params)
        self.norm0 = norm0
        self.smoothing = smoothing

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

    def fit_L2(self, data: RadiometricData):
        """
        Fit the Smoothing Splines model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        noise = np.nanmean(pd.Series(data.intensity).rolling(window=20).std())
        noise = noise if noise > 0 else 1 / len(data.x_value)  # Avoid division by zero
        # popt_cos = splrep(data.x_value, data.intensity, s=len(data.x_value) * noise ** 2)

        smoothing_factor = len(data.x_value) * (noise**2) * self.smoothing
        popt_cos = splrep(data.x_value, data.intensity, s=smoothing_factor)
        self.params = popt_cos
        print("Fitted SmoothingSplines")

    def fit(self, data: RadiometricData):
        """
        Fit the Smoothing Splines model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """

        noise = np.nanmean(pd.Series(data.intensity).rolling(window=20).std())

        smoothing_factor = len(data.x_value) * (noise**2) * self.smoothing

        pred = data.x_value
        weights = np.ones(len(data.intensity), dtype=float)
        change = 1000
        it = 0
        while change > 500 and it <= 20:
            it = it + 1
            popt_cos = splrep(
                data.x_value, data.intensity, w=weights, s=smoothing_factor
            )
            self.params = popt_cos
            res = data.intensity - self.model(data.x_value, *self.params)
            mad = np.median(np.abs(res - np.median(res)))
            change = np.median(np.abs(pred - self.model(data.x_value, *self.params)))
            # print("change: "+str(change))
            pred = self.model(data.x_value, *self.params)

            weights = (np.abs(res) <= 3 / 0.6745 * mad).astype(float)
            weights[weights == 0] = weights[weights == 0] + 0.001

        print("Fitted SmoothingSplines")

    def predict(self, data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Correction factor.
        """
        corr = self.model(self.norm0, *self.params) / self.model(data, *self.params)
        corr[corr < 0] = 1
        return corr


class R2Model(RadiometricModel):
    """
    R2 model for radiometric compensation.
    """

    def __init__(self, norm0: float, params=None):
        """
        Initialize the R2 model.

        Args:
            norm0 (float): Normalization value.
            params: Initial parameters for the model.
        """
        super().__init__("R2", params)
        self.norm0 = norm0

    def model(self, data: np.ndarray, a: float, b: float):
        """
        Define the R2 model.

        Args:
            data (np.ndarray): Input data.
            a (float): First parameter.
            b (float): Second parameter.

        Returns:
            np.ndarray: Model output.
        """
        return a * (data**-2) + b

    def fit(self, data: RadiometricData):
        """
        Fit the R2 model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_exp, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_exp
        print("Fitted R2Model")

    def predict(self, data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Correction factor.
        """
        corr = self.model(self.norm0, *self.params) / self.model(data, *self.params)
        corr[corr < 0] = 1
        return corr


class R2Model_extended(RadiometricModel):
    """
    R2 model for radiometric compensation.
    """

    def __init__(self, norm0: float, params=None):
        """
        Initialize the R2 model.

        Args:
            norm0 (float): Normalization value.
            params: Initial parameters for the model.
        """
        super().__init__("R2_extended", params)
        self.norm0 = norm0

    def model(
        self,
        data: np.ndarray,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
    ):
        """
        Define the R2 model.

        Args:
            data (np.ndarray): Input data.
            a (float): First parameter.
            b (float): Second parameter.
            c (float): Third parameter.
            d (float): Fourth parameter.
            e (float): Fifth parameter.
            f (float): Sixth parameter.

        Returns:
            np.ndarray: Model output.
        """
        return a * (data**-2) + b + (c + f * data) * np.sin(d * data + e)

    def fit_L2(self, data: RadiometricData):
        """
        Fit the R2 model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """
        popt_exp, _ = curve_fit(
            self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
        )
        self.params = popt_exp
        print("Fitted R2Model")

    def fit(self, data: RadiometricData):
        """
        Fit the Smoothing Splines model to the given data.

        Args:
            data (RadiometricData): Data to fit the model to.
        """

        pred = data.x_value
        weights = np.ones(len(data.intensity), dtype=float)
        change = 1000
        it = 0
        while change > 500 and it <= 20:
            it = it + 1
            popt_exp, _ = curve_fit(
                self.model, data.x_value, data.intensity, maxfev=5000, p0=self.params
            )
            self.params = popt_exp
            res = data.intensity - self.model(data.x_value, *self.params)
            mad = np.median(np.abs(res - np.median(res)))
            change = np.median(np.abs(pred - self.model(data.x_value, *self.params)))
            # print("change: "+str(change))
            pred = self.model(data.x_value, *self.params)

            weights = (np.abs(res) <= 3 / 0.6745 * mad).astype(float)
            weights[weights == 0] = weights[weights == 0] + 0.001

        print("Fitted SmoothingSplines")

    def predict(self, data: np.ndarray):
        """
        Predict the correction factor using the fitted model.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Correction factor.
        """
        corr = self.model(self.norm0, *self.params) / self.model(data, *self.params)
        corr[corr < 0] = 1
        return corr


class UnityRadiometricModel(RadiometricModel):
    """
    Radiometric model that always returns 1 (no correction).
    """

    def __init__(self):
        """
        Initialize the unity model.
        """
        super().__init__("UnityModel", params=[])

    def model(self, data: np.ndarray):
        """
        Always return 1.

        Args:
            data (np.ndarray): Input data (ignored).

        Returns:
            np.ndarray: Ones of the same shape as input.
        """
        return np.ones_like(data)

    def fit(self):
        return

    def predict(self, aoi_data: np.ndarray):
        """
        Predict the correction factor (always 1).

        Args:
            aoi_data (np.ndarray): Angle of incidence data.

        Returns:
            np.ndarray: Array of ones.
        """
        return self.model(aoi_data)
