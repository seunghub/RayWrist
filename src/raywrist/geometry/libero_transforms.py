"""LIBERO saved-image and native-encoder coordinate transforms."""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class GeometryConfig:
    near_m: float = 1e-06
    box_min: tuple = (-0.06, -0.06, -0.05)
    box_max: tuple = (0.06, 0.06, 0.15)


def token_centers():
    (y, x) = np.meshgrid(
        np.arange(8) * 32 + 16.0, np.arange(8) * 32 + 16.0, indexing="ij"
    )
    return np.stack([x.ravel(), y.ravel()], -1)


def encoder_from_saved():
    a = 256.0 / 243.0
    return np.array([[a, 0, -6 * a], [0, a, -6 * a], [0, 0, 1.0]])


def saved_from_optical_pixel():
    return np.array([[-1.0, 0, 256], [0, 1.0, 0], [0, 0, 1.0]])


def crop_from_saved(c0, r0):
    a = 256.0 / 96.0
    return encoder_from_saved() @ np.array(
        [[a, 0, -c0 * a], [0, a, -r0 * a], [0, 0, 1.0]]
    )
