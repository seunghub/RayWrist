"""RoboCasa source-image geometry and native spatial augmentation affines."""

from pathlib import Path
import numpy as np
from .robocasa_fk import GR1Geometry

LOWER = np.array([-0.06, -0.06, -0.05])
UPPER = np.array([0.06, 0.06, 0.15])
FIELDS = (
    "anchor_uv",
    "anchor_valid",
    "anchor_xyz",
    "volume_grid",
    "depth_code",
    "ray_code",
    "pose_code",
    "valid_ray",
)


def physical_state(value, layout):
    x = np.asarray(value, np.float64)
    if layout == "raw44":
        if x.shape != (44,):
            raise ValueError("Original44D physical state required")
        x = np.concatenate([x[:13], x[22:35], x[41:44]])
    elif layout == "pi_active29":
        if x.shape != (29,):
            raise ValueError("Active29D state required")
        x = np.concatenate([x[:7], x[14:20], x[7:14], x[20:26], x[26:29]])
    elif layout != "geometry29":
        raise ValueError(layout)
    if x.shape != (29,) or not np.isfinite(x).all():
        raise ValueError("Finite unnormalized GR1 physical state required")
    return x


def n17_replay_affine(replay):
    """Map native256 crop pixels through the original spatial ReplayCompose.

    No transform is resampled and photometric transforms do not affect rays.
    The published model uses resize -> fractional crop -> resize, no rotation.
    Unexpected geometry transforms are rejected rather than silently ignored.
    """
    affine = np.eye(3)
    hw = np.array([256.0, 256.0])
    for item in replay["transforms"]:
        if not item.get("applied", False):
            continue
        kind = item["__class_fullname__"].split(".")[-1]
        params = item.get("params") or {}
        if kind == "SmallestMaxSize":
            target = item.get("max_size", 256)
            if isinstance(target, (tuple, list)):
                if len(set(target)) != 1:
                    raise ValueError("Unfixed target resize")
                target = target[0]
            old = hw.copy()
            hw = np.round(hw * float(target) / hw.min())
            scale = hw[::-1] / old[::-1]
            a = np.array(
                [
                    [scale[0], 0, (scale[0] - 1) / 2],
                    [0, scale[1], (scale[1] - 1) / 2],
                    [0, 0, 1],
                ]
            )
            affine = a @ affine
        elif kind in (
            "FractionalRandomCrop",
            "FractionalCenterCrop",
            "RandomCrop",
            "CenterCrop",
        ):
            (x0, y0, x1, y1) = params["crop_coords"]
            affine = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1]]) @ affine
            hw = np.array([y1 - y0, x1 - x0], np.float64)
        elif kind in ("ColorJitter", "Normalize", "NoOp"):
            continue
        else:
            raise ValueError("Untracked geometry augmentation: " + kind)
    if not np.array_equal(hw, [256, 256]):
        raise ValueError(("Native model image size changed", hw))
    return affine


class ToolRayGeometry:

    def __init__(self, calibration=None):
        if calibration is None:
            calibration = (
                Path(__file__).resolve().parents[1]
                / "configs/robocasa_calibration.json"
            )
        self.fk = GR1Geometry(calibration)
        self.center = (LOWER + UPPER) / 2
        self.half = (UPPER - LOWER) / 2
        axes = [np.linspace(LOWER[i], UPPER[i], 17)[1::2] for i in range(3)]
        (z, y, x) = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        self.anchors = np.stack([x, y, z], -1).reshape(512, 3)

    def crops(self, image, state, *, layout="raw44"):
        state = physical_state(state, layout)
        return self.fk.crops(image, state)
