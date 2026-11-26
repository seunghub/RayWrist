"""RoboCasa virtual wrist and grid frames with separate source-image projection."""

from dataclasses import dataclass
import numpy as np
from .robocasa_source import ToolRayGeometry, physical_state, LOWER, UPPER

GR1_EE_FROM_GRIP = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class EEVirtualConfig:
    name: str = "ee_grip_aligned"
    origin_in_ee_m: tuple = (0.0, 0.0, -0.06)
    forward_frame: str = "grip"
    volume_frame: str = "virtual"
    side: int = 8
    depth_samples: int = 16


def ray_box(origin, directions, lower, upper, samples=16):
    n = len(directions)
    lo = np.full(n, 1e-06)
    hi = np.full(n, np.inf)
    valid = np.ones(n, bool)
    for axis in range(3):
        parallel = np.abs(directions[:, axis]) < 1e-12
        valid &= ~(
            parallel & ((origin[axis] < lower[axis]) | (origin[axis] > upper[axis]))
        )
        denom = np.where(parallel, 1.0, directions[:, axis])
        a = (lower[axis] - origin[axis]) / denom
        b = (upper[axis] - origin[axis]) / denom
        lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(a, b)))
        hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(a, b)))
    valid &= np.isfinite(hi) & (hi >= lo)
    rho = np.where(
        valid[:, None],
        lo[:, None] + (hi - lo)[:, None] * (np.arange(samples) + 0.5) / samples,
        0.0,
    )
    xyz = origin + rho[..., None] * directions[:, None, :]
    xyz = np.where(valid[:, None, None], xyz, (lower + upper) / 2)
    return (xyz, rho, valid)


def project(p, xyz):
    h = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], -1) @ p.T
    depth = h[..., 2]
    uv = h[..., :2] / np.where(np.abs(depth[..., None]) > 1e-12, depth[..., None], 1.0)
    return (uv, depth)


class EEVirtualGeometry:

    def __init__(self, config=EEVirtualConfig(), calibration=None):
        self.config = config
        self.source = ToolRayGeometry(calibration)
        if config.forward_frame not in ["ee", "grip"]:
            raise ValueError(config.forward_frame)
        if config.volume_frame not in ["ee", "virtual"]:
            raise ValueError(config.volume_frame)
        self.ee_from_virtual = np.eye(4)
        self.ee_from_virtual[:3, :3] = (
            np.eye(3) if config.forward_frame == "ee" else GR1_EE_FROM_GRIP
        )
        self.ee_from_virtual[:3, 3] = config.origin_in_ee_m
        self.ee_from_volume = (
            self.ee_from_virtual.copy()
            if config.volume_frame == "virtual"
            else np.eye(4)
        )
        self.query_K = self.source.fk.K.copy()
        self.query_hw = tuple(self.source.fk.cal["camera"]["raw_hw"])

    def context(self, state, arm, *, layout="raw44", affine=None):
        state = physical_state(state, layout)
        entry = self.source.fk.project(state)[arm]
        c = self.config
        (h, w) = self.query_hw
        (yy, xx) = np.meshgrid(
            (np.arange(c.side) + 0.5) * h / c.side,
            (np.arange(c.side) + 0.5) * w / c.side,
            indexing="ij",
        )
        pixel = np.stack([xx.ravel(), yy.ravel(), np.ones(c.side * c.side)], -1)
        directions_virtual = np.linalg.solve(self.query_K, pixel.T).T
        directions_virtual /= np.linalg.norm(directions_virtual, axis=-1, keepdims=True)
        volume_from_virtual = np.linalg.inv(self.ee_from_volume) @ self.ee_from_virtual
        origin = volume_from_virtual[:3, 3]
        directions = directions_virtual @ volume_from_virtual[:3, :3].T
        (xyz, rho, valid) = ray_box(origin, directions, LOWER, UPPER, c.depth_samples)
        ee_xyz = xyz @ self.ee_from_volume[:3, :3].T + self.ee_from_volume[:3, 3]
        p_head = entry["P_tool"] @ self.ee_from_volume
        offsets = np.floor(entry["uv"] - 48 + 0.5).astype(int)
        scale = 256 / 96
        crop = np.array(
            [
                [scale, 0, -offsets[0] * scale + (scale - 1) / 2],
                [0, scale, -offsets[1] * scale + (scale - 1) / 2],
                [0, 0, 1.0],
            ]
        )
        affine = np.eye(3) if affine is None else np.asarray(affine)
        p_crop = affine @ crop @ p_head
        (uv_head, z_head) = project(p_head, xyz)
        (uv_crop, z_crop) = project(p_crop, xyz)
        anchors = self.source.anchors.copy()
        (auv, az) = project(p_crop, anchors)
        (rawuv, rawz) = project(p_head, anchors)
        covered = (
            (az > 1e-06)
            & (auv >= -0.5).all(-1)
            & (auv < 255.5).all(-1)
            & (rawuv >= -0.5).all(-1)
            & (rawuv < 255.5).all(-1)
        )
        observed = (
            (z_head > 1e-06)
            & (uv_head >= -0.5).all(-1)
            & (uv_head < 255.5).all(-1)
            & (uv_crop >= -0.5).all(-1)
            & (uv_crop < 255.5).all(-1)
            & valid[:, None]
        )
        center = (LOWER + UPPER) / 2
        half = (UPPER - LOWER) / 2
        robot_from_volume = entry["tool_from_robot"] @ self.ee_from_volume
        features = dict(
            anchor_uv=(auv + 0.5) / 128 - 1,
            anchor_valid=covered,
            anchor_xyz=(anchors - center) / half,
            volume_grid=(xyz - center) / half,
            depth_code=np.concatenate(
                [(xyz - center) / half, rho[..., None] / 0.2], -1
            ),
            ray_code=np.concatenate(
                [np.broadcast_to(origin, (c.side * c.side, 3)), directions], -1
            ),
            pose_code=np.r_[
                robot_from_volume[:3, 3], robot_from_volume[:3, :3].reshape(9)
            ],
            valid_ray=valid,
        )
        features = {
            k: np.asarray(
                v, dtype=bool if k in ["anchor_valid", "valid_ray"] else np.float32
            )
            for (k, v) in features.items()
        }
        return dict(
            features=features,
            xyz_volume=xyz,
            xyz_ee=ee_xyz,
            xyz_virtual=ee_xyz @ np.linalg.inv(self.ee_from_virtual)[:3, :3].T
            + np.linalg.inv(self.ee_from_virtual)[:3, 3],
            rho=rho,
            valid_ray=valid,
            source_supported=observed,
            anchor_source_supported=covered,
            source_head_uv=uv_head,
            source_crop_uv=uv_crop,
            query_K=self.query_K,
            query_hw=np.array(self.query_hw),
            query_pixels=pixel[:, :2],
            ee_from_virtual=self.ee_from_virtual.copy(),
            ee_from_volume=self.ee_from_volume.copy(),
            source_P_head=p_head,
            source_P_crop=p_crop,
            robot_from_volume=robot_from_volume,
            ray_origin_volume=origin,
            ray_direction_volume=directions,
            crop_offsets=offsets,
            source_projection_not_virtual_query_projection=True,
        )
