"""RoboTwin Aloha geometry from raw physical drive targets and fixed camera calibration."""

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "configs/robotwin_calibration.json"
ARMS = ("left", "right")
TENSOR_FIELDS = (
    "anchor_uv",
    "anchor_valid",
    "anchor_xyz",
    "volume_grid",
    "depth_code",
    "valid_ray",
    "pose_code",
    "ray_code",
)


def _physical_states(states):
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 14 or (not np.isfinite(x).all()):
        raise ValueError(
            "Geometry requires finite RAW physical native joints [B,14], before normalization"
        )
    return x


def _crop_offsets(image, offset):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.shape != (240, 320, 3):
        raise ValueError("Head RGB must be native uint8[240,320,3]")
    (c0, r0) = map(int, offset)
    output = np.zeros((120, 120, 3), np.uint8)
    (xa, xb) = (max(c0, 0), min(c0 + 120, 320))
    (ya, yb) = (max(r0, 0), min(r0 + 120, 240))
    if xb > xa and yb > ya:
        output[ya - r0 : yb - r0, xa - c0 : xb - c0] = image[ya:yb, xa:xb]
    return output


def crop_heads(heads, geometry):
    """Return native120px crops. Geometry and heads must refer to identical rows."""
    heads = np.asarray(heads)
    if heads.ndim != 4 or heads.shape[1:] != (240, 320, 3) or heads.dtype != np.uint8:
        raise ValueError("Heads must be native uint8[B,240,320,3]")
    result = {}
    for arm in ARMS:
        offsets = np.asarray(geometry[arm]["offsets"])
        if offsets.shape != (len(heads), 2):
            raise ValueError("Crop offset batch differs from head batch")
        result[arm] = np.stack(
            [_crop_offsets(im, off) for (im, off) in zip(heads, offsets)]
        )
    return result


class Geometry:

    def __init__(self, calibration_path=CALIBRATION):
        self.calibration_path = Path(calibration_path)
        self.calibration_sha256 = hashlib.sha256(
            self.calibration_path.read_bytes()
        ).hexdigest()
        self.calibration = json.loads(self.calibration_path.read_text())
        cal = self.calibration
        if cal["schema"] != "robotwin-aloha-raygrid-geometry-v1":
            raise ValueError("Unexpected calibration schema")
        if cal["crop"].get("center_frame") != "tool_TCP":
            raise ValueError(
                "This campaign requires the authorized TCP120 crop calibration"
            )
        self.lower = np.asarray(cal["volume"]["lower"], np.float64)
        self.upper = np.asarray(cal["volume"]["upper"], np.float64)
        (self.center, self.half) = (
            (self.lower + self.upper) / 2,
            (self.upper - self.lower) / 2,
        )
        self.tool_from_link = np.linalg.inv(
            np.asarray(cal["T_link6_from_tool"], np.float64)
        )
        self.link_from_tool = np.asarray(cal["T_link6_from_tool"], np.float64)
        self.head_K = np.asarray(cal["head"]["K"], np.float64)
        self.head_E = np.asarray(cal["head"]["T_optical_from_world"], np.float64)
        self.head_P = self.head_K @ self.head_E[:3]
        self.head_near = float(cal["head"]["near_optical_z_m"])
        self.chains = cal["fk_chains"]
        axes = [np.linspace(self.lower[a], self.upper[a], 17)[1::2] for a in range(3)]
        (zz, yy, xx) = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        self.anchors = np.stack((xx, yy, zz), -1).reshape(512, 3)
        (yy, xx) = np.meshgrid(
            np.arange(16) * 14 + 7.0, np.arange(16) * 14 + 7.0, indexing="ij"
        )
        self.wrist_raw_uv = np.stack((xx.ravel() / 0.7, (yy.ravel() - 28) / 0.7), -1)
        self.wrist_content_mask = (
            (self.wrist_raw_uv[:, 0] >= 0)
            & (self.wrist_raw_uv[:, 0] < 320)
            & (self.wrist_raw_uv[:, 1] >= 0)
            & (self.wrist_raw_uv[:, 1] < 240)
        )
        self.static = {}
        for arm in ARMS:
            cam = cal["wrists"][arm]
            T = self.tool_from_link @ np.asarray(
                cam["T_link6_from_optical"], np.float64
            )
            o = T[:3, 3]
            dc = (
                np.c_[self.wrist_raw_uv, np.ones(256)]
                @ np.linalg.inv(np.asarray(cam["K"])).T
            )
            dc /= np.linalg.norm(dc, axis=1, keepdims=True)
            d = dc @ T[:3, :3].T
            near_rho = float(cam["near_optical_z_m"]) / dc[:, 2]
            (lo, hi) = (near_rho.copy(), np.full(256, np.inf))
            valid = self.wrist_content_mask.copy()
            for a in range(3):
                parallel = np.abs(d[:, a]) < 1e-14
                valid &= ~(parallel & ((o[a] < self.lower[a]) | (o[a] > self.upper[a])))
                den = np.where(parallel, 1.0, d[:, a])
                (s0, s1) = ((self.lower[a] - o[a]) / den, (self.upper[a] - o[a]) / den)
                lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(s0, s1)))
                hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(s0, s1)))
            valid &= np.isfinite(hi) & (hi >= lo)
            rho = np.where(
                valid[:, None],
                lo[:, None] + (hi - lo)[:, None] * (np.arange(16) + 0.5) / 16,
                0.0,
            )
            xyz = o[None, None] + rho[..., None] * d[:, None]
            self.static[arm] = dict(
                o=o,
                d=d,
                valid=valid,
                rho=rho,
                xyz=xyz,
                optical_direction=dc,
                near_rho=near_rho,
                rho_min=lo,
                rho_max=hi,
            )

    def fk_ee(self, raw_states):
        """Vectorized world-from-native-link6 FK from raw drive targets."""
        states = _physical_states(raw_states)
        B = len(states)
        result = {}
        for arm, offset in (("left", 0), ("right", 7)):
            chain = self.chains[arm]
            T = np.broadcast_to(np.asarray(chain["base"]), (B, 4, 4)).copy()
            for joint, angle in zip(chain["joints"], states[:, offset : offset + 6].T):
                motion = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
                motion[:, :3, :3] = Rotation.from_rotvec(
                    angle[:, None] * np.asarray(joint["axis"])
                ).as_matrix()
                T = T @ np.asarray(joint["origin"]) @ motion
            result[arm] = T
        return result

    def batch(self, raw_states):
        states = _physical_states(raw_states)
        B = len(states)
        ee = self.fk_ee(states)
        result = {}
        anchors = np.broadcast_to(self.anchors, (B, 512, 3))
        anchor_code = np.broadcast_to(
            (self.anchors - self.center) / self.half, (B, 512, 3)
        )

        def project(P, points, width, height):
            h = np.einsum(
                "bij,b...j->b...i",
                P,
                np.concatenate((points, np.ones((*points.shape[:-1], 1))), -1),
            )
            uv = h[..., :2] / np.where(np.abs(h[..., 2:]) > 1e-15, h[..., 2:], 1.0)
            good = (
                (h[..., 2] > self.head_near)
                & np.isfinite(uv).all(-1)
                & (uv[..., 0] >= 0)
                & (uv[..., 0] < width)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] < height)
            )
            return (uv, good)

        for arm in ARMS:
            T = ee[arm] @ self.link_from_tool
            P = self.head_P[None] @ T
            center_h = np.c_[T[:, :3, 3], np.ones(B)] @ self.head_P.T
            uv = center_h[:, :2] / np.where(
                np.abs(center_h[:, 2:]) > 1e-15, center_h[:, 2:], 1.0
            )
            center_ok = (center_h[:, 2] > self.head_near) & np.isfinite(uv).all(-1)
            safe_uv = np.where(center_ok[:, None], uv, [-500.0, -500.0])
            offsets = np.floor(safe_uv - 60 + 0.5).astype(np.int64)
            C = np.broadcast_to(np.diag([224 / 120, 224 / 120, 1.0]), (B, 3, 3)).copy()
            C[:, :2, 2] = -offsets * 224 / 120
            Pc = C @ P
            (auv, am) = project(Pc, anchors, 224, 224)
            (_, ar) = project(P, anchors, 320, 240)
            am &= ar & center_ok[:, None]
            st = self.static[arm]
            xyz = np.broadcast_to(st["xyz"], (B, 256, 16, 3))
            rho = np.broadcast_to(st["rho"], (B, 256, 16))
            normalized_xyz = (xyz - self.center) / self.half
            result[arm] = {
                "anchor_uv": (auv / 112 - 1).astype(np.float32),
                "anchor_valid": am,
                "anchor_xyz": anchor_code.astype(np.float32),
                "volume_grid": normalized_xyz.astype(np.float32),
                "depth_code": np.concatenate(
                    (normalized_xyz, rho[..., None] / 0.2), -1
                ).astype(np.float32),
                "ray_code": np.concatenate(
                    (np.broadcast_to(st["o"], (256, 3)), st["d"]), -1
                )[None]
                .repeat(B, 0)
                .astype(np.float32),
                "pose_code": np.concatenate(
                    (T[:, :3, 3], T[:, :3, :3].reshape(B, 9)), -1
                ).astype(np.float32),
                "valid_ray": np.broadcast_to(st["valid"], (B, 256)).copy(),
                "offsets": offsets,
                "P_saved": P,
                "P_crop": Pc,
                "xyz": xyz,
                "rho": rho,
            }
        return result

    def crop(self, image, raw_state, arm):
        if arm not in ARMS:
            raise ValueError("Arm must be left or right")
        state = np.asarray(raw_state)
        if state.shape != (14,):
            raise ValueError("One raw native state[14] required")
        return _crop_offsets(image, self.batch(state[None])[arm]["offsets"][0])

    @staticmethod
    def crop_heads(heads, geometry):
        return crop_heads(heads, geometry)

    @staticmethod
    def tensor_fields(g):
        if set(g) == set(ARMS):
            return {arm: Geometry.tensor_fields(g[arm]) for arm in ARMS}
        return {key: g[key] for key in TENSOR_FIELDS}
