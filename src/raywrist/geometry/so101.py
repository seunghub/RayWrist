"""SO-101 tool geometry using physical joints, camera distortion and explicit preprocessing affines."""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .camera import camera_from_calibration

SCHEMA = "so101-raygrid-geometry-v1"
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
HW_FIELDS = (
    "source_hw",
    "source_early_hw",
    "target_hw",
    "source_image_hw",
    "target_image_hw",
)
VIEW_KEYS = ("left_ego", "left_wrist", "right_wrist")
NATIVE_HW = {
    "left_ego": (720, 1280),
    "left_wrist": (480, 640),
    "right_wrist": (480, 640),
}
STATE_DIM = 12


def _hw(value, name):
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any((type(x) is not int or x <= 0 for x in value))
    ):
        raise ValueError(f"{name} must be positive static (height,width) integers")
    return tuple(value)


def _states(raw):
    value = np.asarray(raw, np.float64)
    if value.ndim != 2 or value.shape[1] != STATE_DIM or (not np.isfinite(value).all()):
        raise ValueError(
            "Geometry needs finite unnormalized LeRobot-degree joints [B,12], before dropout"
        )
    return value


def _affine(value, batch, name):
    result = np.asarray(value, np.float64)
    if result.shape == (3, 3):
        result = np.broadcast_to(result, (batch, 3, 3)).copy()
    if result.shape != (batch, 3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [3,3] or [B,3,3]")
    if not np.allclose(result[:, 2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be an affine in pixel-edge coordinates")
    if np.any(np.abs(np.linalg.det(result[:, :2, :2])) <= 1e-14):
        raise ValueError(f"{name} must be invertible")
    return result


def transform_pixels(affine, uv):
    """Apply batch affine to [B,...,2], retaining explicit pixel-edge convention."""
    points = np.asarray(uv, np.float64)
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("Pixel coordinates must be [B,...,2]")
    matrix = _affine(affine, len(points), "pixel_affine")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), -1)
    result = np.einsum("bij,b...j->b...i", matrix, homogeneous)
    return result[..., :2] / result[..., 2:]


def resize_affine(source_hw, destination_hw):
    """Exact pixel-edge resize affine, with no invented center offset."""
    (height, width) = _hw(source_hw, "source_hw")
    (new_height, new_width) = _hw(destination_hw, "destination_hw")
    return np.diag([new_width / width, new_height / height, 1.0]).astype(np.float64)


def token_centers(target_hw=(8, 8), image_hw=(256, 256)):
    """Raster-major query centers of actual native target image coordinates."""
    (rows, columns) = _hw(target_hw, "target_hw")
    (height, width) = _hw(image_hw, "image_hw")
    (yy, xx) = np.meshgrid(
        (np.arange(rows) + 0.5) * height / rows,
        (np.arange(columns) + 0.5) * width / columns,
        indexing="ij",
    )
    return np.stack((xx.ravel(), yy.ravel()), -1)


def Rz_deg(delta_deg):
    a = math.radians(delta_deg)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return T


class SO101Geometry:

    def __init__(self, calibration_path):
        self.calibration_path = Path(calibration_path)
        contents = self.calibration_path.read_bytes()
        self.calibration_sha256 = hashlib.sha256(contents).hexdigest()
        self.calibration = json.loads(contents)
        cal = self.calibration
        if cal.get("schema") != SCHEMA or cal.get("world") != "external_optical":
            raise ValueError(
                "SO-101 calibration in the external optical frame is required"
            )
        self.world_from_ext = np.asarray(cal["T_world_from_ext"], np.float64)
        if not np.array_equal(self.world_from_ext, np.eye(4)):
            raise ValueError(
                "World must be the external optical frame (T_world_from_ext = I)"
            )
        self.ext_from_world = np.linalg.inv(self.world_from_ext)
        self.near = float(cal["near_m"])
        self.roll_delta_deg = float(cal["roll_delta_deg"])
        self.lower = np.asarray(cal["volume"]["lower"], np.float64)
        self.upper = np.asarray(cal["volume"]["upper"], np.float64)
        (self.center, self.half) = (
            (self.lower + self.upper) / 2,
            (self.upper - self.lower) / 2,
        )
        if np.any(self.half <= 0) or cal["volume"]["frame"] != "tool":
            raise ValueError("Invalid physical volume bounds")
        self.head = camera_from_calibration(cal["cameras"]["external"], "external")
        self.wrists = {
            arm: camera_from_calibration(cal["cameras"][f"{arm}_wrist"], f"{arm}_wrist")
            for arm in ARMS
        }
        for view, camera in zip(
            VIEW_KEYS, (self.head, self.wrists["left"], self.wrists["right"])
        ):
            if camera.size_hw != NATIVE_HW[view]:
                raise ValueError(
                    f"{view}: calibration size {camera.size_hw} != {NATIVE_HW[view]}"
                )
        self.native_hw = {view: tuple(NATIVE_HW[view]) for view in VIEW_KEYS}
        self.joints = [
            (np.asarray(j["origin"], np.float64), np.asarray(j["axis"], np.float64))
            for j in cal["joints"]
        ]
        if [j["name"] for j in cal["joints"]] != [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]:
            raise ValueError("Unexpected SO101 joint chain")
        (self.arms, self.link_from_tool, self.tool_from_link) = ({}, {}, {})
        for arm in ARMS:
            a = cal["arms"][arm]
            X_fit = np.asarray(a["X_fit"], np.float64)
            X_true = np.asarray(a["X_true"], np.float64)
            if np.abs(Rz_deg(-self.roll_delta_deg) @ X_fit - X_true).max() > 1e-15:
                raise ValueError(
                    f"{arm}: X_true != Rz(-delta) X_fit; wrist_roll delta and hand-eye are out of sync"
                )
            self.arms[arm] = dict(
                state_offset=int(a["state_offset"]),
                offsets_deg=np.asarray(a["joint_offsets_deg"], np.float64),
                base=self.world_from_ext @ np.asarray(a["T_ext_from_base"], np.float64),
                X_true=X_true,
            )
            self.link_from_tool[arm] = np.asarray(a["T_gripper_from_tool"], np.float64)
            self.tool_from_link[arm] = np.linalg.inv(self.link_from_tool[arm])
        axes = [np.linspace(self.lower[a], self.upper[a], 17)[1::2] for a in range(3)]
        (zz, yy, xx) = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        self.anchors = np.stack((xx, yy, zz), -1).reshape(512, 3)

    @staticmethod
    def _arm(arm):
        if arm not in ARMS:
            raise ValueError("Arm must be left or right")
        return arm

    def q_urdf(self, state12_deg, arm):
        """[B,5] URDF radians: deg2rad(state + offsets), wrist_roll + delta. Never clamped."""
        states = _states(state12_deg)
        a = self.arms[self._arm(arm)]
        o = a["state_offset"]
        q = np.deg2rad(states[:, o : o + 5] + a["offsets_deg"])
        q[:, 4] += np.deg2rad(self.roll_delta_deg)
        return q

    def T_world_from_gripper(self, state12_deg, arm):
        """Exact world<-gripper_link FK (RoboTwin fk_ee product form)."""
        q = self.q_urdf(state12_deg, arm)
        batch = len(q)
        value = np.broadcast_to(self.arms[arm]["base"], (batch, 4, 4)).copy()
        for (origin, axis), angle in zip(self.joints, q.T):
            motion = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
            motion[:, :3, :3] = Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
            value = value @ origin @ motion
        return value

    def T_world_from_tool(self, state12_deg, arm):
        return (
            self.T_world_from_gripper(state12_deg, arm)
            @ self.link_from_tool[self._arm(arm)]
        )

    def T_world_from_wristcam(self, state12_deg, arm):
        return (
            self.T_world_from_gripper(state12_deg, arm)
            @ self.arms[self._arm(arm)]["X_true"]
        )

    def T_tool_from_wristcam(self, arm):
        """State independent: the wrist camera and the TCP are both rigid on gripper_link."""
        return self.tool_from_link[self._arm(arm)] @ self.arms[arm]["X_true"]

    def project_tcp(self, state12_deg, arm):
        """Raw external pixel-edge TCP uv for choosing a crop on replayed ego RGB.

        uv_raw_edge is NaN where the projection is invalid (behind / beyond the fold).
        """
        tool = self.T_world_from_tool(state12_deg, arm)
        ext = np.einsum("ij,bj->bi", self.ext_from_world, tool[:, :, 3])
        (uv, valid) = self.head.project(ext[:, :3], self.near)
        return {
            "uv_raw_edge": uv,
            "uv": uv,
            "valid": valid,
            "in_image": self.head.in_image(uv),
            "depth": ext[:, 2],
            "world_xyz": tool[:, :3, 3],
            "world_from_tool": tool,
        }

    def batch(self, state12_deg):
        """Physical crop centers only; use build_context for neural ray tensors."""
        return {side: self.project_tcp(state12_deg, side) for side in ARMS}

    def _rays(self, side, wrist_affine, target_hw, target_image_hw):
        batch = len(wrist_affine)
        centers = np.broadcast_to(
            token_centers(target_hw, target_image_hw), (batch, math.prod(target_hw), 2)
        )
        raw_uv = transform_pixels(np.linalg.inv(wrist_affine), centers)
        cam = self.wrists[side]
        mount = self.tool_from_link[side] @ self.arms[side]["X_true"]
        origin = mount[:3, 3]
        (xy_u, inverse_ok) = cam.unproject(raw_uv)
        xy = np.where(inverse_ok[..., None], xy_u, cam.pixel_to_normalized(raw_uv))
        dc = np.concatenate((xy, np.ones((*raw_uv.shape[:-1], 1))), -1)
        dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
        direction = dc @ mount[:3, :3].T
        near_rho = self.near / dc[..., 2]
        (lo, hi) = (near_rho.copy(), np.full(raw_uv.shape[:-1], np.inf))
        (width, height) = cam.size_wh
        valid = (
            (raw_uv[..., 0] >= 0)
            & (raw_uv[..., 0] < width)
            & (raw_uv[..., 1] >= 0)
            & (raw_uv[..., 1] < height)
            & inverse_ok
            & (dc[..., 2] > 0)
        )
        for axis in range(3):
            parallel = np.abs(direction[..., axis]) < 1e-14
            valid &= ~(
                parallel
                & (
                    (origin[axis] < self.lower[axis])
                    | (origin[axis] > self.upper[axis])
                )
            )
            denominator = np.where(parallel, 1.0, direction[..., axis])
            (s0, s1) = (
                (self.lower[axis] - origin[axis]) / denominator,
                (self.upper[axis] - origin[axis]) / denominator,
            )
            lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(s0, s1)))
            hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(s0, s1)))
        valid &= np.isfinite(hi) & (hi >= lo)
        (safe_lo, safe_hi) = (np.where(valid, lo, 0.0), np.where(valid, hi, 0.0))
        rho = (
            safe_lo[..., None]
            + (safe_hi - safe_lo)[..., None] * (np.arange(16) + 0.5) / 16
        )
        xyz = origin[None, None, None] + rho[..., None] * direction[:, :, None]
        return dict(
            o=origin,
            d=direction,
            raw_uv=raw_uv,
            valid=valid,
            rho=rho,
            xyz=xyz,
            near_rho=near_rho,
            rho_min=lo,
            rho_max=hi,
            inverse_ok=inverse_ok,
            dc=dc,
        )

    def build_context(
        self,
        state12_deg,
        arm,
        *,
        crop_affine,
        wrist_affine,
        head_affine,
        source_early_hw=(16, 16),
        source_hw=(8, 8),
        target_hw=(8, 8),
        source_image_hw=(256, 256),
        target_image_hw=(256, 256),
        head_image_hw=(256, 256),
    ):
        """Build geometry for actual source/target processor coordinates.

        crop_affine: raw ego 1280x720 -> final Qwen crop pixel-edge coordinates; includes the
          letterbox replay, the raw480 crop translation and the Qwen resize (101 -> 256).
        wrist_affine: raw wrist 640x480 -> final native wrist slot coordinates; includes the
          letterbox replay and Qwen resize (inverse maps each target query ray).
        head_affine/head_image_hw: raw ego -> observed augmented ego canvas (256x256), used to
          reject anchors beyond that canvas. No surface visibility is inferred.
        source_early_hw is patch raster before2x2 merging; source_hw after it.
        """
        states = _states(state12_deg)
        side = self._arm(arm)
        batch = len(states)
        source_early_hw = _hw(source_early_hw, "source_early_hw")
        source_hw = _hw(source_hw, "source_hw")
        source_image_hw = _hw(source_image_hw, "source_image_hw")
        target_hw = _hw(target_hw, "target_hw")
        target_image_hw = _hw(target_image_hw, "target_image_hw")
        head_image_hw = _hw(head_image_hw, "head_image_hw")
        if tuple((x * 2 for x in source_hw)) != source_early_hw:
            raise ValueError(
                "Qwen crop merger raster must be half each early patch dimension"
            )
        if tuple((x * 16 for x in source_early_hw)) != source_image_hw:
            raise ValueError(
                "Final Qwen crop pixel dimensions and16px patch raster differ"
            )
        if tuple((x * 32 for x in target_hw)) != target_image_hw:
            raise ValueError(
                "Native wrist query raster and32px merged-token spacing differ"
            )
        crop_affine = _affine(crop_affine, batch, "crop_affine")
        wrist_affine = _affine(wrist_affine, batch, "wrist_affine")
        head_affine = _affine(head_affine, batch, "head_affine")
        tool = self.T_world_from_tool(states, side)
        projection = self.ext_from_world[None] @ tool
        anchors = np.broadcast_to(self.anchors, (batch, 512, 3))
        homogeneous = np.concatenate((anchors, np.ones((batch, 512, 1))), -1)
        image = np.einsum("bij,bnj->bni", projection, homogeneous)
        (raw_uv, projected) = self.head.project(image[..., :3], self.near)
        crop_uv = transform_pixels(crop_affine, raw_uv)
        augmented_uv = transform_pixels(head_affine, raw_uv)
        (source_h, source_w) = source_image_hw
        (head_h, head_w) = head_image_hw
        (raw_w, raw_h) = self.head.size_wh
        (center_uv, center_ok) = self.head.project(projection[:, :3, 3], self.near)
        with np.errstate(invalid="ignore"):
            valid = (
                projected
                & np.isfinite(raw_uv).all(-1)
                & (raw_uv[..., 0] >= 0)
                & (raw_uv[..., 0] < raw_w)
                & (raw_uv[..., 1] >= 0)
                & (raw_uv[..., 1] < raw_h)
                & (crop_uv[..., 0] >= 0)
                & (crop_uv[..., 0] < source_w)
                & (crop_uv[..., 1] >= 0)
                & (crop_uv[..., 1] < source_h)
                & (augmented_uv[..., 0] >= 0)
                & (augmented_uv[..., 0] < head_w)
                & (augmented_uv[..., 1] >= 0)
                & (augmented_uv[..., 1] < head_h)
                & center_ok[:, None]
            )
        rays = self._rays(side, wrist_affine, target_hw, target_image_hw)
        normalized_xyz = (rays["xyz"] - self.center) / self.half
        count = math.prod(target_hw)
        anchor_uv = np.where(
            valid[..., None], crop_uv / np.array([source_w / 2, source_h / 2]) - 1, 0.0
        )
        result = {
            "anchor_uv": anchor_uv.astype(np.float32),
            "anchor_valid": valid,
            "anchor_xyz": np.broadcast_to(
                (self.anchors - self.center) / self.half, (batch, 512, 3)
            ).astype(np.float32),
            "volume_grid": normalized_xyz.astype(np.float32),
            "depth_code": np.concatenate(
                (normalized_xyz, rays["rho"][..., None] / 0.2), -1
            ).astype(np.float32),
            "ray_code": np.concatenate(
                (np.broadcast_to(rays["o"], (batch, count, 3)), rays["d"]), -1
            ).astype(np.float32),
            "pose_code": np.concatenate(
                (tool[:, :3, 3], tool[:, :3, :3].reshape(batch, 9)), -1
            ).astype(np.float32),
            "valid_ray": rays["valid"],
            "side": side,
            "source_hw": source_hw,
            "source_early_hw": source_early_hw,
            "target_hw": target_hw,
            "source_image_hw": source_image_hw,
            "target_image_hw": target_image_hw,
            "crop_affine": crop_affine,
            "wrist_affine": wrist_affine,
            "head_affine": head_affine,
            "raw_target_uv": rays["raw_uv"],
            "rho": rays["rho"],
            "xyz": rays["xyz"],
            "rho_min": rays["rho_min"],
            "rho_max": rays["rho_max"],
            "ray_dc": rays["dc"],
            "raw_anchor_uv": raw_uv,
            "crop_anchor_uv": crop_uv,
            "tcp_raw_uv": center_uv,
            "tcp_valid": center_ok,
            "world_from_tool": tool,
            "calibration_sha256": self.calibration_sha256,
            "surface_visibility_measured": False,
        }
        if not all(
            (
                np.isfinite(result[name]).all()
                for name in TENSOR_FIELDS
                if name not in ("anchor_valid", "valid_ray")
            )
        ):
            raise ValueError("Nonfinite physical geometry")
        return result

    @staticmethod
    def tensor_fields(context):
        return {key: context[key] for key in (*TENSOR_FIELDS, *HW_FIELDS, "side")}


def build_context(state12_deg, arm, *, geometry=None, calibration_path=None, **kwargs):
    if geometry is None:
        if calibration_path is None:
            raise ValueError("Pass an SO101Geometry or an explicit calibration_path")
        geometry = SO101Geometry(calibration_path)
    return geometry.build_context(state12_deg, arm, **kwargs)
