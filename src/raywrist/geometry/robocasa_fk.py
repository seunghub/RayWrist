"""Observed-state GR1 forward kinematics and supplied head-image projection."""

from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .crop import zoom_crop


def transform(pos, quat_wxyz):
    out = np.eye(4)
    q = np.asarray(quat_wxyz, np.float64)
    out[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = pos
    return out


class GR1Geometry:

    def __init__(self, calibration=None):
        self.path = (
            Path(calibration)
            if calibration is not None
            else Path(__file__).resolve().parents[1]
            / "configs/robocasa_calibration.json"
        )
        self.cal = json.loads(self.path.read_text())
        assert self.cal["schema"] == "ROBOCASA_GR1_RAYWRIST_CALIBRATION_V1"
        c = self.cal["camera"]
        (h, w) = c["raw_hw"]
        f = h / (2 * np.tan(np.deg2rad(c["fovy_deg"]) / 2))
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        p = self.cal["processed_image"]
        (y0, y1, x0, x1) = p["crop_yxyx"]
        (rh, rw) = p["first_resize_hw"]
        (oh, ow) = p["final_resize_hw"]
        a = np.array(
            [
                [rw / (x1 - x0), 0, -x0 * rw / (x1 - x0)],
                [0, rh / (y1 - y0), -y0 * rh / (y1 - y0) + p["vertical_padding"]],
                [0, 0, 1.0],
            ]
        )
        self.pre_affine = np.diag([ow / rw, oh / rw, 1.0]) @ a

    def _fk(self, name, joints):
        T = np.eye(4)
        for body in self.cal["chains"][name]:
            T = T @ transform(body["pos"], body["quat"])
            for j in body["joints"]:
                angle = joints.get(j["name"], 0.0)
                motion = np.eye(4)
                if j["type"] == "hinge":
                    motion[:3, :3] = Rotation.from_rotvec(
                        np.asarray(j["axis"]) * angle
                    ).as_matrix()
                    p = np.asarray(j["pos"])
                    motion[:3, 3] = p - motion[:3, :3] @ p
                elif j["type"] == "slide":
                    motion[:3, 3] = np.asarray(j["axis"]) * angle
                else:
                    raise ValueError("Unsupported joint type")
                T = T @ motion
        return T

    def project(self, raw_state):
        state = np.asarray(raw_state, np.float64)
        if state.shape != (29,) or not np.isfinite(state).all():
            raise ValueError("Current RAW physical GR1 state[29] required")
        joints = {}
        for group in ("left", "right", "waist"):
            key = group + "_arm" if group != "waist" else "waist"
            joints.update(
                zip(self.cal["joint_order"][group], state[self.cal["state_order"][key]])
            )
        camera = self._fk("camera", joints) @ transform(
            self.cal["camera"]["pos"], self.cal["camera"]["quat_wxyz"]
        )
        camera[:3, :3] = camera[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        P = self.pre_affine @ self.K @ np.linalg.inv(camera)[:3]
        result = {}
        for arm in ("left", "right"):
            tool = self._fk(arm, joints)
            xyz = P @ tool[:, 3]
            valid = bool(np.isfinite(xyz).all() and xyz[2] > 1e-06)
            uv = xyz[:2] / xyz[2] if valid else np.array([-500.0, -500.0])
            result[arm] = dict(
                uv=uv,
                depth=float(xyz[2]),
                center_valid=valid,
                P_tool=P @ tool,
                tool_from_robot=tool,
            )
        return result

    def crops(self, headcam, raw_state):
        image = np.asarray(headcam)
        if image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError(
                "Decoded, pre-augmentation headcam uint8[256,256,3] required"
            )
        centers = self.project(raw_state)
        images = []
        audit = {}
        for arm in ("left", "right"):
            uv = centers[arm]["uv"]
            (crop, info) = zoom_crop(
                image, float(uv[0]), float(uv[1]), crop=96, out_hw=256
            )
            images.append(crop)
            audit[arm] = dict(
                uv=uv.tolist(),
                depth=centers[arm]["depth"],
                center_valid=centers[arm]["center_valid"],
                **info
            )
        return (images, audit)
