"""LIBERO GR00T geometry from fixed calibration and current physical end-effector pose."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .libero_transforms import (
    GeometryConfig,
    token_centers,
    encoder_from_saved,
    saved_from_optical_pixel,
    crop_from_saved,
)


class RayContext:

    def __init__(self, cameras, calibration=None, device="cpu"):
        self.arm = "RayGrid"
        self.device = device
        self.training = False
        self.cams = {str(k): v for (k, v) in cameras.items()}
        self.cal = {
            str(k): v
            for (k, v) in (
                calibration
                or {k: v["wrist_calibration"] for (k, v) in self.cams.items()}
            ).items()
        }
        if set(self.cal) != set(self.cams):
            raise ValueError("camera/mount task IDs differ")
        self.config = GeometryConfig()
        self.static = {}
        self.world_projection = {}
        self.camera_matrix = {}
        for key, cam in self.cams.items():
            t = int(key)
            K = np.asarray(cam["K"], dtype=np.float64)
            if (
                K.shape != (3, 3)
                or not np.isclose(K[0, 2], 128)
                or (not np.isclose(K[1, 2], 128))
            ):
                raise ValueError("native256 calibration required")
            self.world_projection[t] = (
                saved_from_optical_pixel()
                @ K
                @ np.linalg.inv(np.asarray(cam["T_cam2world_optical"]))[:3]
            )
            self.camera_matrix[t] = np.asarray(cam["M"], dtype=np.float64)
            if self.camera_matrix[t].shape != (4, 4):
                raise ValueError("native homogeneous4x4 projection required")
            self.static[t] = self._static(t)
        self.task_by_env = {
            "libero_sim/" + v["bddl"].removesuffix(".bddl"): int(k)
            for (k, v) in self.cams.items()
        }

    def _static(self, t):
        c = self.config
        T = np.asarray(self.cal[str(t)]["T_EE_from_wrist_optical"])
        o = T[:3, 3]
        h = (
            np.c_[token_centers(), np.ones(64)]
            @ np.linalg.inv(encoder_from_saved()).T
            @ saved_from_optical_pixel().T
        )
        d = h @ np.linalg.inv(np.asarray(self.cal[str(t)]["wrist_K_256"])).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d = d @ T[:3, :3].T
        lo = np.full(64, c.near_m)
        hi = np.full(64, np.inf)
        valid = np.ones(64, bool)
        for a in range(3):
            pa = abs(d[:, a]) < 1e-14
            valid &= ~(pa & ((o[a] < c.box_min[a]) | (o[a] > c.box_max[a])))
            v0 = (c.box_min[a] - o[a]) / np.where(pa, 1, d[:, a])
            v1 = (c.box_max[a] - o[a]) / np.where(pa, 1, d[:, a])
            lo = np.maximum(lo, np.where(pa, -np.inf, np.minimum(v0, v1)))
            hi = np.minimum(hi, np.where(pa, np.inf, np.maximum(v0, v1)))
        valid &= np.isfinite(hi) & (hi >= lo)
        rho = np.where(
            valid[:, None],
            lo[:, None] + (hi - lo)[:, None] * (np.arange(16)[None] + 0.5) / 16,
            0,
        )
        xyz = o[None, None] + rho[..., None] * d[:, None]
        axes = [np.linspace(c.box_min[a], c.box_max[a], 17)[1::2] for a in range(3)]
        (zz, yy, xx) = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        anchors = np.stack([xx, yy, zz], -1).reshape(512, 3)
        return dict(
            o=o, d=d, lo=lo, hi=hi, valid=valid, rho=rho, xyz=xyz, anchors=anchors
        )

    def numpy(self, tasks, states):
        tasks = np.asarray(tasks, dtype=np.int64)
        states = np.asarray(states, dtype=np.float64)
        if (
            states.ndim != 2
            or states.shape[1] < 6
            or (not np.isfinite(states[:, :6]).all())
        ):
            raise ValueError("bad physical pose")
        B = len(tasks)
        st = [self.static[int(t)] for t in tasks]
        R = Rotation.from_rotvec(states[:, 3:6]).as_matrix()
        T = np.zeros((B, 4, 4))
        T[:, 3, 3] = 1
        T[:, :3, :3] = R
        T[:, :3, 3] = states[:, :3]
        P = np.stack([self.world_projection[int(t)] for t in tasks]) @ T
        h = np.einsum(
            "bij,bj->bi",
            np.stack([self.camera_matrix[int(t)] for t in tasks]),
            np.c_[states[:, :3], np.ones(B)],
        )
        uv = h[:, :2] / h[:, 2:3]
        off = np.stack(
            [np.floor(256 - uv[:, 0] - 48 + 0.5), np.floor(uv[:, 1] - 48 + 0.5)], 1
        ).astype(np.int64)
        Pc = np.stack([crop_from_saved(c0, r0) for (c0, r0) in off]) @ P
        Pf = encoder_from_saved()[None] @ P
        xyz = np.stack([a["xyz"] for a in st])
        rho = np.stack([a["rho"] for a in st])
        valid = np.stack([a["valid"] for a in st])
        o = np.stack([a["o"] for a in st])
        d = np.stack([a["d"] for a in st])
        anchors = np.stack([a["anchors"] for a in st])

        def project(Pa, points):
            h = np.einsum(
                "bij,b...j->b...i",
                Pa,
                np.concatenate([points, np.ones((*points.shape[:-1], 1))], -1),
            )
            uv = h[..., :2] / np.where(abs(h[..., 2:]) > 1e-15, h[..., 2:], 1)
            ok = (
                (h[..., 2] > self.config.near_m)
                & (uv[..., 0] >= 0)
                & (uv[..., 0] < 256)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] < 256)
            )
            return (uv, ok)

        (uvraw, rawok) = project(P, xyz)
        (uvc, mc) = project(Pc, xyz)
        (uvf, mf) = project(Pf, xyz)
        mc &= rawok & valid[:, :, None]
        mf &= rawok & valid[:, :, None]
        (auv, am) = project(Pc, anchors)
        (_, ar) = project(P, anchors)
        am &= ar
        center = (np.asarray(self.config.box_min) + np.asarray(self.config.box_max)) / 2
        half = (np.asarray(self.config.box_max) - np.asarray(self.config.box_min)) / 2
        code = np.concatenate([(xyz - center) / half, rho[..., None] / 0.2], -1)
        pose = np.concatenate([states[:, :3], R.reshape(B, 9)], 1)
        out = {
            "uv_crop": uvc / 128 - 1,
            "uv_full": uvf / 128 - 1,
            "mask_crop": mc,
            "mask_full": mf,
            "valid_ray": valid,
            "depth_code": code,
            "ray_code": np.concatenate(
                [np.broadcast_to(o[:, None], (B, 64, 3)), d], -1
            ),
            "pose_code": pose,
            "anchor_uv": auv / 128 - 1,
            "anchor_valid": am,
            "anchor_xyz": (anchors - center) / half,
            "volume_grid": (xyz - center) / half,
            "rho": rho,
            "xyz": xyz,
            "offsets": off,
        }
        return out

    def tensor(self, n):
        out = {}
        for k, v in n.items():
            if k in ["offsets", "path_start", "path_end", "rho", "xyz"]:
                continue
            out[k] = torch.as_tensor(
                v,
                device=self.device,
                dtype=torch.bool if np.asarray(v).dtype == bool else torch.float32,
            )
        return out

    def crop(self, external, task, state):
        from PIL import Image

        image = np.asarray(external)
        if image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError("native saved external RGB256 required")
        (c0, r0) = self.numpy(
            [task], np.asarray(state, dtype=np.float64).reshape(1, -1)
        )["offsets"][0]
        return np.asarray(
            Image.fromarray(image)
            .crop((int(c0), int(r0), int(c0) + 96, int(r0) + 96))
            .resize((256, 256), Image.Resampling.BILINEAR)
        )
