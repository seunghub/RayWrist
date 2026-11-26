"""LIBERO pi_0.5 geometry with the native 224-pixel SigLIP raster."""

from pathlib import Path
import json
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image


class Geometry:

    def __init__(self, cameras_path=None):
        cameras_path = (
            Path(cameras_path)
            if cameras_path is not None
            else Path(__file__).resolve().parents[1] / "configs/libero_cameras.json"
        )
        self.cams = json.loads(cameras_path.read_text())
        self.cal = {k: v["wrist_calibration"] for (k, v) in self.cams.items()}
        self.lower = np.array([-0.06, -0.06, -0.05])
        self.upper = np.array([0.06, 0.06, 0.15])
        self.center = (self.lower + self.upper) / 2
        self.half = (self.upper - self.lower) / 2
        flip = np.array([[-1.0, 0, 256], [0, 1.0, 0], [0, 0, 1.0]])
        (yy, xx) = np.meshgrid(
            np.arange(16) * 14 + 7.0, np.arange(16) * 14 + 7.0, indexing="ij"
        )
        centers = np.stack([xx.ravel(), yy.ravel()], -1)
        saved = np.c_[centers * 256 / 224, np.ones(256)]
        self.static = {}
        self.P = {}
        self.M = {}
        axes = [np.linspace(self.lower[a], self.upper[a], 17)[1::2] for a in range(3)]
        (zz, yy, xx) = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        self.anchors = np.stack([xx, yy, zz], -1).reshape(512, 3)
        for t in sorted(map(int, self.cams)):
            cam = self.cams[str(t)]
            cal = self.cal[str(t)]
            K = (
                np.asarray(cam.get("K", cal["wrist_K_256"]))
                if "K" in cam
                else np.array(
                    [
                        [128 / np.tan(np.deg2rad(cam["fovy_deg"]) / 2), 0, 128],
                        [0, 128 / np.tan(np.deg2rad(cam["fovy_deg"]) / 2), 128],
                        [0, 0, 1],
                    ]
                )
            )
            self.P[t] = (
                flip @ K @ np.linalg.inv(np.asarray(cam["T_cam2world_optical"]))[:3]
            )
            self.M[t] = np.asarray(cam["M"])
            T = np.asarray(cal["T_EE_from_wrist_optical"])
            o = T[:3, 3]
            d = saved @ flip.T @ np.linalg.inv(np.asarray(cal["wrist_K_256"])).T
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            d = d @ T[:3, :3].T
            lo = np.full(256, 1e-06)
            hi = np.full(256, np.inf)
            valid = np.ones(256, bool)
            for a in range(3):
                parallel = abs(d[:, a]) < 1e-14
                valid &= ~(parallel & ((o[a] < self.lower[a]) | (o[a] > self.upper[a])))
                denominator = np.where(parallel, 1, d[:, a])
                s0 = (self.lower[a] - o[a]) / denominator
                s1 = (self.upper[a] - o[a]) / denominator
                lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(s0, s1)))
                hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(s0, s1)))
            valid &= np.isfinite(hi) & (hi >= lo)
            rho = np.where(
                valid[:, None],
                lo[:, None] + (hi - lo)[:, None] * (np.arange(16) + 0.5) / 16,
                0,
            )
            xyz = o[None, None] + rho[..., None] * d[:, None]
            self.static[t] = dict(o=o, d=d, valid=valid, rho=rho, xyz=xyz)

    def batch(self, tasks, states):
        tasks = np.asarray(tasks, int)
        states = np.asarray(states, np.float64)
        B = len(tasks)
        if states.shape != (B, 8) or not np.isfinite(states).all():
            raise ValueError("physical EE state must be Bx8")
        R = Rotation.from_rotvec(states[:, 3:6]).as_matrix()
        T = np.zeros((B, 4, 4))
        T[:, 3, 3] = 1
        T[:, :3, :3] = R
        T[:, :3, 3] = states[:, :3]
        P = np.stack([self.P[int(t)] for t in tasks]) @ T
        h = np.einsum(
            "bij,bj->bi",
            np.stack([self.M[int(t)] for t in tasks]),
            np.c_[states[:, :3], np.ones(B)],
        )
        uv = h[:, :2] / h[:, 2:3]
        offsets = np.stack(
            [np.floor(256 - uv[:, 0] - 48 + 0.5), np.floor(uv[:, 1] - 48 + 0.5)], 1
        ).astype(np.int64)
        C = np.zeros((B, 3, 3))
        C[:, 0, 0] = C[:, 1, 1] = 224 / 96
        C[:, 2, 2] = 1
        C[:, 0, 2] = -offsets[:, 0] * 224 / 96
        C[:, 1, 2] = -offsets[:, 1] * 224 / 96
        Pc = C @ P

        def project(mat, xyz, side):
            h = np.einsum(
                "bij,b...j->b...i",
                mat,
                np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], -1),
            )
            pix = h[..., :2] / np.where(abs(h[..., 2:]) > 1e-15, h[..., 2:], 1)
            ok = (
                (h[..., 2] > 1e-06)
                & (pix[..., 0] >= 0)
                & (pix[..., 0] < side)
                & (pix[..., 1] >= 0)
                & (pix[..., 1] < side)
            )
            return (pix, ok)

        anchors = np.broadcast_to(self.anchors, (B, 512, 3))
        (auv, am) = project(Pc, anchors, 224)
        (_, ar) = project(P, anchors, 256)
        am &= ar
        st = [self.static[int(t)] for t in tasks]
        xyz = np.stack([x["xyz"] for x in st])
        rho = np.stack([x["rho"] for x in st])
        valid = np.stack([x["valid"] for x in st])
        o = np.stack([x["o"] for x in st])
        d = np.stack([x["d"] for x in st])
        return {
            "anchor_uv": (auv / 112 - 1).astype(np.float32),
            "anchor_valid": am,
            "anchor_xyz": ((anchors - self.center) / self.half).astype(np.float32),
            "volume_grid": ((xyz - self.center) / self.half).astype(np.float32),
            "depth_code": np.concatenate(
                [(xyz - self.center) / self.half, rho[..., None] / 0.2], -1
            ).astype(np.float32),
            "ray_code": np.concatenate(
                [np.broadcast_to(o[:, None], (B, 256, 3)), d], -1
            ).astype(np.float32),
            "pose_code": np.concatenate([states[:, :3], R.reshape(B, 9)], -1).astype(
                np.float32
            ),
            "valid_ray": valid,
            "offsets": offsets,
            "P_saved": P,
            "P_crop": Pc,
            "xyz": xyz,
            "rho": rho,
        }

    def crop(self, image, offset):
        if image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError("source RGB must be uint8 256x256")
        (c, r) = map(int, offset)
        return np.asarray(
            Image.fromarray(image)
            .crop((c, r, c + 96, r + 96))
            .resize((256, 256), Image.Resampling.BILINEAR)
            .resize((224, 224), Image.Resampling.BILINEAR)
        )

    @staticmethod
    def tensor_fields(g):
        return {
            k: v
            for (k, v) in g.items()
            if k not in ("offsets", "P_saved", "P_crop", "xyz", "rho")
        }
