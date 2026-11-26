"""OpenCV five-coefficient camera projection and guarded backprojection in pixel-edge coordinates."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

NEAR_M = 1e-06


def distort_normalized(xy_u: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Undistorted normalized [...,2] -> distorted normalized [...,2] (OpenCV order k1,k2,p1,p2,k3)."""
    (k1, k2, p1, p2, k3) = (float(v) for v in np.asarray(D, np.float64).reshape(5))
    xy = np.asarray(xy_u, np.float64)
    (x, y) = (xy[..., 0], xy[..., 1])
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    return np.stack(
        [
            x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        ],
        -1,
    )


def _jacobian(x, y, D):
    """Analytic d(x_d, y_d)/d(x, y) of distort_normalized."""
    (k1, k2, p1, p2, k3) = (float(v) for v in np.asarray(D, np.float64).reshape(5))
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    drad = k1 + 2 * k2 * r2 + 3 * k3 * r2**2
    j11 = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    j12 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    j21 = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    j22 = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    return (j11, j12, j21, j22)


def fold_radius(D: np.ndarray) -> tuple[float, float]:
    """Return the positive monotonic radial branch and its distorted limit."""
    (k1, k2, p1, p2, k3) = (float(v) for v in np.asarray(D, np.float64).reshape(5))
    roots = [
        x.real
        for x in np.roots([7 * k3, 5 * k2, 3 * k1, 1])
        if abs(x.imag) < 1e-12 and x.real > 0
    ]
    if not roots:
        return (float("inf"), float("inf"))
    rc = float(np.sqrt(min(roots)))
    return (rc, float(rc * (1 + k1 * rc**2 + k2 * rc**4 + k3 * rc**6)))


@dataclass(frozen=True)
class OpenCV5Camera:
    """K_edge: 3x3 pixel-edge intrinsics (cx, cy already +0.5). size_wh: native (width, height)."""

    K_edge: np.ndarray
    D: np.ndarray
    size_wh: tuple[int, int]
    r_u_crit: float
    r_d_max: float
    name: str = field(default="camera", compare=False)

    def __post_init__(self):
        K = np.asarray(self.K_edge, np.float64)
        D = np.asarray(self.D, np.float64).reshape(-1)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 1] != 0
            or (K[1, 0] != 0)
            or (tuple(K[2]) != (0.0, 0.0, 1.0))
            or (K[0, 0] <= 0)
            or (K[1, 1] <= 0)
        ):
            raise ValueError(
                f"{self.name}: K_edge must be a zero-skew finite [3,3] intrinsic matrix"
            )
        if D.shape != (5,) or not np.isfinite(D).all():
            raise ValueError(
                f"{self.name}: D must be OpenCV pinhole-5 [k1,k2,p1,p2,k3]"
            )
        if len(self.size_wh) != 2 or any(
            (type(v) is not int or v <= 0 for v in self.size_wh)
        ):
            raise ValueError(f"{self.name}: size_wh must be positive integers")
        if not (self.r_u_crit > 0 and self.r_d_max > 0):
            raise ValueError(f"{self.name}: fold radii must be positive")
        object.__setattr__(self, "K_edge", K.copy())
        object.__setattr__(self, "D", D.copy())
        object.__setattr__(
            self, "size_wh", (int(self.size_wh[0]), int(self.size_wh[1]))
        )

    @property
    def size_hw(self) -> tuple[int, int]:
        return (self.size_wh[1], self.size_wh[0])

    def normalized_to_pixel(self, xy_d: np.ndarray) -> np.ndarray:
        K = self.K_edge
        xy = np.asarray(xy_d, np.float64)
        return np.stack(
            [K[0, 0] * xy[..., 0] + K[0, 2], K[1, 1] * xy[..., 1] + K[1, 2]], -1
        )

    def pixel_to_normalized(self, uv_edge: np.ndarray) -> np.ndarray:
        K = self.K_edge
        uv = np.asarray(uv_edge, np.float64)
        return np.stack(
            [(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1]], -1
        )

    def in_image(self, uv_edge: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv_edge, np.float64)
        (w, h) = self.size_wh
        with np.errstate(invalid="ignore"):
            return (
                np.isfinite(uv).all(-1)
                & (uv[..., 0] >= 0)
                & (uv[..., 0] < w)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] < h)
            )

    def project(
        self, P_cam: np.ndarray, near: float = NEAR_M, *, guard: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points [...,3] -> (uv_edge [...,2], valid [...]).

        valid = finite & z > near & r_u < r_u_crit. Invalid uv are NaN. In-image is NOT part of
        valid (callers test bounds explicitly). guard=False exists only for the negative control
        test that proves the fold guard is needed; runtime code never passes it.
        """
        P = np.asarray(P_cam, np.float64)
        if P.shape[-1] != 3:
            raise ValueError("Camera points must be [...,3]")
        z = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_z = np.where(np.abs(z) > 1e-300, z, 1.0)
            xy = P[..., :2] / safe_z[..., None]
            r_u = np.hypot(xy[..., 0], xy[..., 1])
            finite = np.isfinite(P).all(-1)
            valid = (
                finite & (z > near) & (r_u < self.r_u_crit)
                if guard
                else finite & (z > near)
            )
            uv = self.normalized_to_pixel(distort_normalized(xy, self.D))
        uv = np.where(valid[..., None], uv, np.nan)
        return (uv, valid)

    def unproject(
        self,
        uv_edge: np.ndarray,
        *,
        iters: int = 50,
        roundtrip_px: float = 0.05,
        fixed_point_iters: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-edge uv [...,2] -> (undistorted normalized xy [...,2] (z = 1), valid [...]).

        valid = finite & r_d <= r_d_max & solution r_u < r_u_crit & round trip < roundtrip_px.
        Newton (analytic Jacobian) first; points that do not converge restart from the distorted
        coordinates with the OpenCV-style fixed point iteration. Invalid xy are NaN.
        """
        uv = np.asarray(uv_edge, np.float64)
        if uv.shape[-1] != 2:
            raise ValueError("Pixel coordinates must be [...,2]")
        shape = uv.shape[:-1]
        flat = uv.reshape(-1, 2)
        xd = self.pixel_to_normalized(flat)
        finite = np.isfinite(xd).all(-1)
        xd = np.where(finite[:, None], xd, 0.0)
        r_d = np.hypot(xd[:, 0], xd[:, 1])
        (x, y) = (xd[:, 0].copy(), xd[:, 1].copy())
        converged = np.zeros(len(xd), bool)
        with np.errstate(all="ignore"):
            for _ in range(int(iters)):
                f = distort_normalized(np.stack([x, y], -1), self.D)
                (ex, ey) = (f[:, 0] - xd[:, 0], f[:, 1] - xd[:, 1])
                (j11, j12, j21, j22) = _jacobian(x, y, self.D)
                det = j11 * j22 - j12 * j21
                dx = (j22 * ex - j12 * ey) / det
                dy = (-j21 * ex + j11 * ey) / det
                step = np.isfinite(dx) & np.isfinite(dy) & ~converged
                x = np.where(step, x - dx, x)
                y = np.where(step, y - dy, y)
                converged |= step & (np.hypot(dx, dy) < 1e-15)
                if converged.all():
                    break
            residual = np.hypot(
                *(distort_normalized(np.stack([x, y], -1), self.D) - xd).T
            )
            newton_ok = converged | np.isfinite(residual) & (residual < 1e-12)
            redo = ~newton_ok
            if redo.any():
                (k1, k2, p1, p2, k3) = self.D
                (fx, fy) = (xd[redo, 0].copy(), xd[redo, 1].copy())
                (x0, y0) = (fx.copy(), fy.copy())
                for _ in range(int(fixed_point_iters)):
                    r2 = fx * fx + fy * fy
                    icdist = 1.0 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
                    deltax = 2 * p1 * fx * fy + p2 * (r2 + 2 * fx * fx)
                    deltay = p1 * (r2 + 2 * fy * fy) + 2 * p2 * fx * fy
                    fx = (x0 - deltax) * icdist
                    fy = (y0 - deltay) * icdist
                (x[redo], y[redo]) = (fx, fy)
            xy = np.stack([x, y], -1)
            back = self.normalized_to_pixel(distort_normalized(xy, self.D))
            err = np.hypot(*(back - flat).T)
            r_u = np.hypot(x, y)
            valid = (
                finite
                & (r_d <= self.r_d_max)
                & np.isfinite(xy).all(-1)
                & (r_u < self.r_u_crit)
                & (err < roundtrip_px)
            )
        xy = np.where(valid[:, None], xy, np.nan)
        return (xy.reshape(*shape, 2), valid.reshape(shape))

    def ray_directions(
        self, uv_edge: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit camera-frame directions [...,3] and validity. Invalid directions are NaN."""
        (xy, valid) = self.unproject(uv_edge, **kwargs)
        d = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return (d, valid)

    def specification(self) -> dict:
        return {
            "name": self.name,
            "K_edge": self.K_edge.tolist(),
            "D": self.D.tolist(),
            "size_wh": list(self.size_wh),
            "r_u_crit": self.r_u_crit,
            "r_d_max": self.r_d_max,
            "pixel_convention": "edge",
        }


def camera_from_calibration(entry: dict, name: str) -> OpenCV5Camera:
    """Construct a camera from explicit intrinsics, distortion, raster size and radial limits."""
    return OpenCV5Camera(
        K_edge=np.asarray(entry["K_edge"], np.float64),
        D=np.asarray(entry["D"], np.float64),
        size_wh=tuple((int(v) for v in entry["size_wh"])),
        r_u_crit=float(entry["r_u_crit"]),
        r_d_max=float(entry["r_d_max"]),
        name=name,
    )
