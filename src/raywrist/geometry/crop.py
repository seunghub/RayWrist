"""Fixed-size, zero-padded tool-centered RGB crops."""

import numpy as np

CROP = 96
OUT_HW = 256


def zoom_window(col: float, row: float, crop: int = CROP) -> tuple[int, int]:
    """Round the top-left corner while preserving the projected center."""
    return (int(np.floor(col - crop / 2 + 0.5)), int(np.floor(row - crop / 2 + 0.5)))


def zoom_crop(
    img: np.ndarray, col: float, row: float, crop: int = CROP, out_hw: int = OUT_HW
) -> tuple[np.ndarray, dict]:
    from PIL import Image

    assert img.dtype == np.uint8 and img.ndim == 3 and (img.shape[2] == 3), (
        img.dtype,
        img.shape,
    )
    (h, w) = img.shape[:2]
    (c0, r0) = zoom_window(col, row, crop)
    pad = np.zeros((crop, crop, 3), np.uint8)
    (rs, re_) = (max(r0, 0), min(r0 + crop, h))
    (cs, ce) = (max(c0, 0), min(c0 + crop, w))
    if re_ > rs and ce > cs:
        pad[rs - r0 : re_ - r0, cs - c0 : ce - c0] = img[rs:re_, cs:ce]
    inside_px = max(re_ - rs, 0) * max(ce - cs, 0)
    z = np.array(
        Image.fromarray(pad).resize((out_hw, out_hw), Image.BILINEAR),
        dtype=np.uint8,
        copy=True,
    )
    return (z, {"c0": c0, "r0": r0, "inside_frac": inside_px / float(crop * crop)})
