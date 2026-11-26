import numpy as np
from raywrist.geometry.robotwin import Geometry
from raywrist.geometry.camera import OpenCV5Camera, fold_radius
from raywrist.geometry.robocasa import EEReaderGeometry
from raywrist.geometry.crop import zoom_crop


def test_crop_preserves_center_and_padding():
    image = np.full((16, 20, 3), 73, dtype=np.uint8)
    crop, info = zoom_crop(image, 0.0, 0.0, crop=8, out_hw=8)
    assert (info["c0"], info["r0"]) == (-4, -4)
    assert np.all(crop[:4] == 0) and np.all(crop[:, :4] == 0)
    assert np.all(crop[4:, 4:] == 73)


def test_robotwin_bin_centers_and_tool_transform():
    geometry = Geometry()
    context = geometry.batch(np.zeros((2, 14)))
    for arm in ("left", "right"):
        c = context[arm]
        static = geometry.static[arm]
        valid = c["valid_ray"][0]
        assert 0 < valid.sum() < 256
        assert c["volume_grid"].shape == (2, 256, 16, 3)
        np.testing.assert_allclose(np.linalg.norm(static["d"], axis=-1), 1.0)
        lo, hi = static["rho_min"][valid], static["rho_max"][valid]
        expected = lo[:, None] + (hi - lo)[:, None] * (np.arange(16) + 0.5) / 16
        np.testing.assert_allclose(c["rho"][0, valid], expected)
        assert np.all(c["xyz"][0, valid] >= geometry.lower - 1e-12)
        assert np.all(c["xyz"][0, valid] <= geometry.upper + 1e-12)
        assert np.all(c["rho"][0, valid] >= static["near_rho"][valid, None])
    with np.testing.assert_raises(ValueError):
        geometry.batch(np.zeros((1, 12)))


def test_distortion_inverse_and_invalid_depth():
    intrinsic = np.array([[400.0, 0.0, 320.5], [0.0, 400.0, 240.5], [0.0, 0.0, 1.0]])
    distortion = np.array([0.02, -0.005, 0.001, -0.001, 0.001])
    camera = OpenCV5Camera(intrinsic, distortion, (640, 480), *fold_radius(distortion))
    xyz = np.array([[0.1, 0.03, 0.8], [-0.15, 0.09, 1.2], [0.0, 0.0, -1.0]])
    uv, valid = camera.project(xyz)
    assert valid.tolist() == [True, True, False]
    inverse, inverse_valid = camera.unproject(uv[:2])
    assert inverse_valid.all()
    np.testing.assert_allclose(inverse, xyz[:2, :2] / xyz[:2, 2:], atol=1e-7)


def test_robocasa_physical_crop_and_virtual_grid():
    geometry = EEReaderGeometry()
    raw = np.zeros(29)
    pair = geometry.pair(raw, layout="geometry29")
    assert pair["volume_grid"].shape == (2, 64, 16, 3)
    assert pair["pose_code"].shape == (2, 12)
    np.testing.assert_allclose(
        geometry.native.ee_from_virtual[:3, 3], [0.0, 0.0, -0.06]
    )
    np.testing.assert_allclose(geometry.native.query_K, geometry.native.source.fk.K)
    assert geometry.native.query_hw == (800, 1280)
    physical = geometry.native.source.fk.project(raw)
    for ai, arm in enumerate(("left", "right")):
        detail = geometry.native.context(raw, arm, layout="geometry29")
        np.testing.assert_array_equal(
            detail["crop_offsets"], np.floor(physical[arm]["uv"] - 48 + 0.5).astype(int)
        )
        np.testing.assert_allclose(
            detail["robot_from_volume"],
            physical[arm]["tool_from_robot"] @ geometry.native.ee_from_virtual,
        )
        assert np.array_equal(
            pair["valid_ray"][ai], detail["valid_ray"] & physical[arm]["center_valid"]
        )
