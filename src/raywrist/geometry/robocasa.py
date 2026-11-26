"""RoboCasa virtual-wrist geometry with the physical-source-origin validity gate."""

import numpy as np
from .robocasa_source import ToolRayGeometry, physical_state, FIELDS
from .robocasa_virtual import EEVirtualConfig, EEVirtualGeometry

EE = dict(
    mode="ee_virtual_head_intrinsics",
    origin_in_ee_m=[0.0, 0.0, -0.06],
    forward_frame="grip",
    volume_frame="virtual",
    query_side=8,
    depth_samples=16,
    query_intrinsics="exact_raw_headcam_K",
    query_sensor_hw=[800, 1280],
    source_geometry="existing_headcam_crop_and_native_augmentation",
    source_coverage="original_grid_anchor_coverage_embedding_and_learned_null",
    invalid_ray="box_intersection_and_positive_source_headcam_depth",
)


class EEReaderGeometry:

    def __init__(self, calibration=None):
        self.native = EEVirtualGeometry(EEVirtualConfig(), calibration=calibration)

    def crops(self, image, state, *, layout="raw44"):
        return self.native.source.crops(image, state, layout=layout)

    def pair(self, state, *, layout="raw44", side=8, affine=None):
        if side != 8:
            raise ValueError("N17 EE contract requires8x8 query lattice")
        current = physical_state(state, layout)
        camera = self.native.source.fk.project(current)
        rows = []
        for arm in ["left", "right"]:
            value = self.native.context(
                current, arm, layout="geometry29", affine=affine
            )["features"]
            value["valid_ray"] = value["valid_ray"] & camera[arm]["center_valid"]
            rows.append(value)
        return {key: np.stack([row[key] for row in rows]) for key in FIELDS}
