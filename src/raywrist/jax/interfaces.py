"""Insert completed wrist features while retaining available native streams."""

from .reader import PiRayGrid

WRIST_KEYS = {"left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}


def complete_visual_features(visual, early, geometry, parameters, missing_sides):
    """Return native feature streams with readers enabled only for missing wrists.

    visual maps native image keys to [B,256,2048] encoder outputs. For a
    missing wrist, that input is the head-derived crop encoding. early and
    geometry are keyed by arm. An available wrist retains its native tensor.
    """
    missing_sides = tuple(missing_sides)
    if len(set(missing_sides)) != len(missing_sides) or any(
        s not in WRIST_KEYS for s in missing_sides
    ):
        raise ValueError("Missing wrists must be distinct left/right entries")
    output = dict(visual)
    reader = PiRayGrid()
    for side in missing_sides:
        key = WRIST_KEYS[side]
        output[key] = reader.apply(
            {"params": parameters[side]}, visual[key], early[side], geometry[side]
        )
    return output
