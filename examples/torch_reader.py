"""CPU example using synthetic encoder features and fixed LIBERO calibration."""

import torch
import numpy as np
from pathlib import Path
import raywrist
from raywrist.torch import RayWrist
from raywrist.geometry.libero_torch import RayContext


def main():
    torch.set_num_threads(4)
    cameras = Path(raywrist.__file__).parent / "configs/libero_cameras.json"
    import json

    geometry = RayContext(json.loads(cameras.read_text()))
    context = geometry.tensor(
        geometry.numpy([0], np.array([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    )
    context.update(source_hw=(8, 8), source_early_hw=(16, 16), target_hw=(8, 8))
    context["crop_early"] = torch.zeros(1, 256, 1024)
    merged = torch.zeros(1, 64, 2048)
    deepstack = tuple(torch.zeros_like(merged) for _ in range(3))
    reader = RayWrist().eval()
    with torch.no_grad():
        output, deepstack_output = reader(merged, deepstack, context)
    print(
        "Native output:",
        tuple(output.shape),
        "; DeepStack streams:",
        len(deepstack_output),
    )
    print("Reader parameters:", sum(p.numel() for p in reader.parameters()))


if __name__ == "__main__":
    main()
