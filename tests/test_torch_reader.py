import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import raywrist
from raywrist.torch import RayWrist
from raywrist.torch.losses import conditioning_alignment, grouped_conditioning_alignment
from raywrist.torch.native_path import replace_wrist_streams
from raywrist.geometry.libero_torch import RayContext


def context():
    cameras = json.loads(
        (Path(raywrist.__file__).parent / "configs/libero_cameras.json").read_text()
    )
    geometry = RayContext(cameras)
    c = geometry.tensor(
        geometry.numpy([0], np.array([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    )
    c.update(
        source_hw=(8, 8),
        source_early_hw=(16, 16),
        target_hw=(8, 8),
        crop_early=torch.zeros(1, 256, 1024),
    )
    return c


def test_native_shapes_null_and_zero_update_gate():
    torch.set_num_threads(4)
    reader = RayWrist(seed=0).eval()
    assert sum(p.numel() for p in reader.parameters()) == 15_005_056
    merged = torch.zeros(1, 64, 2048)
    ds = tuple(torch.zeros_like(merged) for _ in range(3))
    c = context()
    c["valid_ray"][:, 0] = False
    reader.capture = True
    with torch.no_grad():
        plain = reader(merged, ds)
        completed = reader(merged, ds, c)
    for a, b in zip((plain[0], *plain[1]), (completed[0], *completed[1])):
        assert a.shape == (1, 64, 2048)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    alpha = reader.last_diagnostics["fp32_recomputed_candidate_attention"]
    torch.testing.assert_close(alpha.sum(-1), torch.ones(1, 64))
    assert torch.equal(alpha[0, 0, :16], torch.zeros(16)) and alpha[0, 0, 16] == 1
    # Check the validity gate with a nonzero output projection as well.
    with torch.no_grad():
        reader.ray_output.weight.fill_(0.001)
        q = torch.randn(1, 64, 512)
        refined = reader._ray_residual(q, (), c)
    torch.testing.assert_close(refined[:, 0], q[:, 0], rtol=0, atol=0)
    assert torch.isfinite(refined).all()


def test_alignment_mask_stop_gradient_and_group_counts():
    student = torch.tensor([[[1.0, 3.0], [99.0, 99.0]]], requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    mask = torch.tensor([[True, False]])
    loss = conditioning_alignment(student, teacher, mask)
    torch.testing.assert_close(loss, torch.tensor(5.0))
    loss.backward()
    assert teacher.grad is None
    assert torch.equal(student.grad[:, 1], torch.zeros(1, 2))
    second = (
        torch.ones(1, 2, 2),
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    grouped = grouped_conditioning_alignment([(student, teacher, mask), second])
    torch.testing.assert_close(grouped, torch.tensor(14.0 / 6.0))


def test_wrist_insertion_keeps_retained_streams_and_gradients():
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, merged, ds):
            return merged + self.offset, tuple(x + self.offset for x in ds)

    reader = Reader()
    merged = (torch.zeros(64, 2048), torch.ones(64, 2048))
    ds = tuple(torch.arange(128.0)[:, None].expand(128, 2048).clone() for _ in range(3))
    out, out_ds = replace_wrist_streams(merged, ds, reader)
    assert out[0] is merged[0]
    for a, b in zip(out_ds, ds):
        assert torch.equal(a[:64], b[:64])
    out[1].sum().backward()
    assert reader.offset.grad is not None and reader.offset.grad > 0
