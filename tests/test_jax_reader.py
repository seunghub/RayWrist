import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
from raywrist.jax.reader import PiRayGrid, sample
from raywrist.jax.losses import conditioning_alignment
from raywrist.jax.interfaces import complete_visual_features
from raywrist.geometry.robotwin import Geometry


def test_prefix_alignment_mask_suffix_and_teacher_gradient():
    teacher = jnp.zeros((18, 1, 3, 2, 4))
    student = jnp.ones((18, 1, 5, 2, 4))
    mask = jnp.array([[True, False, True]])
    # Suffix and the padded middle token have deliberately different values.
    student = student.at[:, :, 3:].set(1000.0).at[:, :, 1].set(1000.0)
    loss = conditioning_alignment((student, student), (teacher, teacher), mask)
    np.testing.assert_allclose(loss, 1.0)
    grad = jax.grad(lambda t: conditioning_alignment((student, student), (t, t), mask))(
        teacher
    )
    np.testing.assert_array_equal(grad, 0.0)


def test_border_interpolation_and_cell_centers():
    field = jnp.arange(4, dtype=jnp.float32).reshape(1, 2, 2, 1)
    coordinates = jnp.array([[[[-0.5, -0.5], [0.5, 0.5], [0.0, 0.0], [-2.0, -2.0]]]])
    result = sample(field, coordinates)
    np.testing.assert_allclose(np.asarray(result).ravel(), [0.0, 3.0, 1.5, 0.0])


def test_pi_reader_shape_parameter_count_and_invalid_null():
    geom = Geometry()
    c = geom.tensor_fields(geom.batch(np.zeros((1, 14))))["left"]
    c = jax.tree.map(jnp.asarray, c)
    native = jnp.zeros((1, 256, 2048), jnp.bfloat16)
    early = tuple(jnp.zeros((1, 256, 1152), jnp.bfloat16) for _ in range(3))
    reader = PiRayGrid()
    variables = reader.init(jax.random.key(0), native, early, c)
    assert sum(x.size for x in jax.tree.leaves(variables["params"])) == 13_746_048
    output, diagnostic = reader.apply(variables, native, early, c, diagnostics=True)
    assert output.shape == native.shape and output.dtype == native.dtype
    weights = np.asarray(diagnostic["attention"])
    np.testing.assert_allclose(weights.sum(-1), 1.0, atol=2e-6)
    invalid = ~np.asarray(c["valid_ray"])
    assert np.all(weights[invalid, :16] == 0) and np.all(weights[invalid, 16] == 1)
    assert np.all(np.asarray(diagnostic["ray_delta_rms"]) == 0)


def test_full_view_bypasses_all_readers():
    image = jnp.zeros((1, 256, 2048))
    visual = {
        "base_0_rgb": image,
        "left_wrist_0_rgb": image,
        "right_wrist_0_rgb": image,
    }
    output = complete_visual_features(visual, {}, {}, {}, ())
    assert all(output[k] is visual[k] for k in visual)
