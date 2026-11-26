"""Alignment of native pi_0.5 prefix keys and values."""

import jax
import jax.numpy as jnp


def conditioning_alignment(student, teacher, prefix_mask):
    """Match valid prefix positions across all 18 layers, excluding the suffix."""
    if teacher is None or len(student) != 2 or len(teacher) != 2:
        raise ValueError("Expected student and teacher (key, value) caches")
    shape = teacher[0].shape
    if len(shape) != 5 or shape[0] != 18 or shape[1:3] != prefix_mask.shape:
        raise ValueError("Native cache layout is [18, batch, prefix, heads, width]")
    if any(t.shape != shape for t in teacher):
        raise ValueError("Teacher key/value shapes differ")
    if any(
        s.ndim != 5
        or s.shape[:2] != shape[:2]
        or s.shape[2] < shape[2]
        or s.shape[3:] != shape[3:]
        for s in student
    ):
        raise ValueError("Student cache does not contain the same native prefix")
    mask = prefix_mask[None, :, :, None, None].astype(bool)
    layers, _, tokens, heads, width = shape
    count = jnp.sum(prefix_mask, dtype=jnp.float32)
    denominator = jnp.asarray(2 * layers * heads * width, jnp.float32) * jnp.maximum(
        count, 1.0
    )
    numerator = sum(
        jnp.sum(
            jnp.where(
                mask,
                jnp.square(
                    predicted[:, :, :tokens].astype(jnp.float32)
                    - jax.lax.stop_gradient(target).astype(jnp.float32)
                ),
                0.0,
            )
        )
        for predicted, target in zip(student, teacher)
    )
    return numerator / denominator
