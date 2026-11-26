"""RayWrist reader for the native pi_0.5 SigLIP feature interface."""

import itertools
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

EARLY_BLOCKS = (5, 12, 19)


def spatial(side, width):
    (yy, xx) = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    omega = np.exp(-np.log(10000) * np.arange(width // 4) / max(width // 4 - 1, 1))
    return jnp.asarray(
        np.concatenate(
            [
                np.sin(yy.reshape(-1, 1) * omega),
                np.cos(yy.reshape(-1, 1) * omega),
                np.sin(xx.reshape(-1, 1) * omega),
                np.cos(xx.reshape(-1, 1) * omega),
            ],
            -1,
        ),
        jnp.float32,
    )


def sample(field, grid):
    """Exact border/align_corners=False interpolation as a deterministic linear map.

    Coordinates are fixed calibration/state. A dense interpolation matrix avoids
    atomic scatter-add in backward when many wrist rays read the same voxel.
    """
    n = field.ndim - 2
    B = field.shape[0]
    shape = grid.shape[1:-1]
    V = int(np.prod(field.shape[1:-1]))
    g = grid.reshape(B, -1, n)
    positions = [
        jnp.clip(
            (g[..., n - 1 - a] + 1) * field.shape[a + 1] / 2 - 0.5,
            0,
            field.shape[a + 1] - 1,
        )
        for a in range(n)
    ]
    low = [jnp.floor(p).astype(jnp.int32) for p in positions]
    f = [p - i for (p, i) in zip(positions, low)]
    weights = jnp.zeros((B, g.shape[1], V), jnp.float32)
    for bits in itertools.product((0, 1), repeat=n):
        indices = [
            jnp.minimum(i + b, field.shape[a + 1] - 1)
            for (a, (i, b)) in enumerate(zip(low, bits))
        ]
        weight = jnp.ones_like(f[0])
        flat = jnp.zeros_like(low[0])
        for a, (frac, b, index) in enumerate(zip(f, bits, indices)):
            weight *= frac if b else 1 - frac
            flat = flat * field.shape[a + 1] + index
        weights += (flat[..., None] == jnp.arange(V)) * weight[..., None]
    result = jnp.einsum(
        "bkv,bvc->bkc",
        weights,
        field.reshape(B, V, field.shape[-1]).astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    return result.reshape(B, *shape, field.shape[-1])


class Decoder(nn.Module):

    @nn.compact
    def __call__(self, q, m):
        kw = dict(
            num_heads=8,
            qkv_features=512,
            out_features=512,
            dropout_rate=0.0,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        h = nn.LayerNorm(name="self_norm")(q)
        q = q + nn.MultiHeadDotProductAttention(name="self_attention", **kw)(h, h)
        h = nn.LayerNorm(name="cross_norm")(q)
        mem = nn.LayerNorm(name="memory_norm")(m)
        q = q + nn.MultiHeadDotProductAttention(name="cross_attention", **kw)(h, mem)
        h = nn.LayerNorm(name="ff_norm")(q)
        h = nn.Dense(2048, dtype=jnp.bfloat16, name="ff_in")(h)
        return q + nn.Dense(512, dtype=jnp.bfloat16, name="ff_out")(nn.gelu(h))


def gr_output_kernel(key, shape, dtype=jnp.float32):
    bound = 1.0 / np.sqrt(shape[0])
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class PiRayGrid(nn.Module):

    @nn.compact
    def __call__(self, native, early, c, *, diagnostics=False):
        B = native.shape[0]
        if (
            native.shape[1:] != (256, 2048)
            or not isinstance(early, tuple)
            or len(early) != len(EARLY_BLOCKS)
            or any((x.shape != (B, 256, 1152) for x in early))
        ):
            raise ValueError("Pi/SigLIP final and three early raster shapes changed")
        dense = lambda d, name, zero=False: nn.Dense(
            d,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=(
                nn.initializers.zeros if zero else nn.initializers.lecun_normal()
            ),
            name=name,
        )
        final = dense(512, "input_projection")(
            nn.LayerNorm(name="input_norm")(native.astype(jnp.float32))
        )
        streams = [
            dense(512, f"early_input_projection_{block:02d}")(
                nn.LayerNorm(name=f"early_input_norm_{block:02d}")(
                    x.astype(jnp.float32)
                )
            )
            for (block, x) in zip(EARLY_BLOCKS, early)
        ]
        streams.append(final)
        positions = spatial(16, 512)[None]
        stage = self.param("stage_embedding", nn.initializers.normal(0.02), (4, 512))
        m = jnp.concatenate(
            [x + positions + stage[i][None, None] for (i, x) in enumerate(streams)],
            axis=1,
        )
        queries = self.param("queries", nn.initializers.normal(0.02), (256, 512))
        q = jnp.broadcast_to(queries, (B, 256, 512))
        for i in range(2):
            q = Decoder(name=f"decoder_{i}")(q, m)
        feature = dense(128, "grid_input_projection")(
            nn.LayerNorm(name="grid_input_norm")(early[0].astype(jnp.float32))
        )
        anchors = sample(feature.reshape(B, 16, 16, 128), c["anchor_uv"])
        coverage = c["anchor_valid"].astype(jnp.float32)[..., None]
        anchors = (
            anchors * coverage
            + dense(128, "grid_position")(c["anchor_xyz"])
            + dense(128, "grid_coverage")(coverage)
        )
        volume = anchors.reshape(B, 8, 8, 8, 128)
        h = nn.Conv(
            128, (3, 3, 3), padding="SAME", dtype=jnp.bfloat16, name="volume_conv1"
        )(volume)
        volume = volume + nn.Conv(
            128, (3, 3, 3), padding="SAME", dtype=jnp.bfloat16, name="volume_conv2"
        )(nn.gelu(h))
        sampled = sample(volume, c["volume_grid"])
        cov = sample(coverage.reshape(B, 8, 8, 8, 1), c["volume_grid"])
        depth = dense(128, "ray_depth_out")(
            nn.gelu(
                dense(128, "ray_depth_in")(jnp.concatenate([c["depth_code"], cov], -1))
            )
        )
        values = sampled + depth
        null = self.param("ray_null", nn.initializers.normal(0.02), (1, 1, 1, 128))
        values = jnp.concatenate([values, jnp.broadcast_to(null, (B, 256, 1, 128))], 2)
        valid = jnp.concatenate(
            [
                jnp.broadcast_to(c["valid_ray"][..., None], (B, 256, 16)),
                jnp.ones((B, 256, 1), bool),
            ],
            2,
        )
        query = (
            dense(128, "ray_query")(q)
            + dense(128, "ray_pose")(c["pose_code"])[:, None]
            + dense(128, "ray_position")(c["ray_code"])
        )
        qq = dense(128, "attention_q")(nn.LayerNorm(name="ray_qnorm")(query)).reshape(
            B, 256, 4, 32
        )
        mm = nn.LayerNorm(name="ray_mnorm")(values)
        kk = dense(128, "attention_k")(mm).reshape(B, 256, 17, 4, 32)
        vv = dense(128, "attention_v")(mm).reshape(B, 256, 17, 4, 32)
        logits = jnp.einsum(
            "bqhd,bqkhd->bqhk", qq.astype(jnp.float32), kk.astype(jnp.float32)
        ) / np.sqrt(32)
        alpha = jax.nn.softmax(jnp.where(valid[:, :, None], logits, -jnp.inf), axis=-1)
        update = jnp.einsum("bqhk,bqkhd->bqhd", alpha, vv.astype(jnp.float32)).reshape(
            B, 256, 128
        )
        update = dense(128, "attention_out")(update)
        h = nn.LayerNorm(name="ray_ff_norm")(update)
        h = nn.gelu(dense(256, "ray_ff_in")(h))
        update += dense(128, "ray_ff_out")(h)
        delta = dense(512, "ray_output", zero=True)(update) * c["valid_ray"][..., None]
        mapped = nn.Dense(
            2048,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=gr_output_kernel,
            bias_init=nn.initializers.zeros,
            name="output_projection",
        )(nn.LayerNorm(name="output_norm")(q + delta))
        out = mapped.astype(native.dtype)
        if diagnostics:
            return (
                out,
                {
                    "attention": alpha.mean(2),
                    "volume_norm": jnp.sqrt(
                        jnp.mean(volume.astype(jnp.float32) ** 2, -1)
                    ),
                    "ray_delta_rms": jnp.sqrt(
                        jnp.mean(delta.astype(jnp.float32) ** 2, -1)
                    ),
                    "writer_rms": jnp.sqrt(
                        jnp.mean(mapped.astype(jnp.float32) ** 2, -1)
                    ),
                },
            )
        return out
