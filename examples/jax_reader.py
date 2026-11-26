"""CPU example using synthetic encoder features and fixed RoboTwin calibration."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import jax.numpy as jnp
from raywrist.jax import PiRayWrist
from raywrist.geometry.robotwin import Geometry


def main():
    geometry = Geometry()
    context = geometry.tensor_fields(geometry.batch(np.zeros((1, 14))))["left"]
    context = jax.tree.map(jnp.asarray, context)
    final = jnp.zeros((1, 256, 2048), jnp.bfloat16)
    early = tuple(jnp.zeros((1, 256, 1152), jnp.bfloat16) for _ in range(3))
    reader = PiRayWrist()
    variables = reader.init(jax.random.key(0), final, early, context)
    output, diagnostic = reader.apply(
        variables, final, early, context, diagnostics=True
    )
    print("Native output:", output.shape, "; attention:", diagnostic["attention"].shape)
    print(
        "Reader parameters:", sum(x.size for x in jax.tree.leaves(variables["params"]))
    )


if __name__ == "__main__":
    main()
