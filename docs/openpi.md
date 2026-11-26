# π₀.₅ / OpenPI

## Environment

Use a separate Python 3.11 environment for the JAX OpenPI backend:

```bash
export RAYWRIST_ROOT="$PWD"
python scripts/setup_upstream.py openpi --submodules
cd "$RAYWRIST_ROOT/third_party/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
uv pip install --python .venv/bin/python -e "$RAYWRIST_ROOT[jax,test]"
cd "$RAYWRIST_ROOT"
JAX_PLATFORMS=cpu third_party/openpi/.venv/bin/python examples/jax_reader.py
```

The integration targets the pinned [official OpenPI source](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479), with JAX 0.5.3 and Flax 0.10.2. Its SigLIP encoder already exposes intermediate block outputs; no upstream file patch is needed.

## Native model and reader

Load the experiment's native π₀.₅ checkpoint using OpenPI's policy loader. For the official LIBERO recipient:

```python
from openpi.policies import policy_config
from openpi.training import config as train_config
from raywrist.integrations.openpi import OpenPIRayWrist

policy = policy_config.create_trained_policy(
    train_config.get_config("pi05_libero"),
    "gs://openpi-assets/checkpoints/pi05_libero",
)
adapter = OpenPIRayWrist(policy._model, missing_sides=("left",))
```

Keep the recipient's input/output transforms and normalization assets. LIBERO uses a 10-step native action horizon; RoboTwin uses its 50-step recipient. The integration takes the horizon from the supplied model.

For each missing wrist, put the head-derived crop into its native image entry (`left_wrist_0_rgb` or `right_wrist_0_rgb`) and retain a valid native image mask for the completed slot. Available camera entries retain their actual observations and masks. Run the native input transforms and image preprocessing, and construct geometry using the corresponding affines. Pass this prepared `Observation` to the adapter; spatial preprocessing is not applied a second time inside it.

For LIBERO, `raywrist.geometry.libero_jax.Geometry` exposes `batch(task_ids, physical_states)` and the zero-padded 96-pixel crop. For RoboTwin, `raywrist.geometry.robotwin.Geometry` supplies the raw-drive-target FK, 120-pixel source crops, and per-arm fields. Convert the returned geometry arrays to JAX arrays on the model device. Physical-state input is separate from the normalized native state.

A LIBERO source crop and geometry batch can be prepared with:

```python
import jax.numpy as jnp
from raywrist.geometry.libero_jax import Geometry

calibration = Geometry()
# Descriptor IDs match configs/libero_cameras.json; physical_states has shape [B,8].
raw = calibration.batch(task_ids, physical_states)
crop = calibration.crop(external_rgb_uint8, raw["offsets"][0])
geometry = {"left": {k: jnp.asarray(v) for k, v in
                     calibration.tensor_fields(raw).items()}}
```

The π₀.₅ reader consumes blocks 5, 12 and 19 plus the final SigLIP embedding, and returns one native 256×2048 wrist stream. A reader can be initialized from an encoded training crop as follows:

```python
import jax
from raywrist.jax import PiRayWrist
from raywrist.jax.initialization import calibrate_writer

visual, early = adapter.encode_images(prepared_observation)
reader = PiRayWrist()
variables = reader.init(
    jax.random.key(2029), visual["left_wrist_0_rgb"],
    early["left"], geometry["left"],
)
initial = reader.apply(
    variables, visual["left_wrist_0_rgb"], early["left"], geometry["left"],
)
parameters = {"left": calibrate_writer(
    variables["params"], visual["left_wrist_0_rgb"], initial,
)}
```

Use a representative training crop batch for this initialization. When deploying a trained adapter, load its saved final EMA tree instead. Use independent parameter trees for left and right readers.

## Training and inference

The full-view teacher is a prepared observation with paired actual wrist RGB. It shares the student's retained views, language, and native masks.

```python
teacher = adapter.teacher_cache(full_view_prepared_observation)
(loss, terms), gradients = jax.value_and_grad(adapter.loss, has_aux=True)(
    parameters, prepared_observation, geometry, actions, training_key,
    teacher=teacher,
)
```

Apply these gradients to the reader tree with the AdamW schedule and EMA in [the training recipe](training.md). The frozen recipient stays outside the optimized parameter tree. Prefix conditioning alignment excludes suffix/action tokens and detaches the teacher cache.

For inference, generate native Gaussian action noise and use the original 10-step Euler integration:

```python
import jax.numpy as jnp
noise = jax.random.normal(
    sampling_key,
    (prepared_observation.state.shape[0], policy._model.action_horizon,
     policy._model.action_dim),
)
actions = adapter.sample_actions(
    parameters, prepared_observation, geometry, noise, num_steps=10,
)
```

The returned actions are in the model's normalized space; apply the original policy output transforms. The adapter retains native image order, language embeddings, prefix/suffix masks, cache positions, and action decoding scope. For the native full-view path, call the original policy with full-view observations; the same pretrained recipient is reused.
