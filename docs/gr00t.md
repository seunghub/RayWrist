# GR00T N1.7

## Environment

From the RayWrist repository root:

```bash
export RAYWRIST_ROOT="$PWD"
python scripts/setup_upstream.py gr00t --submodules
cd "$RAYWRIST_ROOT/third_party/Isaac-GR00T"
uv sync --python 3.12
uv pip install --python .venv/bin/python -e "$RAYWRIST_ROOT[test]"
cd "$RAYWRIST_ROOT"
third_party/Isaac-GR00T/.venv/bin/python examples/torch_reader.py
```

Follow the pinned [official installation guide](https://github.com/NVIDIA/Isaac-GR00T/tree/d2b7e75b937e3ec9aa5dbc798f08b89692c49734#installation) for the platform-specific CUDA setup and backbone access. The integration uses the upstream PyTorch 2.9 / Transformers 4.57.3 environment. RayWrist does not replace these packages with an older PyTorch build.

For RoboCasa's two-reader joint training and head-only inference, use the dedicated [RoboCasa guide](robocasa.md) and training entry point. The following sections describe frozen missing-view adaptation.

## Load the native policy and reader

```python
import torch
from gr00t.policy.gr00t_policy import Gr00tPolicy
from raywrist.torch import RayWrist
from raywrist.integrations.gr00t import GrootRayWrist

policy = Gr00tPolicy(
    "LIBERO_PANDA", "/path/to/native_libero_checkpoint", device="cuda:0"
)
reader = RayWrist().to("cuda:0")
reader.load_state_dict(torch.load("/path/to/reader_state.pt", weights_only=True))
reader.eval()
adapter = GrootRayWrist(policy.model, {"left": reader}, {"left": 1})
```

For LIBERO, native image order is external view, then wrist view. Put the head-derived tool crop at the missing wrist's image slot **before** calling the processor. The native processor still constructs its original image tokens and position IDs. Geometry must describe the crop, image preprocessing, physical tool pose, and target wrist-token lattice.

The included LIBERO geometry can construct the source crop and reader context:

```python
import json
from pathlib import Path
import raywrist
from raywrist.geometry.libero_torch import RayContext

cameras = json.loads(
    (Path(raywrist.__file__).parent / "configs/libero_cameras.json").read_text()
)
geometry = RayContext(cameras, device="cuda:0")
# task_id is the descriptor ID in libero_cameras.json.
# physical_state is [x,y,z,rotvec_x,rotvec_y,rotvec_z,gripper_0,gripper_1].
crop = geometry.crop(external_rgb_uint8, task_id, physical_state)
context = geometry.tensor(geometry.numpy([task_id], physical_state[None]))
```

Descriptor IDs cover all four suites; use `geometry.task_by_env` to resolve the native environment name. The physical rotation is an axis-angle vector. Retain the recorded image orientation; changing the source raster requires the corresponding calibration/affine change.

## Inference

Prepare the usual batched `video`, `state`, and `language` observation using the checkpoint's modality configuration, with `crop` replacing the unavailable wrist image. Then call the original policy inside the adapter context:

```python
with adapter.complete({"left": context}):
    actions, info = policy.get_action(student_observation)
```

The policy performs its own processing, action sampling, and decoding. The context completes only the missing slot's merged feature and three DeepStack streams. Block-5 crop features are captured in the same source encoder pass and restored to their unmerged raster order. Outside this context, a normal full-view `policy.get_action(full_view_observation)` call follows the native path.

For a three-view embodiment, pass its actual processor image order, for example `images_per_sample=3` and `slot_by_side={"left": 1, "right": 2}`. Each missing side has an independent reader and geometry. Include only the missing sides when constructing an adapter; the other native image slots pass through. Source and target slot packing must match the recipient checkpoint, including its native target raster size. Use the relevant benchmark geometry and recorded preprocessing affines.

## Adaptation

Native training batches come from the upstream processor/collator and contain the native action targets and masks. Construct student crops and the paired full-view teacher with shared spatial preprocessing. The reader uses physical geometry before state normalization or dropout.

```python
policy.model.train()
reader.train()
optimizer = torch.optim.AdamW(
    adapter.readers.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5
)
# student_inputs and teacher_inputs are native processed batches.
teacher = adapter.teacher_conditioning(teacher_inputs)
with torch.autocast("cuda", dtype=torch.bfloat16):
    student = adapter(student_inputs, {"left": context})
    loss, terms = adapter.training_loss(student, teacher)
loss.backward()
torch.nn.utils.clip_grad_norm_(adapter.readers.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Initialize writer scale before updates and apply the scheduled learning rate, accumulation, native training regularization and final checkpoint selection in [the training recipe](training.md). Freeze recipient parameters while retaining the input-gradient path. The integration does not change the upstream module's train/eval flags. Use the native handling of variable-length conditioning; combine valid-element sums across length groups when computing alignment. Save `reader.state_dict()` for the single reader above.
