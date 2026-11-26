# Head-camera-only learning on RoboCasa

RoboCasa GR1 Tabletop evaluates joint policy learning from head-camera RGB. RayWrist adds two independent virtual-wrist readers to GR00T N1.7 and trains them together with the native projector and action modules. Vision and language stay frozen. This setting uses action supervision, with no wrist images or full-view teacher.

## Environment and data

Use the [GR00T N1.7 environment](gr00t.md#environment) and the pinned official source revision in [`configs/upstream.json`](../configs/upstream.json). The [official GR1 Tabletop guide](https://github.com/NVIDIA/Isaac-GR00T/tree/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/examples/robocasa-gr1-tabletop-tasks) describes dataset conversion and simulator setup. Use the native base checkpoint with its processor, GR1 normalization statistics and embodiment mapping.

Place the 24 prepared LeRobot task directories under one dataset root. Their names are listed in [`configs/robocasa_gr1.json`](../configs/robocasa_gr1.json). The dataset reads the recorded head view, physical state and native action targets. Tool crops are created in the processor; they are not additional recorded cameras.

## Observation path

The input is the benchmark's pre-augmentation `uint8[256,256,3]` head RGB and unnormalized physical GR1 state. The processor computes forward kinematics before native state normalization or dropout. Each physical end-effector origin selects a 96-pixel square crop, resized to 256 pixels with the original out-of-image zero padding.

Native image augmentation runs once and replays the same spatial transform across views. Its recorded affine also transforms the source-image projections used by the readers. The physical crop origin remains separate from each virtual wrist: the virtual origin is 6 cm behind the end effector, aligned with its grip frame. The virtual wrist and feature grid use this frame, and the 8 × 8 query lattice uses the raw head-camera intrinsics.

The native image order is head view, left tool crop, right tool crop. The readers replace only the crop slots' merged and three DeepStack feature streams. Head-view features pass through unchanged. The block-5 feature raster supplies each reader's 8 × 8 × 8 grid, with 16 samples per query ray.

## Joint training

Run from the RayWrist repository root in the GR00T environment:

```bash
torchrun --standalone --nproc-per-node=4 scripts/train_robocasa.py \
  --base-model /path/to/native_gr00t_n17_base \
  --dataset-root /path/to/prepared_gr1_tabletop \
  --output-dir runs/robocasa_raywrist
```

Add `--dry-run` to the Python command to inspect the resolved task list and training budget without loading a model. Add `--baseline` for native head-only fine-tuning with the same data and optimization budget.

| Setting | Value |
| --- | --- |
| Optimizer updates | 30,000 |
| Global batch | 512 = 4 devices × 32 examples × 4 accumulation steps |
| Optimizer | AdamW, β=(0.9, 0.999), ε=1e-8, weight decay 1e-5 |
| Learning rate | 1e-4, 1,500-update warm-up, cosine decay to zero |
| Gradient clipping | 1.0 |
| Training seed / reader initialization | 42 / left 42, right 43 |
| Trainable scope | Native projector, action modules, conditioning normalization, two readers |
| Precision | FP32 trainable parameters and optimizer state; native BF16 computation |
| Checkpoint | Final scheduled update; no success-rate-based selection |

The training entry uses the official `DatasetFactory`, native action objective and `Gr00tTrainer`. The readers are registered in the native model, so both readers and the selected policy parameters enter its optimizer and checkpoint. The two readers add 30,010,112 trainable parameters to the native scope of 1,620,515,968. Reader gradients pass through the frozen language model.

## Inference

Load the same base model and the saved joint checkpoint:

```python
from raywrist.integrations.robocasa import RoboCasaPolicy

policy = RoboCasaPolicy.from_pretrained(
    "/path/to/native_gr00t_n17_base",
    "runs/robocasa_raywrist/final30000",
)
actions = policy.infer(state44, head_rgb, prompt)
```

`state44` follows the benchmark's raw 44-dimensional physical-state layout. `head_rgb` is the native 256-pixel head observation before image augmentation. Use the original task instruction, including its `unlocked_waist:` prefix. The policy returns physical action chunks for the left arm, right arm, left hand, right hand and waist. It uses native action sampling and decoding and returns the first eight action steps, matching the benchmark interface.

For a saved observation, use [`examples/robocasa_inference.py`](../examples/robocasa_inference.py). The NPZ input contains `state44` and `head_rgb`. The loader restores both trained readers together with the native policy weights and checks for missing tensors.

## Evaluation and reported results

Start the policy server in the GR00T environment:

```bash
python scripts/serve_robocasa.py \
  --base-model /path/to/native_gr00t_n17_base \
  --checkpoint runs/robocasa_raywrist/final30000
```

This uses the official `Gr00tSimPolicyWrapper` and `PolicyServer`. Run the [official GR1 Tabletop rollout client](https://github.com/NVIDIA/Isaac-GR00T/tree/d2b7e75b937e3ec9aa5dbc798f08b89692c49734/examples/robocasa-gr1-tabletop-tasks#evaluate-checkpoint) in its simulator environment, with `--policy-client-host localhost --n-action-steps 8 --max-episode-steps 720`. The wrapper requests only the head camera and native state; crops are constructed inside the processor.

The paper evaluates all 24 tasks with 50 episodes per task, a 720-step episode limit and eight executed actions per policy call. The native 40-step action representation is retained; eight steps and 29 active dimensions are supervised through its original masks. Evaluate the final scheduled checkpoint with the benchmark's full-task success predicate.

| Method | Successes | Success rate |
| --- | ---: | ---: |
| Native head-only FT | 394 / 1,200 | 32.83% |
| FT + RayWrist | 453 / 1,200 | 37.75% |

The 4.92-point aggregate improvement is concentrated in the six placement-and-closure tasks: 22/300 to 75/300 successes. The remaining 18 pick-and-place tasks change from 372/900 to 378/900. These are complete-task results from the paper's fixed evaluations.

The [per-task results CSV](../results/robocasa.csv) contains all 24 tasks, success counts and rates from those evaluations.
