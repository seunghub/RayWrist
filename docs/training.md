# Training details

The scheduled recipes are in [`configs/training_recipes.json`](../configs/training_recipes.json); the reader architecture is in [`configs/architecture.json`](../configs/architecture.json). Use the native backbone data transforms, action targets, and action objective. Geometry uses physical state before native state normalization or dropout.

## Writer initialization

Before adapter updates, initialize writer scale from training source-crop features. GR00T's `reader.initialize_native_rms(merged, deepstack, geometry)` rescales each writer matrix by its source-to-output RMS ratio and keeps zero writer biases. For π₀.₅, use `raywrist.jax.initialization.calibrate_writer(parameters, source_features, initial_output)`. This matches the per-channel source mean and the centered RMS, as in supplementary Eq. S9. The initial statistics use source crops only. The geometric-residual output remains zero-initialized.

## Frozen adaptation

Freeze the native recipient's parameters and detach encoder features. Keep the gradient path from the action objective through the recipient to completed wrist features. A student batch contains retained views and external-derived crops at missing wrist slots; the full-view teacher has the paired actual wrist views.

For GR00T, the comparison point is the native visual-language conditioning after its normalization and self-attention. For π₀.₅, it is the prefix keys and values from all 18 layers. The valid-token mask refers to this conditioning sequence, including generated wrist-token positions. Combine squared-error sums and valid-element counts across variable-length GR00T groups before normalization. Native preprocessing and masks stay shared between the student and teacher.

Use AdamW and the cosine schedules recorded in the recipe file. Missing-view adaptation uses alignment weight 1. The π₀.₅ recipe uses final scheduled EMA parameters with decay 0.999; GR00T uses the final scheduled reader. Native state dropout remains a training regularizer, while geometry is constructed from the physical state before dropout.

## Head-only joint learning

RoboCasa uses two independent readers with the native action fine-tuning scope for 30,000 updates, with vision and language frozen. The [RoboCasa integration](../src/raywrist/integrations/robocasa.py) registers both readers in the native model, retains the selected policy parameters, and applies the original action objective with alignment weight zero. The physical end-effector origin selects the source crop; `EEReaderGeometry` constructs the fixed virtual wrist/grid and the processor tracks native image augmentation. See the [RoboCasa guide](robocasa.md) for the complete training and inference commands and [`configs/robocasa_gr1.json`](../configs/robocasa_gr1.json) for all 24 tasks.

## Checkpoints

For frozen adaptation, store reader state separately from the pretrained recipient. PyTorch readers retain their original state-dict names; save `reader.state_dict()` or `adapter.readers.state_dict()`. JAX readers retain their Linen parameter tree; save the final EMA parameter tree in the upstream checkpoint workflow. RoboCasa's joint checkpoint includes both readers and the native policy, along with its processor and normalization statistics. Load the same backbone, task normalization statistics, calibration, and preprocessing when applying a trained reader.
