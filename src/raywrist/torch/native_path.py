"""GR00T missing-wrist feature insertion and frozen-recipient input gradients."""

from contextlib import contextmanager
from functools import wraps
import torch
from torch.utils.checkpoint import checkpoint


class InputGradientCheckpointing:
    """Private reversible nonreentrant wrappers; frozen weights stay frozen."""

    def __init__(self, blocks):
        if len({id(b) for b in blocks}) != len(blocks):
            raise ValueError("duplicate checkpoint block")
        (self.blocks, self.saved) = (list(blocks), [])

    def install(self):
        if self.saved:
            raise ValueError("checkpoint wrappers already installed")
        for block in self.blocks:
            original = block.forward
            had = "forward" in block.__dict__
            previous = block.__dict__.get("forward")

            @wraps(original)
            def forward(*args, _original=original, **kwargs):
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return checkpoint(
                    _original,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **kwargs
                )

            block.forward = forward
            self.saved.append((block, had, previous))

    def remove(self):
        for block, had, previous in self.saved:
            if had:
                block.forward = previous
            else:
                delattr(block, "forward")
        self.saved.clear()


@contextmanager
def override_image_features(inner, replacement):
    """Restore exact prior method ownership even when HF forward raises."""
    had = "get_image_features" in inner.__dict__
    old = inner.__dict__.get("get_image_features")
    original = inner.get_image_features
    inner.get_image_features = replacement
    try:
        yield original
    finally:
        if had:
            inner.get_image_features = old
        else:
            delattr(inner, "get_image_features")


def replace_wrist_streams(merged, deepstack, reader):
    """Pure tensor assembly shared by CPU fixtures and actual native forward.

    Tuple merged is row-major [external,crop] per sample; DeepStack is flattened
    with exactly that same ordering. External outputs are retained unchanged.
    Only crop streams are supplied to R; output has the original native shape.
    """
    merged = tuple(merged)
    if not merged or len(merged) % 2 or len(deepstack) != 3:
        raise ValueError(
            "exactly two images per row and three DeepStack streams required"
        )
    batch = len(merged) // 2
    if any((t.shape != (64, 2048) for t in merged)):
        raise ValueError("each native image merger must have 64x2048 tokens")
    if any((t.shape != (batch * 128, 2048) for t in deepstack)):
        raise ValueError("native DeepStack flattened image order differs")
    crop = torch.stack(merged[1::2])
    crop_ds = tuple((t.reshape(batch, 2, 64, 2048)[:, 1] for t in deepstack))
    (new, new_ds) = reader(crop, crop_ds)
    (dtype, device) = (merged[0].dtype, merged[0].device)
    if (
        new.shape != (batch, 64, 2048)
        or len(new_ds) != 3
        or any((t.shape != new.shape for t in new_ds))
    ):
        raise ValueError("reader changed native stream shapes")
    output = tuple(
        (
            v
            for i in range(batch)
            for v in (merged[2 * i], new[i].to(device=device, dtype=dtype))
        )
    )
    output_ds = []
    for original, replacement in zip(deepstack, new_ds):
        external = original.reshape(batch, 2, 64, 2048)[:, 0]
        output_ds.append(
            torch.stack(
                (external, replacement.to(device=device, dtype=original.dtype)), dim=1
            ).reshape(batch * 128, 2048)
        )
    return (output, output_ds)


class NativeFrozenReaderPath:

    def __init__(self, policy, reader, *, checkpoint_blocks=True):
        (self.policy, self.reader) = (policy, reader)
        self.model = policy.model
        self.model.requires_grad_(False).eval()
        self.outer = self.model.backbone.model
        self.inner = self.outer.model
        self.head = self.model.action_head
        if (
            len(self.outer.language_model.layers) != 16
            or self.model.backbone.select_layer != 16
            or len(self.inner.visual.deepstack_visual_indexes) != 3
        ):
            raise ValueError("native GR00T LM16/DeepStack3 interface changed")
        self.checkpointing = None
        if checkpoint_blocks:
            blocks = [
                *self.outer.language_model.layers,
                *self.head.vl_self_attention.transformer_blocks,
                *self.head.model.transformer_blocks,
            ]
            self.checkpointing = InputGradientCheckpointing(blocks)
            self.checkpointing.install()
        self.last_trace = None

    def close(self):
        if self.checkpointing is not None:
            self.checkpointing.remove()

    def _validate_inputs(self, vl):
        required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        if set(vl) != set(required):
            raise ValueError("only original native image prefix inputs allowed")
        (ids, attn) = (vl["input_ids"], vl["attention_mask"])
        if ids.ndim != 2 or ids.shape != attn.shape or attn.ndim != 2:
            raise ValueError("native prefix IDs/attention dimensions changed")
        batch = ids.shape[0]
        if (
            vl["image_grid_thw"].shape != (2 * batch, 3)
            or not torch.equal(
                vl["image_grid_thw"],
                vl["image_grid_thw"].new_tensor([[1, 16, 16]]).expand(2 * batch, -1),
            )
            or vl["pixel_values"].shape != (2 * batch * 256, 1536)
        ):
            raise ValueError("native 256px external/crop image packing differs")
        image = ids == self.outer.config.image_token_id
        expected = torch.zeros_like(image)
        (expected[:, 4:68], expected[:, 70:134]) = (True, True)
        if (
            ids.shape[1] < 134
            or not torch.equal(image, expected)
            or (not bool(attn[image].all()))
        ):
            raise ValueError("both native image slots must remain present and attended")
        if (
            any((p.requires_grad for p in self.model.parameters()))
            or self.model.training
        ):
            raise ValueError("the full native recipient must remain frozen and eval")
        return (image, attn.bool())

    def encode_images(self, vl):
        """Frozen native merger/DS3 outputs suitable for caller-owned caching.

        Returned streams retain row-major [external,crop/wrist] order. This
        never caches a reader-dependent LM or action conditioning output.
        """
        self._validate_inputs(vl)
        with torch.no_grad():
            (merged, ds) = self.inner.get_image_features(
                vl["pixel_values"], vl["image_grid_thw"]
            )
        self._validate_visual_features(merged, ds, vl["input_ids"].shape[0])
        return (tuple(merged), tuple(ds))

    @staticmethod
    def _validate_visual_features(merged, ds, batch):
        if (
            len(merged) != batch * 2
            or len(ds) != 3
            or any((v.shape != (64, 2048) or v.dtype != torch.bfloat16 for v in merged))
            or any(
                (
                    v.shape != (batch * 128, 2048) or v.dtype != torch.bfloat16
                    for v in ds
                )
            )
        ):
            raise ValueError("frozen native BF16 merger/DS3 image packing changed")
        device = merged[0].device
        if any(
            (
                v.device != device or v.requires_grad
                for v in (*tuple(merged), *tuple(ds))
            )
        ):
            raise ValueError(
                "frozen visual streams must share a device and have no gradient graph"
            )

    def encode(
        self, vl, *, mode="reader", second_view_kind="ee_crop", visual_features=None
    ):
        (image, attn) = self._validate_inputs(vl)
        if mode not in ("reader", "native_crop", "full"):
            raise ValueError("unknown native mode")
        if second_view_kind != ("native_wrist" if mode == "full" else "ee_crop"):
            raise ValueError(
                "student crop and full-view wrist input provenance must be explicit"
            )
        trace = {
            "mode": mode,
            "reader_calls": 0,
            "vision_calls": 0,
            "native_merger_and_deepstack_replaced": mode == "reader",
            "full_view_bypass": mode == "full",
            "LM_calls": 0,
        }
        if mode == "reader" or visual_features is not None:
            if (
                mode == "reader"
                and self.reader.training
                and torch.is_inference_mode_enabled()
            ):
                raise ValueError(
                    "reader input backward cannot run inside inference_mode"
                )
            original = self.inner.get_image_features
            calls = [0]

            def transform(pixels, grid):
                calls[0] += 1
                if calls[0] != 1:
                    raise ValueError(
                        "native HF consumed visual features more than once"
                    )
                if pixels is not vl["pixel_values"] or grid is not vl["image_grid_thw"]:
                    raise ValueError("HF native visual input identity changed")
                if visual_features is None:
                    with torch.no_grad():
                        (merged, ds) = original(pixels, grid)
                    trace["vision_calls"] += 1
                else:
                    (merged, ds) = visual_features
                self._validate_visual_features(merged, ds, vl["input_ids"].shape[0])
                if mode == "reader":
                    trace["reader_calls"] += 1
                    return replace_wrist_streams(merged, ds, self.reader)
                return (tuple(merged), list(ds))

            with override_image_features(self.inner, transform):
                outputs = self.outer(**vl, output_hidden_states=True, use_cache=False)
            if (
                calls[0] != 1
                or trace["vision_calls"] != int(visual_features is None)
                or trace["reader_calls"] != int(mode == "reader")
            ):
                raise ValueError(
                    "one reader call and at most one frozen vision call required"
                )
        else:
            with torch.no_grad():
                outputs = self.outer(**vl, output_hidden_states=True, use_cache=False)
        trace["LM_calls"] = 1
        if len(outputs.hidden_states) != 17 or outputs.past_key_values is not None:
            raise ValueError(
                "native no-cache H17 pre-final-normalization recorder changed"
            )
        hidden = outputs.hidden_states[-1]
        if hidden.shape != (*vl["input_ids"].shape, 2048):
            raise ValueError("native joint LM feature shape changed")
        if mode == "reader" and torch.is_grad_enabled() and (not hidden.requires_grad):
            raise ValueError("frozen LM cut the reader input-gradient path")
        self.last_trace = trace
        return {
            "backbone_features": hidden,
            "backbone_attention_mask": attn,
            "image_mask": image,
        }

    def condition(self, encoded):
        """Existing native post-SA B comparison point, without detach/no_grad.

        Same-length unpadded training buckets use the original head directly.
        Variable-length padding is rejected because native SA has no mask.
        """
        if not bool(encoded["backbone_attention_mask"].all()):
            raise ValueError("native SA requires genuine same-length unpadded buckets")
        h = self.head.vlln(encoded["backbone_features"])
        h = self.head.vl_self_attention(h)
        if isinstance(h, (tuple, list)):
            h = h[0]
        return h
