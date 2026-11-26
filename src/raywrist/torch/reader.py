"""RayWrist reader for the GR00T merged and DeepStack feature interfaces."""

from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def hw(value, name="raster"):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a static (height,width) tuple")
    if any((type(x) is not int or x <= 0 for x in value)):
        raise ValueError(f"{name} dimensions must be positive Python integers")
    return tuple(value)


@dataclass(frozen=True)
class ReaderConfig:
    tokens: int = 64
    native_dim: int = 2048
    width: int = 512
    blocks: int = 2
    heads: int = 8
    ff_multiplier: int = 4
    streams: int = 4
    grid_side: int = 8
    target_hw: tuple = (8, 8)
    source_hw: tuple = (8, 8)

    def validate(self):
        target = hw(self.target_hw, "target_hw")
        hw(self.source_hw, "source_hw")
        if (
            self.tokens != math.prod(target)
            or self.native_dim != 2048
            or self.streams != 4
        ):
            raise ValueError("Native four-stream interface/target query raster differs")
        if (
            self.width != 512
            or self.heads != 8
            or self.blocks != 2
            or (self.ff_multiplier != 4)
        ):
            raise ValueError("Keep the validated GR00T decoder architecture")


def spatial_sincos_rect(shape, width):
    """Original row/column sinusoid, extended to rectangular input rasters."""
    (height, columns) = hw(shape)
    k = torch.arange(width // 4, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * k / max(1, width // 4 - 1))
    (y, x) = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(columns, dtype=torch.float32),
        indexing="ij",
    )
    return torch.cat(
        (
            (y.reshape(-1, 1) * omega).sin(),
            (y.reshape(-1, 1) * omega).cos(),
            (x.reshape(-1, 1) * omega).sin(),
            (x.reshape(-1, 1) * omega).cos(),
        ),
        dim=-1,
    )


class QueryDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.width)
        self.self_attention = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(cfg.width)
        self.memory_norm = nn.LayerNorm(cfg.width)
        self.cross_attention = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=0.0, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * cfg.ff_multiplier),
            nn.GELU(),
            nn.Linear(cfg.width * cfg.ff_multiplier, cfg.width),
        )

    def forward(self, q, memory):
        h = self.self_norm(q)
        q = q + self.self_attention(h, h, h, need_weights=False)[0]
        (h, m) = (self.cross_norm(q), self.memory_norm(memory))
        q = q + self.cross_attention(h, m, m, need_weights=False)[0]
        return q + self.ff(self.ff_norm(q))


def grid2(feature, grid, shape):
    (height, width) = hw(shape)
    (batch, count, dim) = feature.shape
    if count != height * width:
        raise ValueError("Early feature count/raster mismatch")
    image = feature.reshape(batch, height, width, dim).permute(0, 3, 1, 2)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        result = F.grid_sample(
            image.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
    return result.permute(0, 2, 3, 1)


def diagnostic_attention(module, q, m, mask=None):
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
        (dim, heads, batch, nq, nk) = (
            module.embed_dim,
            module.num_heads,
            q.shape[0],
            q.shape[1],
            m.shape[1],
        )
        (weight, bias) = (module.in_proj_weight.float(), module.in_proj_bias.float())
        qq = (
            F.linear(q.detach().float(), weight[:dim], bias[:dim])
            .reshape(batch, nq, heads, dim // heads)
            .transpose(1, 2)
        )
        kk = (
            F.linear(m.detach().float(), weight[dim : 2 * dim], bias[dim : 2 * dim])
            .reshape(batch, nk, heads, dim // heads)
            .transpose(1, 2)
        )
        logits = qq @ kk.transpose(-2, -1) / (dim // heads) ** 0.5
        if mask is not None:
            logits = logits.masked_fill(mask[:, None, None], -torch.inf)
        return logits.softmax(-1).mean(1)


class RayReader(nn.Module):

    def __init__(self, config=None, *, arm="RayGrid", seed=0, checkpoint_blocks=False):
        super().__init__()
        if arm != "RayGrid":
            raise ValueError("The reader implements the RayGrid branch")
        self.config = config or ReaderConfig()
        self.config.validate()
        self.arm = arm
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.geometry = None
        self.capture = False
        self.last_diagnostics = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            cfg = self.config
            self.queries = nn.Parameter(torch.randn(cfg.tokens, cfg.width) * 0.02)
            self.stream_embedding = nn.Parameter(
                torch.randn(cfg.streams, cfg.width) * 0.02
            )
            self.input_norm = nn.LayerNorm(cfg.native_dim)
            self.input_projection = nn.Linear(cfg.native_dim, cfg.width)
            self.decoder = nn.ModuleList((QueryDecoder(cfg) for _ in range(cfg.blocks)))
            self.output_norm = nn.LayerNorm(cfg.width)
            self.output_projection = nn.ModuleList(
                (nn.Linear(cfg.width, cfg.native_dim) for _ in range(cfg.streams))
            )
            for projection in self.output_projection:
                nn.init.zeros_(projection.bias)
            self.register_buffer(
                "spatial_code",
                spatial_sincos_rect(cfg.source_hw, cfg.width),
                persistent=True,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202610140 + seed)
            width = 128
            self.ray_query = nn.Linear(512, width)
            self.ray_pose = nn.Linear(12, width)
            self.ray_position = nn.Linear(6, width)
            self.ray_depth = nn.Sequential(
                nn.Linear(5, width), nn.GELU(), nn.Linear(width, width)
            )
            self.ray_qnorm = nn.LayerNorm(width)
            self.ray_mnorm = nn.LayerNorm(width)
            self.ray_attention = nn.MultiheadAttention(
                width, 4, batch_first=True, dropout=0.0
            )
            self.ray_ff = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 2 * width),
                nn.GELU(),
                nn.Linear(2 * width, width),
            )
            self.ray_null = nn.Parameter(torch.randn(1, 1, width) * 0.02)
            self.ray_output = nn.Linear(width, 512)
            nn.init.zeros_(self.ray_output.weight)
            nn.init.zeros_(self.ray_output.bias)
            self.grid_input_norm = nn.LayerNorm(1024)
            self.grid_input_projection = nn.Linear(1024, width)
            self.grid_position = nn.Linear(3, width)
            self.grid_coverage = nn.Linear(1, width)
            self.volume_processor = nn.Sequential(
                nn.Conv3d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(width, width, 3, padding=1),
            )
        self.float()

    def _streams(self, merged, deepstack):
        if merged.ndim != 3 or len(deepstack) != 3 or merged.shape[-1] != 2048:
            raise ValueError(
                "Expected crop merger and three DeepStack [B,Nc,2048] streams"
            )
        streams = (merged, *tuple(deepstack))
        if any(
            (
                t.shape != merged.shape
                or t.device != merged.device
                or (not t.is_floating_point())
                for t in streams
            )
        ):
            raise ValueError("Crop streams must share floating shape and device")
        return streams

    def _native_memory(self, merged, deepstack, source_hw):
        streams = self._streams(merged, deepstack)
        shape = hw(source_hw, "source_hw")
        if merged.shape[1] != math.prod(shape):
            raise ValueError("Crop source token count differs from its actual raster")
        x = torch.stack(streams, 1).float()
        memory = self.input_projection(self.input_norm(x))
        positions = (
            self.spatial_code
            if shape == tuple(self.config.source_hw)
            else spatial_sincos_rect(shape, self.config.width).to(merged.device)
        )
        memory = (
            memory
            + positions.to(memory.dtype)[None, None]
            + self.stream_embedding.to(memory.dtype)[None, :, None]
        )
        return (
            streams,
            memory.reshape(
                len(x), self.config.streams * math.prod(shape), self.config.width
            ),
        )

    def forward(self, merged, deepstack, ctx=None):
        ctx = self.geometry if ctx is None else ctx
        if merged.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._forward(merged, deepstack, ctx)
        return self._forward(merged, deepstack, ctx)

    def _forward(self, merged, deepstack, ctx=None):
        shape = self.config.source_hw if ctx is None else ctx["source_hw"]
        (streams, memory) = self._native_memory(merged, deepstack, shape)
        q = self.queries[None].expand(len(merged), -1, -1)
        self.last_diagnostics = {}
        for block in self.decoder:
            if self.checkpoint_blocks and torch.is_grad_enabled():
                q = checkpoint(
                    block, q, memory, use_reentrant=False, preserve_rng_state=True
                )
            else:
                q = block(q, memory)
        if ctx is not None:
            q = self._ray_residual(q, streams, ctx)
        native = self.output_norm(q)
        output = tuple((projection(native) for projection in self.output_projection))
        return (output[0], output[1:])

    def _ray_residual(self, q, streams, ctx):
        (batch, count) = q.shape[:2]
        if count != self.config.tokens or hw(ctx["target_hw"], "target_hw") != tuple(
            self.config.target_hw
        ):
            raise ValueError("Ray target raster differs from native wrist slot")
        expected = {
            "crop_early": (batch, math.prod(hw(ctx["source_early_hw"])), 1024),
            "pose_code": (batch, 12),
            "ray_code": (batch, count, 6),
            "anchor_uv": (batch, 512, 2),
            "anchor_valid": (batch, 512),
            "anchor_xyz": (batch, 512, 3),
            "volume_grid": (batch, count, 16, 3),
            "depth_code": (batch, count, 16, 4),
            "valid_ray": (batch, count),
        }
        for name, shape in expected.items():
            if (
                name not in ctx
                or tuple(ctx[name].shape) != shape
                or ctx[name].device != q.device
            ):
                raise ValueError(
                    f"Invalid geometry/early tensor {name}; expected {shape}"
                )
        pose = self.ray_pose(ctx["pose_code"])[:, None]
        query = self.ray_query(q) + pose + self.ray_position(ctx["ray_code"])
        feature = self.grid_input_projection(
            self.grid_input_norm(ctx["crop_early"].float())
        )
        anchors = grid2(
            feature, ctx["anchor_uv"][:, :, None], ctx["source_early_hw"]
        ).squeeze(2)
        coverage = ctx["anchor_valid"].float()[..., None]
        anchors = (
            anchors * coverage
            + self.grid_position(ctx["anchor_xyz"])
            + self.grid_coverage(coverage)
        )
        volume = (
            anchors.reshape(batch, 8, 8, 8, 128).permute(0, 4, 1, 2, 3).contiguous()
        )
        volume = volume + self.volume_processor(volume)
        with torch.autocast(device_type=q.device.type, enabled=False):
            grid = ctx["volume_grid"][:, :, :, None].float()
            sampled = (
                F.grid_sample(
                    volume.float(),
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .squeeze(-1)
                .permute(0, 2, 3, 1)
            )
            cov = (
                F.grid_sample(
                    coverage.reshape(batch, 1, 8, 8, 8).float(),
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(-1)
            )
        depth = self.ray_depth(torch.cat([ctx["depth_code"], cov[..., None]], -1))
        values = sampled + depth
        valid = ctx["valid_ray"][:, :, None].expand(-1, -1, 16)
        values = torch.cat([values, self.ray_null.expand(batch, count, 1, -1)], 2)
        valid = torch.cat(
            [valid, torch.ones((batch, count, 1), device=q.device, dtype=torch.bool)], 2
        )
        candidates = values.shape[2]
        qq = self.ray_qnorm(query).reshape(batch * count, 1, 128)
        mm = self.ray_mnorm(values).reshape(batch * count, candidates, 128)
        padding = ~valid.reshape(batch * count, candidates)
        update = self.ray_attention(
            qq, mm, mm, key_padding_mask=padding, need_weights=False
        )[0].reshape(batch, count, 128)
        update = update + self.ray_ff(update)
        delta = self.ray_output(update).float() * ctx["valid_ray"][..., None]
        if self.capture:
            alpha = diagnostic_attention(
                self.ray_attention, qq, mm, mask=padding
            ).reshape(batch, count, candidates)
            self.last_diagnostics = {
                "fp32_recomputed_candidate_attention": alpha.cpu(),
                "query_delta_rms": delta.detach().square().mean(-1).sqrt().cpu(),
                "candidate_valid": valid.detach().cpu(),
                "grid_feature_norm": volume.detach()
                .float()
                .square()
                .mean(1)
                .sqrt()
                .cpu(),
            }
        return q + delta

    @torch.no_grad()
    def initialize_native_rms(self, merged, deepstack, ctx=None):
        target = self._streams(merged, deepstack)
        (primary, ds) = self(merged, deepstack, ctx)
        report = []
        for index, (want, have, projection) in enumerate(
            zip(target, (primary, *ds), self.output_projection)
        ):
            (wrms, prms) = (
                want.float().square().mean().sqrt(),
                have.float().square().mean().sqrt(),
            )
            if (
                not bool(torch.isfinite(wrms) & torch.isfinite(prms))
                or float(prms) <= 0
            ):
                raise ValueError("Cannot initialize from nonfinite/zero source RMS")
            gain = wrms / prms
            projection.weight.mul_(gain.to(projection.weight.dtype))
            projection.bias.mul_(gain.to(projection.bias.dtype))
            report.append(
                {
                    "stream": index,
                    "native_rms": float(wrms),
                    "reader_rms_before": float(prms),
                    "projection_initial_gain": float(gain),
                }
            )
        return report

    def specification(self):
        return {
            "schema": "raywrist_gr00t/v1",
            "config": asdict(self.config),
            "trainable_parameters": sum((p.numel() for p in self.parameters())),
            "identity_skip": False,
            "four_stream_outputs": True,
            "source_early_block": 5,
            "geometry_visibility_is_surface": False,
        }


class DualRayGridReader(nn.Module):

    def __init__(
        self, *, seed=2029, target_hw=(8, 8), source_hw=(8, 8), checkpoint_blocks=False
    ):
        super().__init__()
        target = hw(target_hw, "target_hw")
        self.config = ReaderConfig(
            tokens=math.prod(target), target_hw=target, source_hw=hw(source_hw)
        )
        self.readers = nn.ModuleDict(
            {
                side: RayReader(
                    self.config, seed=seed + index, checkpoint_blocks=checkpoint_blocks
                )
                for (index, side) in enumerate(("left", "right"))
            }
        )

    def forward_per_arm(self, arm, crop_merged, crop_ds3, ctx):
        if arm not in self.readers:
            raise ValueError("Arm must select left or right")
        if ctx.get("side", arm) != arm:
            raise ValueError("Context belongs to the other wrist")
        return self.readers[arm](crop_merged, crop_ds3, ctx)

    def specification(self):
        return {
            "schema": "raywrist_gr00t_dual/v1",
            "arms": ["left", "right"],
            "parameters_per_arm": {
                k: sum((p.numel() for p in m.parameters()))
                for (k, m) in self.readers.items()
            },
            "trainable_parameters": sum((p.numel() for p in self.parameters())),
            "target_hw": list(self.config.target_hw),
            "native_streams": 4,
            "independent_parameters": True,
            "native_crop_skip": False,
        }


class SingleRayGridReader(nn.Module):
    """Exactly one original arm reader; retained camera has no trainable writer."""

    def __init__(
        self,
        missing_side,
        *,
        seed=2029,
        target_hw=(8, 8),
        source_hw=(8, 8),
        checkpoint_blocks=False,
    ):
        super().__init__()
        if missing_side not in ("left", "right"):
            raise ValueError("A single missing side must be left or right")
        self.missing_side = self.side = missing_side
        target = hw(target_hw, "target_hw")
        self.config = ReaderConfig(
            tokens=math.prod(target), target_hw=target, source_hw=hw(source_hw)
        )
        self.reader_seed = seed + int(missing_side == "right")
        self.readers = nn.ModuleDict(
            {
                missing_side: RayReader(
                    self.config,
                    seed=self.reader_seed,
                    checkpoint_blocks=checkpoint_blocks,
                )
            }
        )

    def forward_per_arm(self, arm, crop_merged, crop_ds3, ctx):
        if arm != self.missing_side or set(self.readers) != {self.missing_side}:
            raise ValueError("A retained wrist cannot be passed to a generated writer")
        if ctx.get("side", arm) != arm:
            raise ValueError("Context belongs to the other wrist")
        return self.readers[arm](crop_merged, crop_ds3, ctx)

    def specification(self):
        return {
            "schema": "raywrist_gr00t_single/v1",
            "missing_side": self.missing_side,
            "retained_side": "right" if self.missing_side == "left" else "left",
            "reader_seed": self.reader_seed,
            "trainable_parameters": sum((p.numel() for p in self.parameters())),
            "target_hw": list(self.config.target_hw),
            "native_streams": 4,
            "single_generated_wrist": True,
            "retained_pre_LM_streams_passthrough": True,
            "native_crop_skip": False,
        }
