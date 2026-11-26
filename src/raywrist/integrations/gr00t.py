"""RayWrist at the official GR00T N1.7 Qwen3-VL image-feature interface."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from raywrist.torch.native_path import override_image_features
from raywrist.torch.losses import conditioning_alignment


def unmerge_early(tokens, grid):
    """Undo Qwen's 2x2 spatial-merge ordering without interpolating features."""
    time, height, width = map(int, grid)
    if time != 1 or height % 2 or width % 2 or tokens.shape != (height*width, 1024):
        raise ValueError('Expected one native Qwen image with 1024-channel early features')
    return tokens.reshape(height//2, width//2, 2, 2, 1024).permute(0, 2, 1, 3, 4).reshape(height*width, 1024)


class GrootRayWrist(nn.Module):
    """Complete selected native image slots while keeping retained features intact.

    `slot_by_side` indexes images inside each sample, after native image packing.
    Missing slots must contain external-derived tool crops. Other slots contain
    the actual retained views. Geometry is computed before state normalization.
    The upstream processor, language model and action head keep their APIs.
    """
    def __init__(self, model, readers, slot_by_side, *, images_per_sample=2, freeze_recipient=True):
        super().__init__()
        if not readers or set(readers) != set(slot_by_side):
            raise ValueError('Each missing wrist needs one reader and one native slot')
        slots=list(slot_by_side.values())
        if len(set(slots)) != len(slots) or any(type(i) is not int or not 0 <= i < images_per_sample for i in slots):
            raise ValueError('Missing wrist slots must be distinct native image indices')
        object.__setattr__(self, 'recipient', model)
        self.readers=nn.ModuleDict(readers)
        self.slot_by_side=dict(slot_by_side)
        self.images_per_sample=images_per_sample
        self.active=False
        if freeze_recipient:
            model.requires_grad_(False)
        self.inner=model.backbone.model.model
        if len(self.inner.visual.blocks) < 6:
            raise ValueError('GR00T N1.7 requires the Qwen block-5 visual interface')

    def _complete(self, original, geometry, pixels, grid, *args, **kwargs):
        if any(p.requires_grad for p in self.inner.visual.parameters()):
            raise ValueError('The source visual encoder must remain frozen')
        if grid.ndim != 2 or grid.shape[1] != 3 or len(grid)%self.images_per_sample:
            raise ValueError('Expected sample-major native image_grid_thw')
        batch=len(grid)//self.images_per_sample
        captures=[]
        handle=self.inner.visual.blocks[5].register_forward_hook(lambda module, arguments, output: captures.append(output.detach()))
        try:
            with torch.no_grad():
                merged, deepstack=original(pixels, grid, *args, **kwargs)
        finally:
            handle.remove()
        if len(captures)!=1 or len(merged)!=len(grid) or len(deepstack)!=3:
            raise ValueError('The native merger/DeepStack interface differs from the pinned revision')
        dimensions=grid.tolist()
        sizes=[math.prod(d) for d in dimensions]
        if captures[0].shape != (sum(sizes),1024):
            raise ValueError('Unexpected block-5 feature packing')
        early=[unmerge_early(v,d) for v,d in zip(captures[0].split(sizes),dimensions)]
        lengths=[len(t) for t in merged]
        if any(n!=size//4 for n,size in zip(lengths,sizes)):
            raise ValueError('Native visual merger does not use 2x2 spatial merging')
        streams=[list(merged),*(list(v.split(lengths)) for v in deepstack)]
        for side, reader in self.readers.items():
            indices=[b*self.images_per_sample+self.slot_by_side[side] for b in range(batch)]
            if len({tuple(dimensions[i]) for i in indices}) != 1:
                raise ValueError('Group native source rasters of the same shape in each batch')
            ctx=dict(geometry[side])
            ctx['crop_early']=torch.stack([early[i] for i in indices])
            height,width=map(int,dimensions[indices[0]][1:])
            ctx.setdefault('source_early_hw',(height,width))
            ctx.setdefault('source_hw',(height//2,width//2))
            ctx.setdefault('target_hw',reader.config.target_hw)
            source=torch.stack([streams[0][i] for i in indices])
            ds=tuple(torch.stack([stream[i] for i in indices]) for stream in streams[1:])
            out, out_ds=reader(source,ds,ctx)
            if out.shape != source.shape or len(out_ds)!=3 or any(v.shape!=source.shape for v in out_ds):
                raise ValueError('Writer outputs must match the original native image slots')
            for si,output in enumerate((out,*out_ds)):
                for b,i in enumerate(indices):
                    streams[si][i]=output[b].to(device=streams[si][i].device,dtype=streams[si][i].dtype)
        return tuple(streams[0]),[torch.cat(stream,0) for stream in streams[1:]]

    @contextmanager
    def complete(self, geometry):
        """Apply to one upstream forward or get_action call; restore on exit."""
        if self.active:
            raise RuntimeError('Use a separate adapter instance for concurrent calls')
        if set(geometry)!=set(self.readers):
            raise ValueError('Geometry must correspond exactly to the missing wrists')
        original=self.inner.get_image_features
        def replacement(pixels,grid,*args,**kwargs):
            return self._complete(original,geometry,pixels,grid,*args,**kwargs)
        self.active=True
        try:
            with override_image_features(self.inner,replacement):
                yield self
        finally:
            self.active=False

    def forward(self, inputs, geometry):
        """The original native action objective, with completed visual features."""
        with self.complete(geometry):
            return self.recipient(inputs)

    @torch.no_grad()
    def teacher_conditioning(self, full_view_inputs):
        """Full-view teacher C, before action/noise generation, with no adapter."""
        if self.active:
            raise RuntimeError('Compute the full-view teacher outside student completion')
        backbone_inputs,_=self.recipient.prepare_input(full_view_inputs)
        encoded=self.recipient.backbone(backbone_inputs)
        return self.recipient.action_head.process_backbone_output(encoded)

    @staticmethod
    def training_loss(student_output, teacher_output):
        """Native action loss plus valid-token conditioning alignment (lambda=1)."""
        c=student_output['backbone_features']
        mask=teacher_output['backbone_attention_mask']
        align=conditioning_alignment(c,teacher_output['backbone_features'],mask)
        return student_output['loss']+align,{'action':student_output['loss'],'alignment':align}
