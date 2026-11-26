"""Head-camera-only GR1 inputs and joint RayWrist / GR00T learning."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import types

import numpy as np
import torch
from torch import nn

from raywrist.geometry.robocasa import EEReaderGeometry
from raywrist.geometry.robocasa_source import FIELDS, n17_replay_affine
from raywrist.integrations.gr00t import GrootRayWrist
from raywrist.torch import RayWrist

TAG = 'robocasa_gr1_tabletop'
HEAD = 'ego_view_bg_crop_pad_res256_freq20'
CROPS = ('raygrid_left', 'raygrid_right')
PARTS = {'left_arm': (0, 7), 'right_arm': (22, 29),
         'left_hand': (7, 13), 'right_hand': (29, 35), 'waist': (41, 44)}


class ReplayTap:
    """Record the native transform's replay without drawing new augmentations."""
    def __init__(self, native):
        if not hasattr(native, 'replay'):
            import albumentations as A
            native = A.ReplayCompose(native.transforms, p=native.p)
        self.native = native
        self.last = None

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, **kwargs):
        result = self.native(**kwargs)
        self.last = result['replay']
        return result

    def replay(self, **kwargs):
        self.last = kwargs['saved_augmentations']
        return self.native.replay(**kwargs)


class RoboCasaProcessorMixin:
    """Add physical-state crops before native state regularization and encoding."""
    def __call__(self, messages):
        content = messages[0]['content']
        state = np.concatenate([np.asarray(content.states[key])[-1] for key in
                                ('left_arm', 'left_hand', 'right_arm', 'right_hand', 'waist')])
        geometry = EEReaderGeometry()
        crops, _ = geometry.crops(np.asarray(content.images[HEAD][-1]), state, layout='geometry29')
        images = dict(content.images)
        images.update({key: [crop] for key, crop in zip(CROPS, crops)})
        content = replace(content, images=images)
        transform_name = 'train_image_transform' if self.training else 'eval_image_transform'
        native = getattr(self, transform_name)
        tap = ReplayTap(native)
        setattr(self, transform_name, tap)
        try:
            row = super().__call__([{**messages[0], 'content': content}])
        finally:
            setattr(self, transform_name, native)
        if tap.last is None:
            raise ValueError('Native image transform did not return spatial replay')
        affine = n17_replay_affine(tap.last)
        context = geometry.pair(state, layout='geometry29', affine=affine)
        row.update({'raygrid_' + key: value for key, value in context.items()})
        return row

    def save_pretrained(self, directory):
        paths = super().save_pretrained(directory)
        path = Path(directory) / 'processor_config.json'
        config = json.loads(path.read_text())
        config['processor_class'] = 'Gr00tN1d7Processor'
        path.write_text(json.dumps(config, indent=2) + '\n')
        return paths


def processor_from_checkpoint(checkpoint, *, enabled=True, **kwargs):
    """Load native statistics, modality definitions and image transforms."""
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    class RoboCasaProcessor(RoboCasaProcessorMixin, Gr00tN1d7Processor):
        pass

    cls = RoboCasaProcessor if enabled else Gr00tN1d7Processor
    processor = cls.from_pretrained(checkpoint, **kwargs)
    if TAG not in processor.modality_configs:
        raise ValueError('The checkpoint processor must contain the GR1 Tabletop embodiment')
    processor.modality_configs[TAG]['video'].modality_keys = [HEAD, *CROPS] if enabled else [HEAD]
    return processor


def dataset_modalities(processor):
    """The dataset loads only recorded head RGB; the processor creates crops."""
    result = deepcopy(processor.modality_configs[TAG])
    result['video'].modality_keys = [HEAD]
    return {TAG: result}


class IndependentReaders(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.readers = nn.ModuleDict({side: RayWrist(seed=seed + i)
                                     for i, side in enumerate(('left', 'right'))})


def split_geometry(inputs, device):
    values = dict(inputs)
    pair = {key: values.pop('raygrid_' + key).to(
        device=device, dtype=torch.bool if key in ('anchor_valid', 'valid_ray') else torch.float32
    ) for key in FIELDS}
    if any(value.ndim < 2 or value.shape[1] != 2 for value in pair.values()):
        raise ValueError('RoboCasa geometry must have sample-major left/right packing')
    return values, {side: {key: value[:, index] for key, value in pair.items()}
                    for index, side in enumerate(('left', 'right'))}


def attach_readers(model, *, seed=42):
    """Register two readers in the native model's optimizer and checkpoint."""
    if hasattr(model, 'raygrid'):
        raise ValueError('This model already has wrist readers')
    model.raygrid = IndependentReaders(seed)
    parameter = next(model.backbone.model.model.parameters(), None)
    if parameter is not None:
        model.raygrid.to(device=parameter.device)
    adapter = GrootRayWrist(model, dict(model.raygrid.readers), {'left': 1, 'right': 2},
                           images_per_sample=3, freeze_recipient=False)
    native_forward, native_get_action = model.forward, model.get_action

    def run(function, inputs, *args, **kwargs):
        device = next(model.raygrid.parameters()).device
        inputs, geometry = split_geometry(inputs, device)
        with adapter.complete(geometry):
            return function(inputs, *args, **kwargs)

    def forward(self, inputs):
        return run(native_forward, inputs)

    def get_action(self, inputs, options=None):
        return run(native_get_action, inputs, options=options)

    model.forward = types.MethodType(forward, model)
    model.get_action = types.MethodType(get_action, model)
    object.__setattr__(model, '_raywrist_adapter', adapter)
    model.config.raygrid = {
        'reader_topology': 'independent_dual', 'shared_between_arms': False,
        'seeds_by_arm': {'left': seed, 'right': seed + 1},
        'view_order': [HEAD, *CROPS], 'teacher': False,
        'geometry': {'mode': 'ee_virtual_head_intrinsics', 'origin_in_ee_m': [0., 0., -.06],
                     'forward_frame': 'grip', 'volume_frame': 'virtual'},
    }
    return model


def load_model(base_checkpoint, *, enabled=True, attention='sdpa', **kwargs):
    """Use the native projector/action scope, frozen vision and language, FP32 trainables."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    config = Gr00tN1d7Config.from_pretrained(base_checkpoint)
    for key, value in dict(tune_llm=False, tune_visual=False, tune_projector=True,
                           tune_diffusion_model=True, tune_vlln=True, load_bf16=True,
                           backbone_trainable_params_fp32=True).items():
        setattr(config, key, value)
    config.use_flash_attention = attention == 'flash_attention_2'
    loading_kwargs = dict(kwargs.pop('transformers_loading_kwargs', {}))
    loading_kwargs.setdefault('trust_remote_code', True)
    if attention != 'flash_attention_2':
        loading_kwargs['attn_implementation'] = attention
    model, loading = Gr00tN1d7.from_pretrained(
        base_checkpoint, config=config, torch_dtype=torch.bfloat16,
        transformers_loading_kwargs=loading_kwargs, output_loading_info=True, **kwargs)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(('Native initialization mismatch', loading))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return attach_readers(model) if enabled else model


def load_joint_checkpoint(model, checkpoint):
    """Restore all native and reader tensors, allowing only omitted tied aliases."""
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / 'config.json').read_text()).get('raygrid') or {}
    if metadata.get('shared_between_arms') is not False or metadata.get('reader_topology') != 'independent_dual':
        raise ValueError('An independent dual-reader checkpoint is required')
    if (checkpoint / 'model.safetensors.index.json').exists():
        from transformers.modeling_utils import load_sharded_checkpoint
        result = load_sharded_checkpoint(model, checkpoint, strict=False, prefer_safe=True)
    else:
        from safetensors.torch import load_file
        result = model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=False)
    if result.unexpected_keys:
        raise ValueError(('Unexpected checkpoint weights', result.unexpected_keys))
    state, missing = model.state_dict(), set(result.missing_keys)
    for key in missing:
        tensor = state[key]
        alias = any(name not in missing and value.data_ptr() == tensor.data_ptr()
                    and value.shape == tensor.shape and value.stride() == tensor.stride()
                    for name, value in state.items())
        if key.startswith('raygrid.') or not alias:
            raise ValueError('Checkpoint is missing a trained weight: ' + key)
    return model


def gr1_message(raw_state, head_rgb, prompt, actions=None):
    """Build a native GR1 message from physical raw44 state and optional labels."""
    from gr00t.data.types import VLAStepData
    from gr00t.data.embodiment_tags import EmbodimentTag
    state, image = np.asarray(raw_state, np.float32), np.asarray(head_rgb)
    if state.shape != (44,) or not np.isfinite(state).all():
        raise ValueError('Expected finite raw44 physical state')
    if np.any(state[19:22]):
        raise ValueError('The GR1 Tabletop calibration uses the benchmark fixed neck')
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError('Expected native pre-augmentation uint8 head RGB [256,256,3]')
    if actions is not None:
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != 44 or not np.isfinite(actions).all():
            raise ValueError('Expected finite physical action labels [horizon,44]')
    content = VLAStepData(
        images={HEAD: [image]}, states={key: state[a:b][None] for key, (a, b) in PARTS.items()},
        actions={} if actions is None else {key: actions[:, a:b] for key, (a, b) in PARTS.items()},
        text=prompt, embodiment=EmbodimentTag.ROBOCASA_GR1_TABLETOP)
    return [{'content': content}]


class RoboCasaPolicy:
    """Native action sampling and physical decoding from head RGB and raw44 state."""
    def __init__(self, model, processor, device='cuda:0'):
        self.device = torch.device(device)
        self.model = model.to(device).eval()
        self.processor = processor
        self.processor.eval()
        self.modality_configs = dataset_modalities(processor)[TAG]

    @classmethod
    def from_pretrained(cls, base_checkpoint, joint_checkpoint, *, device='cuda:0', **kwargs):
        model = load_joint_checkpoint(load_model(base_checkpoint, **kwargs), joint_checkpoint)
        return cls(model, processor_from_checkpoint(joint_checkpoint), device)

    @torch.inference_mode()
    def _predict(self, messages, options=None):
        from gr00t.data.embodiment_tags import EmbodimentTag
        inputs = self.processor.collator([self.processor(message) for message in messages])['inputs']
        if 'action' in inputs or 'action_mask' in inputs:
            raise ValueError('Inference must not contain action labels')
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            prediction = self.model.get_action(inputs, options=options)['action_pred'].float().cpu().numpy()
        states = {key: np.stack([message[0]['content'].states[key] for message in messages]) for key in PARTS}
        actions = self.processor.decode_action(prediction, EmbodimentTag.ROBOCASA_GR1_TABLETOP, states)
        return {key: np.asarray(actions[key], np.float32)[:, :8] for key in PARTS}

    def infer(self, raw_state, head_rgb, prompt):
        actions = self._predict([gr1_message(raw_state, head_rgb, prompt)])
        return {'action.' + key: value[0] for key, value in actions.items()}

    def get_action(self, observation, options=None):
        """The native batched policy protocol, including simulator-wrapper use."""
        from gr00t.data.types import VLAStepData
        from gr00t.data.embodiment_tags import EmbodimentTag
        head = np.asarray(observation['video'][HEAD])
        if head.ndim != 5 or head.shape[1:] != (1, 256, 256, 3) or head.dtype != np.uint8:
            raise ValueError('Expected head RGB [batch,1,256,256,3]')
        batch = len(head)
        language_key = self.modality_configs['language'].modality_keys[0]
        prompts = observation['language'][language_key]
        if len(prompts) != batch or any(len(row) != 1 or not isinstance(row[0], str) for row in prompts):
            raise ValueError('Expected one native task instruction per sample')
        states = {}
        for key, (a, b) in PARTS.items():
            value = np.asarray(observation['state'][key])
            if value.shape != (batch, 1, b - a) or not np.isfinite(value).all():
                raise ValueError('Invalid native physical state for ' + key)
            states[key] = value
        messages = [[{'content': VLAStepData(
            images={HEAD: [head[i, 0]]}, states={key: value[i] for key, value in states.items()},
            actions={}, text=prompts[i][0], embodiment=EmbodimentTag.ROBOCASA_GR1_TABLETOP,
        )}] for i in range(batch)]
        return self._predict(messages, options), {}

    def get_modality_config(self):
        return self.modality_configs

    def reset(self, options=None):
        return {}
