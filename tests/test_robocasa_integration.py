from dataclasses import dataclass
from types import SimpleNamespace
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from raywrist.geometry.robocasa import EEReaderGeometry
from raywrist.geometry.robocasa_source import FIELDS
from raywrist.integrations.robocasa import (
    CROPS, HEAD, TAG, IndependentReaders, RoboCasaProcessorMixin,
    attach_readers, dataset_modalities, load_joint_checkpoint,
)
from test_gr00t_integration import Inner, Reader


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.backbone = SimpleNamespace(model=SimpleNamespace(model=self.inner))
        self.language = torch.nn.Linear(2048, 8, bias=False).requires_grad_(False)
        self.action = torch.nn.Linear(8, 1)
        self.config = SimpleNamespace()
        self.last = None

    def forward(self, inputs):
        assert not any(key.startswith('raygrid_') for key in inputs)
        merged, ds = self.inner.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
        self.last = merged, ds
        features = torch.cat([torch.cat(merged), *ds]).mean(0)
        action = self.action(self.language(features))
        return {'loss': action.square().mean(), 'action_pred': action}

    def get_action(self, inputs, options=None):
        return Native.forward(self, inputs)


def small_joint(monkeypatch):
    import raywrist.integrations.robocasa as integration
    class SmallReaders(torch.nn.Module):
        def __init__(self, seed=42):
            super().__init__()
            self.readers = torch.nn.ModuleDict({'left': Reader(), 'right': Reader()})
    monkeypatch.setattr(integration, 'IndependentReaders', SmallReaders)
    return attach_readers(Native())


def packed_inputs(batch=2):
    result = {'pixel_values': torch.zeros(batch * 3 * 256, 1024),
              'image_grid_thw': torch.tensor([[1, 16, 16]] * (batch * 3))}
    result.update({'raygrid_' + key: torch.zeros(batch, 2, 1) for key in FIELDS})
    return result


def test_joint_gradients_native_scope_and_retained_head(monkeypatch):
    model = small_joint(monkeypatch)
    result = model(packed_inputs())
    result['loss'].backward()
    assert model.action.weight.grad.abs().sum() > 0
    assert model.language.weight.grad is None and not model.language.weight.requires_grad
    for side in ('left', 'right'):
        assert model.raygrid.readers[side].offset.grad.abs() > 0
    for index in (0, 3):
        torch.testing.assert_close(model.last[0][index], torch.full((64, 2048), float(index)))
    keys = model.state_dict()
    assert 'raygrid.readers.left.offset' in keys and 'raygrid.readers.right.offset' in keys
    assert not any('_raywrist_adapter' in key for key in keys)
    assert 'get_image_features' not in model.inner.__dict__
    assert not model.inner.visual.blocks[5]._forward_hooks
    model.get_action(packed_inputs())


def test_independent_reader_parameters_and_initialization():
    model = IndependentReaders()
    left, right = model.readers['left'], model.readers['right']
    assert sum(parameter.numel() for parameter in model.parameters()) == 30_010_112
    assert not ({p.data_ptr() for p in left.parameters()} & {p.data_ptr() for p in right.parameters()})
    assert not torch.equal(left.queries, right.queries)
    assert left.queries.dtype == right.queries.dtype == torch.float32


def test_joint_checkpoint_restores_both_readers_and_rejects_missing(monkeypatch, tmp_path):
    safe = pytest.importorskip('safetensors.torch')
    model = small_joint(monkeypatch)
    (tmp_path / 'config.json').write_text(json.dumps({'raygrid': model.config.raygrid}))
    state = model.state_dict()
    safe.save_file(state, str(tmp_path / 'model.safetensors'))
    restored = small_joint(monkeypatch)
    load_joint_checkpoint(restored, tmp_path)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    del state['raygrid.readers.right.offset']
    safe.save_file(state, str(tmp_path / 'model.safetensors'))
    with pytest.raises(ValueError, match='missing a trained weight'):
        load_joint_checkpoint(restored, tmp_path)


@dataclass
class Content:
    images: dict
    states: dict


class Replay:
    def __call__(self, **kwargs):
        return {**kwargs, 'replay': {'transforms': []}}

    def replay(self, **kwargs):
        return {**kwargs, 'replay': kwargs['saved_augmentations']}


class NativeProcessor:
    def __call__(self, messages):
        self.received = messages[0]['content']
        for images in self.received.images.values():
            self.train_image_transform(image=images[0])
        return {'state': np.zeros(132)}


def test_processor_uses_physical_state_and_dataset_loads_only_head():
    class Processor(RoboCasaProcessorMixin, NativeProcessor):
        training = True
    processor = Processor()
    transform = Replay()
    processor.train_image_transform = transform
    processor.modality_configs = {TAG: {'video': SimpleNamespace(modality_keys=[HEAD, *CROPS])}}
    state = np.zeros(29)
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    slices = {'left_arm': (0, 7), 'left_hand': (7, 13), 'right_arm': (13, 20),
              'right_hand': (20, 26), 'waist': (26, 29)}
    content = Content(images={HEAD: [image]}, states={key: state[a:b][None] for key, (a, b) in slices.items()})
    row = processor([{'content': content}])
    expected = EEReaderGeometry()
    crops, _ = expected.crops(image, state, layout='geometry29')
    context = expected.pair(state, layout='geometry29')
    for key, crop in zip(CROPS, crops):
        np.testing.assert_array_equal(processor.received.images[key][0], crop)
    for key in FIELDS:
        np.testing.assert_array_equal(row['raygrid_' + key], context[key])
    assert set(content.images) == {HEAD}
    assert processor.train_image_transform is transform
    assert dataset_modalities(processor)[TAG]['video'].modality_keys == [HEAD]
    assert processor.modality_configs[TAG]['video'].modality_keys == [HEAD, *CROPS]
