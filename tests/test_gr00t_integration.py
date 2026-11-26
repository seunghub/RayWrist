from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
from raywrist.integrations.gr00t import GrootRayWrist,unmerge_early


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity()for _ in range(6)])


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__();self.visual=Visual()

    def get_image_features(self,pixels,grid):
        early=self.visual.blocks[5](pixels)
        merged=tuple(torch.full((64,2048),float(i),device=early.device)for i in range(len(grid)))
        ds=[torch.cat([v+j*10 for v in merged])for j in range(3)]
        return merged,ds


class Recipient(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(2.));self.inner=Inner()
        self.backbone=SimpleNamespace(model=SimpleNamespace(model=self.inner))

    def forward(self,inputs):
        return self.inner.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.offset=torch.nn.Parameter(torch.tensor(3.));self.config=SimpleNamespace(target_hw=(8,8));self.context=None

    def forward(self,merged,deepstack,context):
        self.context=context
        return merged+self.offset,tuple(d+self.offset for d in deepstack)


def test_two_arms_keep_retained_views_and_reader_gradients():
    model=Recipient();left=Reader();right=Reader()
    adapter=GrootRayWrist(model,{'left':left,'right':right},{'left':1,'right':2},images_per_sample=3)
    inputs={'pixel_values':torch.zeros(6*256,1024),'image_grid_thw':torch.tensor([[1,16,16]]*6)}
    expected,expected_ds=model(inputs)
    actual,ds=adapter(inputs,{'left':{},'right':{}})
    for i in [0,3]:
        torch.testing.assert_close(actual[i],expected[i],rtol=0,atol=0)
        for a,b in zip(ds,expected_ds):torch.testing.assert_close(a[i*64:(i+1)*64],b[i*64:(i+1)*64],rtol=0,atol=0)
    for i in [1,2,4,5]:torch.testing.assert_close(actual[i],expected[i]+3)
    sum(x.mean()for x in actual).backward()
    assert left.offset.grad is not None and right.offset.grad is not None
    assert model.scale.grad is None and not model.scale.requires_grad
    assert left.context['crop_early'].shape==(2,256,1024)
    assert 'get_image_features' not in model.inner.__dict__


def test_missing_side_selection_and_exception_restore():
    model=Recipient();reader=Reader();adapter=GrootRayWrist(model,{'left':reader},{'left':1},images_per_sample=3)
    inputs={'pixel_values':torch.zeros(3*256,1024),'image_grid_thw':torch.tensor([[1,16,16]]*3)}
    expected,_=model(inputs);actual,_=adapter(inputs,{'left':{}})
    torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        with adapter.complete({'left':{}}):raise RuntimeError('caller failed')
    assert 'get_image_features' not in model.inner.__dict__ and not adapter.active
    assert len(model.inner.visual.blocks[5]._forward_hooks)==0


def test_early_unmerge_retains_native_pixel_addresses():
    # Every 2x2 native merge group contains its own raster indices.
    expected=torch.arange(16*16).reshape(16,16)
    grouped=expected.reshape(8,2,8,2).permute(0,2,1,3).reshape(256)
    early=grouped[:,None].expand(256,1024)
    actual=unmerge_early(early,[1,16,16])[:,0].reshape(16,16)
    assert torch.equal(actual,expected)
