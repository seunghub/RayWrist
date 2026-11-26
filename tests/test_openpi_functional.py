from types import SimpleNamespace
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from flax import nnx
import raywrist.integrations.openpi as integration

class Image(nnx.Module):
    def __init__(self):self.calls=nnx.Variable(jnp.array(0))
    def __call__(self,x,**kwargs):
        self.calls.value+=1
        return x,{'encoder':{f'block{i:02d}':{'+mlp':x}for i in [5,12,19]}}

class Language(nnx.Module):
    def __call__(self,parts,**kwargs):
        if kwargs.get('method')=='embed':return jnp.repeat(parts[...,None].astype(jnp.float32),4,-1)
        prefix,suffix=parts
        signal=jnp.mean(prefix,axis=1,keepdims=True)
        out=None if suffix is None else suffix+signal
        return (prefix,out),None

class Model(nnx.Module):
    def __init__(self):
        self.pi05=True;self.action_horizon=2;self.action_dim=4
        self.PaliGemma=nnx.Dict(img=Image(),llm=Language())
    def embed_suffix(self,obs,x,t):return x,jnp.ones(x.shape[:2],bool),jnp.array([True,False]),None
    def action_out_proj(self,x):return x


def test_jitted_reader_gradient_reconstructs_native_nnx_state_inside_trace(monkeypatch):
    def replacement(visual,early,geometry,parameters,missing_sides):
        out=dict(visual);out['left_wrist_0_rgb']=out['left_wrist_0_rgb']+parameters['left']['offset'];return out
    monkeypatch.setattr(integration,'complete_visual_features',replacement)
    model=Model();adapter=integration.OpenPIRayWrist(model)
    observation=SimpleNamespace(images={'base_0_rgb':jnp.ones((1,2,4)),'left_wrist_0_rgb':jnp.ones((1,2,4))},image_masks={'base_0_rgb':jnp.array([True]),'left_wrist_0_rgb':jnp.array([True])},state=jnp.zeros((1,4)),tokenized_prompt=None,tokenized_prompt_mask=None)
    def objective(parameters):return adapter.loss(parameters,observation,{'left':{}},jnp.zeros((1,2,4)),jax.random.key(1),alignment_weight=0)[0]
    loss,gradient=jax.jit(jax.value_and_grad(objective))({'left':{'offset':jnp.array(.5)}})
    assert jnp.isfinite(loss)and jnp.isfinite(gradient['left']['offset'])and gradient['left']['offset']!=0
    assert int(model.PaliGemma.img.calls.value)==0
