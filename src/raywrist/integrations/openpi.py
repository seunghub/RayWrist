"""Native OpenPI pi0.5 prefix construction, action objective and Euler sampler.

Native mask and action equations follow Physical-Intelligence/openpi, Apache-2.0.
Observations passed here have already undergone native preprocessing; source
crops and geometry must use its corresponding spatial transforms.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from raywrist.jax.interfaces import complete_visual_features, WRIST_KEYS
from raywrist.jax.reader import EARLY_BLOCKS
from raywrist.jax.losses import conditioning_alignment


def make_attn_mask(input_mask,mask_ar):
    cumulative=jnp.cumsum(jnp.broadcast_to(mask_ar,input_mask.shape),axis=1)
    valid=input_mask[:,None,:]*input_mask[:,:,None]
    return jnp.logical_and(cumulative[:,None,:]<=cumulative[:,:,None],valid)


class OpenPIRayWrist:
    """A functional reader path around an already loaded native pi0.5 model.

    Differentiate `parameters`, rather than the recipient model. Each arm has
    its own reader tree. Image order, native masks, suffix and action horizon
    are owned by the supplied upstream model.
    """
    def __init__(self,model,missing_sides=('left',)):
        if not model.pi05:
            raise ValueError('This interface targets the native pi0.5 model')
        if len(set(missing_sides))!=len(missing_sides) or any(s not in WRIST_KEYS for s in missing_sides):
            raise ValueError('Expected distinct left/right missing wrists')
        if isinstance(model,nnx.Module):
            self._graph,self._frozen=nnx.split(model)
            self._plain_model=None
        else:
            self._graph,self._frozen=None,None
            self._plain_model=model
        self.missing_sides=tuple(missing_sides)

    @property
    def model(self):
        # Rebuild inside the current JAX trace. Native ToNNX modules may update
        # internal RNG/state counters even during a deterministic forward.
        return nnx.merge(self._graph,self._frozen) if self._graph is not None else self._plain_model

    def encode_images(self,observation):
        visual={};early={}
        for name,image in observation.images.items():
            value,details=self.model.PaliGemma.img(image,train=False)
            visual[name]=jax.lax.stop_gradient(value)
            for side in self.missing_sides:
                if name==WRIST_KEYS[side]:
                    early[side]=tuple(jax.lax.stop_gradient(details['encoder'][f'block{b:02d}']['+mlp']) for b in EARLY_BLOCKS)
        if set(early)!=set(self.missing_sides):
            raise ValueError('A native image slot is required for each missing wrist')
        return visual,early

    def prefix(self,parameters,observation,geometry,*,features=None,adapt=True):
        if features is None:
            features=self.encode_images(observation)
        visual,early=features
        if adapt:
            visual=complete_visual_features(visual,early,geometry,parameters,self.missing_sides)
        tokens=[];masks=[];autoregressive=[]
        for name in observation.images:
            value=visual[name]
            tokens.append(value)
            masks.append(jnp.repeat(observation.image_masks[name][:,None],value.shape[1],axis=1))
            autoregressive.extend([False]*value.shape[1])
        if observation.tokenized_prompt is not None:
            value=self.model.PaliGemma.llm(observation.tokenized_prompt,method='embed')
            tokens.append(value);masks.append(observation.tokenized_prompt_mask)
            autoregressive.extend([False]*value.shape[1])
        return jnp.concatenate(tokens,1),jnp.concatenate(masks,1),jnp.array(autoregressive)

    def teacher_cache(self,full_view_observation):
        prefix,mask,ar=self.prefix(None,full_view_observation,None,adapt=False)
        _,cache=self.model.PaliGemma.llm([prefix,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,1)-1)
        return jax.tree.map(jax.lax.stop_gradient,cache)

    def loss(self,parameters,observation,geometry,actions,key,*,teacher=None,features=None,alignment_weight=1.0):
        """Native flow objective plus full-view prefix-K/V alignment."""
        _,noise_key,time_key=jax.random.split(key,3)
        noise=jax.random.normal(noise_key,actions.shape)
        time=jax.random.beta(time_key,1.5,1,actions.shape[:-2])*.999+.001
        x=time[...,None,None]*noise+(1-time[...,None,None])*actions
        target=noise-actions
        p,pm,pa=self.prefix(parameters,observation,geometry,features=features)
        suffix,sm,sa,cond=self.model.embed_suffix(observation,x,time)
        mask=jnp.concatenate([pm,sm],1);ar=jnp.concatenate([pa,sa],0)
        (_,out),cache=self.model.PaliGemma.llm([p,suffix],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,1)-1,adarms_cond=[None,cond])
        velocity=self.model.action_out_proj(out[:,-self.model.action_horizon:])
        action_loss=jnp.mean(jnp.square(velocity-target))
        if alignment_weight and teacher is None:
            raise ValueError('Alignment training requires the detached full-view teacher cache')
        alignment=conditioning_alignment(cache,teacher,pm) if alignment_weight else jnp.array(0.,jnp.float32)
        return action_loss+alignment_weight*alignment,{'action':action_loss,'alignment':alignment}

    def sample_actions(self,parameters,observation,geometry,noise,*,num_steps=10,adapt=True):
        """Original prefix-cached Euler sampling, with explicit native noise."""
        if type(num_steps)is not int or num_steps<1:
            raise ValueError('num_steps must be a positive integer')
        dt=-1./num_steps
        p,pm,pa=self.prefix(parameters,observation,geometry,adapt=adapt)
        _,cache=self.model.PaliGemma.llm([p,None],mask=make_attn_mask(pm,pa),positions=jnp.cumsum(pm,1)-1)
        batch=observation.state.shape[0]
        def step(carry):
            x,t=carry
            suffix,sm,sa,cond=self.model.embed_suffix(observation,x,jnp.broadcast_to(t,batch))
            prefix_mask=jnp.repeat(pm[:,None],sm.shape[1],axis=1)
            mask=jnp.concatenate([prefix_mask,make_attn_mask(sm,sa)],axis=2)
            positions=jnp.sum(pm,-1)[:,None]+jnp.cumsum(sm,-1)-1
            (_,out),_=self.model.PaliGemma.llm([None,suffix],mask=mask,positions=positions,kv_cache=cache,adarms_cond=[None,cond])
            velocity=self.model.action_out_proj(out[:,-self.model.action_horizon:])
            return x+dt*velocity,t+dt
        return jax.lax.while_loop(lambda c:c[1]>=-dt/2,step,(noise,jnp.array(1.,jnp.float32)))[0]
