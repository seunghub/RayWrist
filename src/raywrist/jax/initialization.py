"""Source-crop writer initialization from supplement Eq. S9."""
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze


def calibrate_writer(parameters,source_features,initial_output):
    """Calibrate the final writer using training crop features, before updates.

    Inputs have shape [examples, tokens, native_width]. No wrist targets or
    action labels are used. Apply once to freshly initialized reader parameters.
    """
    source=jnp.asarray(source_features,jnp.float32)
    output=jnp.asarray(initial_output,jnp.float32)
    if source.ndim!=3 or output.shape!=source.shape:
        raise ValueError('Source and initial writer output must share [B,N,D]')
    mu_source=source.mean((0,1));mu_output=output.mean((0,1))
    source_rms=jnp.sqrt(jnp.mean(jnp.square(source-mu_source)))
    output_rms=jnp.sqrt(jnp.mean(jnp.square(output-mu_output)))
    values=np.asarray(jax.device_get(jnp.stack([source_rms,output_rms])))
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Writer calibration requires finite nonzero centered RMS')
    gain=source_rms/output_rms
    result=unfreeze(parameters)
    result['output_projection']['kernel']=result['output_projection']['kernel']*gain
    result['output_projection']['bias']=mu_source-gain*mu_output
    return freeze(result) if isinstance(parameters,FrozenDict) else result
