import numpy as np
import pytest
jax=pytest.importorskip('jax')
import jax.numpy as jnp
from raywrist.jax.initialization import calibrate_writer


def test_writer_matches_source_mean_and_centered_scale_without_changing_other_parameters():
    rng=np.random.default_rng(7)
    inputs=jnp.array(rng.normal(size=(4,8,3)),jnp.float32)
    kernel=jnp.array(rng.normal(size=(3,5)),jnp.float32)
    output=inputs@kernel
    source=jnp.array(rng.normal(4,3,size=(4,8,5)),jnp.float32)
    params={'output_projection':{'kernel':kernel,'bias':jnp.zeros(5)},'ray_output':{'kernel':jnp.zeros((2,3))}}
    result=calibrate_writer(params,source,output)
    new=inputs@result['output_projection']['kernel']+result['output_projection']['bias']
    np.testing.assert_allclose(new.mean((0,1)),source.mean((0,1)),rtol=1e-6,atol=1e-6)
    rms=lambda x:jnp.sqrt(jnp.mean((x-x.mean((0,1)))**2))
    np.testing.assert_allclose(rms(new),rms(source),rtol=1e-6)
    np.testing.assert_array_equal(result['ray_output']['kernel'],params['ray_output']['kernel'])
    np.testing.assert_array_equal(params['output_projection']['kernel'],kernel)
