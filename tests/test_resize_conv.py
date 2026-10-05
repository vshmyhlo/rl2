import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from rl2.resize_conv import ResizeConv


# Downsample, upsample, and identity resize; both compute dtypes.
@pytest.mark.parametrize("dtype,spatial_shape", [(jnp.float32, (3, 4)), (jnp.bfloat16, (9, 11)), (jnp.float32, (5, 7))])
def test_resize_then_convolve_without_norm_or_activation(
    dtype: jax.typing.DTypeLike, spatial_shape: tuple[int, int]
) -> None:
    x = jax.random.normal(jax.random.key(0), (2, 5, 7, 2)).astype(dtype)
    model = ResizeConv(3, spatial_shape, dtype=dtype)
    params = model.init(jax.random.key(1), x)["params"]
    assert set(params) == {"conv"}
    params["conv"]["bias"] = jnp.full((3,), -10.0)

    output = jax.jit(model.apply)({"params": params}, x)
    resized = jax.image.resize(x, (2, *spatial_shape, 2), method="bilinear", antialias=True)
    expected = nn.Conv(3, (3, 3), padding="SAME", dtype=dtype).apply({"params": params["conv"]}, resized)
    chex.assert_shape(output, (2, *spatial_shape, 3))
    chex.assert_type(output, dtype)
    np.testing.assert_allclose(output.astype(jnp.float32), expected.astype(jnp.float32), rtol=1e-5)
    assert np.all(np.asarray(output) < 0)
