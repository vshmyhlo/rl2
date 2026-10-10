import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.alphazero.model import PolicyValueNet


def test_policy_value_net_batched_observations() -> None:
    model = PolicyValueNet(num_actions=9, channels=4, num_blocks=1)
    observation = jnp.zeros((2, 3, 3, 2), jnp.bool_).at[1, 0, 0, 0].set(True)
    variables = model.init(jax.random.PRNGKey(0), observation)
    predict = jax.jit(model.apply)
    logits, values = predict(variables, observation)
    assert logits.shape == (2, 9)
    assert values.shape == (2,)
    assert logits.dtype == values.dtype == jnp.float32
    assert np.isfinite(logits).all()
    assert np.all(np.abs(values) <= 1)
    # Self-play inference uses boolean PGX boards; training stores float32.
    float_logits, float_values = predict(variables, observation.astype(jnp.float32))
    np.testing.assert_allclose(logits, float_logits)
    np.testing.assert_allclose(values, float_values)


@pytest.mark.parametrize("shape", [(3, 3, 2), (1, 2, 3, 3, 2)], ids=["unbatched", "extra-batch-axis"])
def test_policy_value_net_rejects_wrong_observation_rank(shape: tuple[int, ...]) -> None:
    model = PolicyValueNet(num_actions=9, channels=4, num_blocks=1)
    with pytest.raises(AssertionError, match="rank"):
        model.init(jax.random.PRNGKey(0), jnp.zeros(shape, jnp.float32))
