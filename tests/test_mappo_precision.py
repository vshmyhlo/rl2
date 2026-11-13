"""MAPPO compute dtype propagation and float32 training state."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.linen.module import InterceptorContext
from flax.training.train_state import TrainState
from test_lux_encoder import observation as observation  # noqa: PLC0414
from test_train_mappo import config as config  # noqa: PLC0414

from rl2 import train_mappo as ppo
from rl2.actions import DiscreteSpace, HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxConfig, LuxObservation
from rl2.model.config import LuxModelConfig
from rl2.utils import gae


@pytest.mark.usefixtures("ppo_mesh")
@pytest.mark.parametrize("model_type", ["generic", "lux"])
def test_bfloat16_module_tree_and_ppo_update(
    model_type: str,
    config: ppo.Config,
    observation: LuxObservation[jax.Array],  # unbatched observation leaves
) -> None:
    if model_type == "lux":
        config = replace(
            config,
            env=LuxConfig(),
            model=LuxModelConfig(conv_channels=4, num_blocks=1, heads=2, head_dim=2, attention_implementation="xla"),
            dtype="bfloat16",
        )
        inputs = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (1, 2, *leaf.shape)), observation)  # (B, T, ...)
        space = HierarchicalMultiDiscreteSpace(nvec=(14,) * 3, group_sizes=(1, 4, 9))
        masks = jnp.broadcast_to(inputs.self.units_mask[..., None], (1, 2, 3, 14)).at[..., 0].set(True)  # (B, T, U, A)
    else:
        config = replace(
            config,
            model=replace(config.model, recurrent=ppo.LSTMConfig(hidden_size=4, num_layers=2, intermediate_size=8)),
            dtype="bfloat16",
        )
        inputs = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4) / 8  # (B, T, O)
        space = DiscreteSpace(3)
        masks = jnp.ones((1, 2, 3), dtype=bool).at[..., 2].set(False)  # (B, T, A)
    model = ppo.make_actor_critic(space, config.model, dtype=jnp.dtype(config.dtype))
    starts = jnp.array([[True, False]])  # (B, T)
    memory = model.initial_carry(1)  # per-layer leaves: (B, H)
    visited: set[type[nn.Module]] = set()

    def check_dtype(
        next_fun: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], context: InterceptorContext
    ) -> Any:
        module = context.module
        expected_dtype = (
            jnp.float32 if isinstance(module, nn.Embed) and module.name == "tile_embedding" else jnp.bfloat16
        )
        if hasattr(module, "dtype"):
            assert jnp.dtype(module.dtype) == expected_dtype, context.module.path
            visited.add(type(module))
        result = next_fun(*args, **kwargs)
        if context.method_name == "__call__" and isinstance(
            module, (nn.Dense, nn.DenseGeneral, nn.Conv, nn.Embed, nn.LayerNorm, nn.RMSNorm)
        ):
            assert result.dtype == expected_dtype, context.module.path
        return result

    with nn.intercept_methods(check_dtype):
        (output, carry), variables = jax.jit(model.init_with_output)(jax.random.key(7), inputs, memory, starts, masks)
    assert {nn.Dense, nn.LayerNorm, nn.RMSNorm} <= visited
    if model_type == "lux":
        assert {nn.Conv, nn.Embed, nn.DenseGeneral} <= visited
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves((variables, carry)))
    assert all(leaf.shape == (1, 4) for leaf in jax.tree.leaves(carry))
    assert output.value.shape == (1, 2) and output.value.dtype == jnp.float32
    assert output.policy.logits.shape == masks.shape and output.policy.logits.dtype == jnp.float32
    actions, old_log_prob = output.policy.sample_and_log_prob(key=jax.random.key(8))  # (B, T, *C); (B, T)
    assert actions.shape == masks.shape[:-1]
    assert old_log_prob.shape == (1, 2) and old_log_prob.dtype == jnp.float32
    assert output.policy.entropy().dtype == jnp.float32
    advantage, returns = gae(
        jnp.array([[0.25], [1.0]]),  # (T, B)
        jnp.array([[False], [True]]),  # (T, B)
        output.value.T,  # (T, B)
        jnp.zeros(1, dtype=jnp.float32),  # (B,)
        config.gamma,
        config.gae_lambda,
    )  # each (T, B)
    assert advantage.shape == returns.shape == (2, 1)
    assert advantage.dtype == returns.dtype == jnp.float32
    batch = ppo.PPOBatch(
        observation=inputs,  # leaves: (B, T, ...)
        actions_mask=masks,  # (B, T, *C, A)
        action=actions,  # (B, T, *C)
        old_log_prob=old_log_prob,  # (B, T)
        value=output.value,  # (B, T)
        advantage=advantage.T,  # (B, T)
        return_target=returns.T,  # (B, T)
        model_carry=memory,  # leaves: (B, H)
        episode_start=starts,  # (B, T)
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate, eps=1e-5)),
    )
    updated, metrics = jax.jit(ppo.update, static_argnames="config")(state, batch, config)
    assert int(updated.step) == 1
    assert all(leaf.shape == () and leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(metrics))
    assert float(metrics.approx_kl) < 1e-5
    assert float(metrics.value_loss) > 0
    for leaf in jax.tree.leaves((updated.params, updated.opt_state)):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32
            np.testing.assert_array_less(jnp.abs(leaf), 100)
    # An update must reach every top-level parameter group.
    for name in state.params:
        before = jax.tree.leaves(state.params[name])  # parameter-shaped leaves
        after = jax.tree.leaves(updated.params[name])  # parameter-shaped leaves
        assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True)), name
