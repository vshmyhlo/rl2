from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from rl2.configuration import BaseConfig
from rl2.rnd_checkpoints import TrainingProgress, checkpoint_manager, restore_checkpoint, save_checkpoint
from rl2.utils import RunningMeanStd


def test_rnd_checkpoint_restores_models_normalization_and_rng(tmp_path: Path) -> None:
    state = TrainState.create(apply_fn=jnp.add, params={"weight": jnp.ones(2)}, tx=optax.adam(0.01))
    updated = state.apply_gradients(grads={"weight": jnp.ones(2)})
    target = TrainState.create(apply_fn=jnp.add, params={"weight": jnp.zeros(2)}, tx=optax.set_to_zero())
    progress = TrainingProgress(
        state=updated,
        predictor=updated,
        target=target,
        key=jax.random.key(1),
        rnd_key=jax.random.key(2),
        iteration=3,
        episodes=7,
        return_sum=9.0,
        length_sum=11,
    )
    obs, reward = RunningMeanStd((2,)), RunningMeanStd()
    obs.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
    reward.update(np.array([2.0, 5.0]))
    rng = np.random.default_rng(3)
    with checkpoint_manager(str(tmp_path)) as manager:
        assert restore_checkpoint(manager, progress, obs, reward, rng) is progress
        save_checkpoint(manager, progress, obs, reward, rng, BaseConfig(run_id="rnd", ckpt_dir=str(tmp_path)))
        new_obs, new_reward = RunningMeanStd((2,)), RunningMeanStd()
        new_rng = np.random.default_rng(0)
        restored = restore_checkpoint(
            manager, progress._replace(state=state, predictor=state), new_obs, new_reward, new_rng
        )
    chex.assert_trees_all_equal(restored, progress)
    for original, loaded in ((obs, new_obs), (reward, new_reward)):
        np.testing.assert_array_equal(original.mean, loaded.mean)
        np.testing.assert_array_equal(original.var, loaded.var)
        assert original.count == loaded.count
    np.testing.assert_array_equal(rng.integers(100, size=4), new_rng.integers(100, size=4))
