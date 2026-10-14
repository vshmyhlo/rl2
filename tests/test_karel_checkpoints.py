from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from rl2.karel_checkpoints import TrainingProgress, restore_checkpoint, save_checkpoint


def test_orbax_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = TrainState.create(apply_fn=jnp.add, params={"weight": jnp.ones(2)}, tx=optax.adam(0.01))
    state = state.apply_gradients(grads={"weight": jnp.ones(2)})
    progress = TrainingProgress(state=state, key=jax.random.key(2), iteration=1, steps=4, logged_groups=1, episodes=4)
    rng = np.random.default_rng(3)
    assert restore_checkpoint(str(tmp_path), state, rng) is None
    save_checkpoint(str(tmp_path), progress, rng)
    assert (tmp_path / "1" / "_CHECKPOINT_METADATA").is_file()
    assert not (tmp_path / "checkpoint.msgpack").exists()
    restored_rng = np.random.default_rng(0)
    restored = restore_checkpoint(str(tmp_path), state, restored_rng)
    chex.assert_trees_all_equal(restored, progress)
    np.testing.assert_array_equal(restored_rng.integers(100, size=4), rng.integers(100, size=4))
