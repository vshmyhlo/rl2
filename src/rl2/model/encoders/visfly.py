"""Body-frame navigation features for VisFly's raw state and target observations."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def navigation_features(
    state: jax.Array,  # (*B, S): position, orientation, world velocity, body angular velocity
    target: jax.Array,  # (*B, 3): world target position
    *,
    goal_scale: float,
    velocity_scale: float,
    angular_velocity_scale: float,
) -> jax.Array:  # (*B, 13)
    """Return (*B, 13) float32 orientation, body velocity/rates, and relative goal.

    State is (*B, 13) with a wxyz quaternion or (*B, 12) with roll/pitch/yaw
    in radians. Output orientation is a unit wxyz quaternion. Linear velocity,
    angular velocity, and goal displacement are divided by their scales in
    m/s, rad/s, and metres, respectively, without clipping. Absolute position
    is omitted. *B can include batch and sequence dimensions or be empty.
    """
    if state.shape[-1] not in (12, 13):
        raise ValueError("VisFly state must have 12 Euler or 13 quaternion features")
    chex.assert_shape(target, (*state.shape[:-1], 3))
    flat = state.astype(jnp.float32).reshape((-1, state.shape[-1]))  # (N, S), N = prod(B)
    goal = target.astype(jnp.float32).reshape((-1, 3)) - flat[:, :3]  # (N, 3)
    if state.shape[-1] == 13:
        rotation = Rotation.from_quat(jnp.roll(flat[:, 3:7], -1, axis=-1))  # quaternion: (N, 4), xyzw
    else:
        rotation = Rotation.from_euler("xyz", flat[:, 3:6])  # quaternion: (N, 4)
    orientation = jnp.roll(rotation.as_quat(), 1, axis=-1)  # (N, 4), wxyz
    velocity, goal = jax.tree.map(partial(rotation.apply, inverse=True), (flat[:, -6:-3], goal))  # each (N, 3)
    velocity = velocity / velocity_scale  # (N, 3)
    angular_velocity = flat[:, -3:] / angular_velocity_scale  # (N, 3), already body-frame
    goal = goal / goal_scale  # (N, 3)
    features = jnp.concatenate((orientation, velocity, angular_velocity, goal), axis=-1)  # (N, 13)
    return features.reshape((*state.shape[:-1], 13))  # (*B, 13)
