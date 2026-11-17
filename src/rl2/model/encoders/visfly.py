"""Body-frame navigation features for VisFly's raw state and target observations."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.spatial.transform import Rotation

from rl2.envs.visfly_env import VisFlyObservation
from rl2.model.encoders.conv2d import ConvObservationEncoder, ConvStages
from rl2.model.encoders.mlp import MLPEncoder


def navigation_features(
    state: jax.Array,  # (*B, S): position, orientation, world velocity, body angular velocity
    target: jax.Array,  # (*B, 3): world target position
    *,
    goal_scale: float,
    velocity_scale: float,
    angular_velocity_scale: float,
) -> jax.Array:  # (*B, 15)
    """Return (*B, 15) float32 orientation, body velocity/rates, and relative goal.

    State is (*B, 13) with a wxyz quaternion or (*B, 12) with roll/pitch/yaw
    in radians. Output orientation is (6,): the body's x and y axes in world
    coordinates, from the first two rotation-matrix columns. This encoding is
    continuous and identical for q and -q. Linear velocity,
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
    orientation = rotation.as_matrix()[:, :, :2].swapaxes(-1, -2).reshape((-1, 6))  # (N, 6), body x/y axes
    velocity, goal = jax.tree.map(partial(rotation.apply, inverse=True), (flat[:, -6:-3], goal))  # each (N, 3)
    velocity = velocity / velocity_scale  # (N, 3)
    angular_velocity = flat[:, -3:] / angular_velocity_scale  # (N, 3), already body-frame
    goal = goal / goal_scale  # (N, 3)
    features = jnp.concatenate((orientation, velocity, angular_velocity, goal), axis=-1)  # (N, 15)
    return features.reshape((*state.shape[:-1], 15))  # (*B, 15)


class VisFlyEncoder(nn.Module):
    """Fuse RGB features with scaled body-frame navigation features."""

    stages: ConvStages
    embedding_size: int
    goal_scale: float
    velocity_scale: float
    angular_velocity_scale: float
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(self, obs: VisFlyObservation[jax.Array]) -> jax.Array:  # leaves: (B, *O) -> (B, E)
        image = ConvObservationEncoder(
            channels_last=True,
            stages=self.stages,
            embedding_size=self.embedding_size,
            dtype=self.dtype,
            name="image_0",
        )(obs["rgb"])  # (B, E)
        navigation = navigation_features(
            obs["state"],
            obs["target"],
            goal_scale=self.goal_scale,
            velocity_scale=self.velocity_scale,
            angular_velocity_scale=self.angular_velocity_scale,
        )  # (B, 15)
        combined = jnp.concatenate((image, navigation), axis=-1)  # (B, E + 15)
        return MLPEncoder(hidden_size=self.embedding_size, embedding_size=self.embedding_size, dtype=self.dtype)(
            combined
        )  # (B, E)
