"""VisFly navigation coordinates, scaling, and JIT shape contracts."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rl2.model.encoders.conv2d import ConvStage
from rl2.model.encoders.visfly import VisFlyEncoder, navigation_features


@pytest.mark.parametrize(
    ("batch_shape", "euler"),
    [
        pytest.param((), False, id="terminal_quaternion"),
        pytest.param((1,), False, id="singleton_batch"),
        pytest.param((2, 2), True, id="sequence_euler"),
    ],
)
def test_navigation_features_shapes_and_orientation(batch_shape: tuple[int, ...], euler: bool) -> None:
    angles = np.array([0.3, -0.4, 0.7], dtype=np.float32)  # (3,), roll/pitch/yaw
    rotation = Rotation.from_euler("xyz", angles)
    quaternion = np.roll(rotation.as_quat(), 1).astype(np.float32)  # (4,), wxyz
    orientation = angles if euler else quaternion  # (3,) or (4,)
    state = np.broadcast_to(
        np.concatenate(([10, 20, 30], orientation, [4, -2, 1], [1, -2, 3])).astype(np.float32),
        (*batch_shape, 12 if euler else 13),
    )  # (*B, S)
    target = np.broadcast_to(np.array([12, 24, 36], dtype=np.float32), (*batch_shape, 3))  # (*B, 3)
    features = jax.jit(partial(navigation_features, goal_scale=2.0, velocity_scale=4.0, angular_velocity_scale=8.0))(
        *jax.device_put((state, target))
    )  # (*B, 15)
    assert features.shape == (*batch_shape, 15)
    assert features.dtype == jnp.float32
    expected = np.concatenate(
        (
            rotation.as_matrix()[:, :2].T.reshape(6),
            rotation.inv().apply([4, -2, 1]) / 4,
            np.array([1, -2, 3]) / 8,
            rotation.inv().apply([2, 4, 6]) / 2,
        )
    )  # (15,)
    np.testing.assert_allclose(jax.device_get(features), np.broadcast_to(expected, features.shape), atol=2e-6)


def test_navigation_features_body_axes_and_translation_invariance() -> None:
    half = np.sqrt(0.5)
    quaternions = np.array([[1, 0, 0, 0], [half, 0, 0, half], [half, half, 0, 0]], dtype=np.float32)  # (B, 4)
    state = np.concatenate((np.zeros((3, 3)), quaternions, np.tile([0, 5, 0, 0, 0, 10], (3, 1))), axis=-1).astype(
        np.float32
    )  # (B, 13)
    target = np.tile(np.array([0, 20, 0], dtype=np.float32), (3, 1))  # (B, 3)
    preprocess = jax.jit(partial(navigation_features, goal_scale=10.0, velocity_scale=5.0, angular_velocity_scale=10.0))
    features = jax.device_get(preprocess(*jax.device_put((state, target))))  # (B, 15)
    np.testing.assert_allclose(features[:, 6:9], [[0, 1, 0], [1, 0, 0], [0, 0, -1]], atol=1e-6)
    np.testing.assert_allclose(features[:, 9:12], np.tile([0, 0, 1], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(features[:, 12:], [[0, 2, 0], [2, 0, 0], [0, 0, -2]], atol=1e-6)  # No clipping.
    offset = np.array([100, -200, 300], dtype=np.float32)  # (3,)
    state[:, :3] += offset
    target += offset
    translated = jax.device_get(preprocess(*jax.device_put((state, target))))  # (B, 15)
    np.testing.assert_allclose(translated, features, atol=1e-6)
    state[:, 3:7] *= -1
    sign_flipped = jax.device_get(preprocess(*jax.device_put((state, target))))  # (B, 15)
    np.testing.assert_allclose(sign_flipped, features, atol=1e-6)
    state[:, 3:7] *= 3
    scaled = jax.device_get(preprocess(*jax.device_put((state, target))))  # (B, 15)
    np.testing.assert_allclose(scaled, features, atol=1e-6)
    at_goal = jax.device_get(preprocess(*jax.device_put((state, state[:, :3]))))  # (B, 15)
    np.testing.assert_array_equal(at_goal[:, 12:], 0)


def test_orientation_encoding_is_continuous_across_half_turn() -> None:
    epsilon = 1e-4
    w, z = np.sin(epsilon / 2), np.cos(epsilon / 2)
    state = np.zeros((2, 13), dtype=np.float32)  # (B, S)
    # Positive-w quaternions on either side of the yaw wrap are nearly antipodal.
    state[:, 3:7] = [[w, 0, 0, z], [w, 0, 0, -z]]
    target = np.zeros((2, 3), dtype=np.float32)  # (B, 3)
    features = jax.device_get(
        jax.jit(partial(navigation_features, goal_scale=10.0, velocity_scale=5.0, angular_velocity_scale=10.0))(
            *jax.device_put((state, target))
        )
    )  # (B, 15)
    np.testing.assert_allclose(features[0], features[1], atol=3 * epsilon, rtol=0)
    np.testing.assert_allclose(features[:, :6], np.tile([-1, 0, 0, 0, -1, 0], (2, 1)), atol=2 * epsilon, rtol=0)


@pytest.mark.parametrize(
    ("state_shape", "target_shape", "error"),
    [
        pytest.param((2, 11), (2, 3), ValueError, id="unsupported_state_width"),
        pytest.param((2, 13), (1, 3), AssertionError, id="mismatched_batch"),
    ],
)
def test_navigation_features_reject_invalid_shapes(
    state_shape: tuple[int, ...], target_shape: tuple[int, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        jax.eval_shape(
            partial(navigation_features, goal_scale=10.0, velocity_scale=5.0, angular_velocity_scale=10.0),
            jax.ShapeDtypeStruct(state_shape, jnp.float32),
            jax.ShapeDtypeStruct(target_shape, jnp.float32),
        )


@pytest.mark.parametrize(
    "depth_shape",
    [
        pytest.param((1, 2, 4, 1), id="rectangular_singleton"),
        pytest.param((1, 4, 2, 1), id="mismatched_resolution"),
        pytest.param((2, 2, 4, 1), id="mismatched_batch"),
        pytest.param((1, 2, 4, 3), id="multiple_depth_channels"),
        pytest.param((1, 2, 4), id="missing_channel"),
    ],
)
def test_rgbd_encoder_shape_contract(depth_shape: tuple[int, ...]) -> None:
    encoder = VisFlyEncoder(
        stages=(ConvStage(channels=2, blocks=0),),
        embedding_size=4,
        goal_scale=10.0,
        velocity_scale=5.0,
        angular_velocity_scale=10.0,
    )
    observation = {
        "state": jax.ShapeDtypeStruct((1, 13), jnp.float32),  # (B, S)
        "target": jax.ShapeDtypeStruct((1, 3), jnp.float32),  # (B, 3)
        "rgb": jax.ShapeDtypeStruct((1, 2, 4, 3), jnp.uint8),  # (B, H, W, 3)
        "depth": jax.ShapeDtypeStruct(depth_shape, jnp.float32),  # (B, H, W, 1) when valid
    }
    if depth_shape == (1, 2, 4, 1):
        output, _ = jax.eval_shape(encoder.init_with_output, jax.random.key(0), observation)
        assert output.shape == (1, 4)
        assert output.dtype == jnp.float32
    else:
        with pytest.raises(AssertionError):
            jax.eval_shape(encoder.init_with_output, jax.random.key(0), observation)
