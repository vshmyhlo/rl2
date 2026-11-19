"""Shapes and analytic limits of the compiled CPU flight substeps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.envs.visfly.dynamics import FlightState, PhysicsParameters, integrate_euler, rotate


def parameters() -> PhysicsParameters[jax.Array]:
    return PhysicsParameters(
        allocation=jnp.eye(4),  # (4, 4)
        inertia=jnp.eye(3),  # (3, 3)
        inertia_inverse=jnp.eye(3),  # (3, 3)
        mass=jnp.array(1.0),  # ()
        motor_decay=jnp.array(0.5),  # ()
        thrust_map=jnp.array([1.0, 0.0, 0.0]),  # (3,)
        linear_drag=jnp.zeros((3, 1)),  # (3, 1)
        quadratic_drag=jnp.zeros((3, 1)),  # (3, 1)
    )


def initial_state(batch: int) -> FlightState[jax.Array]:
    return FlightState(
        position=jnp.zeros((3, batch)),  # (3, B)
        orientation=jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0])[:, None], (4, batch)),  # (4, B)
        velocity=jnp.zeros((3, batch)),  # (3, B)
        angular_velocity=jnp.zeros((3, batch)),  # (3, B)
        motor_omega=jnp.zeros((4, batch)),  # (4, B)
    )


@pytest.mark.parametrize("batch", [1, 3], ids=["evaluation", "vector"])
def test_euler_free_fall_and_wind(batch: int) -> None:
    state = initial_state(batch)
    result = jax.device_get(
        integrate_euler(
            state,
            parameters(),
            jnp.zeros((4, batch)),
            jnp.array([[2.0], [0.0], [0.0]]),
            0.01,
            steps=8,
            delay=False,
        )
    )
    assert result.state.orientation.shape == result.state.motor_omega.shape == result.thrust.shape == (4, batch)
    assert (
        result.state.position.shape == result.state.velocity.shape == result.state.angular_velocity.shape == (3, batch)
    )
    assert result.acceleration.shape == result.angular_acceleration.shape == (3, batch)
    np.testing.assert_allclose(result.state.position[0], 0.16, atol=1e-7)
    np.testing.assert_allclose(result.state.position[2], -9.81 * 0.01**2 * 28, atol=1e-7)
    np.testing.assert_allclose(result.state.velocity[2], -9.81 * 0.08, atol=1e-7)
    np.testing.assert_array_equal(result.state.orientation, jax.device_get(state.orientation))
    np.testing.assert_array_equal(result.state.angular_velocity, 0)


def test_euler_motor_delay_and_body_rotation() -> None:
    state = initial_state(1)
    result = jax.device_get(
        integrate_euler(
            state,
            parameters(),
            jnp.ones((4, 1)),
            jnp.zeros((3, 1)),
            0.01,
            steps=1,
            delay=True,
        )
    )
    np.testing.assert_allclose(result.state.motor_omega, 0.5)
    np.testing.assert_allclose(result.thrust, 0.25)
    np.testing.assert_allclose(result.angular_acceleration, 0.25)
    np.testing.assert_allclose(result.state.angular_velocity, 0.0025)
    orientation = jnp.array([[2**-0.5], [0], [2**-0.5], [0]])  # (4, 1), 90 degrees around Y
    rotated = jax.device_get(rotate(orientation, jnp.array([[0.0], [0.0], [1.0]])))  # (3, 1)
    np.testing.assert_allclose(rotated, [[1], [0], [0]], atol=1e-7)
