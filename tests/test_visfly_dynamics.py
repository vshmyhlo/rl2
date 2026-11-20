"""Shapes and analytic limits of the compiled CPU flight substeps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from rl2.envs.visfly.dynamics import FlightState, PhysicsParameters, integrate_euler, rotate
from rl2.envs.visfly.power import NativePowerMeter, PowerConfig


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
    assert result.power_usage is None
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


def test_power_hover_zero_high_and_unequal_rotor_loading() -> None:
    # Equal hover, motors off, twice hover thrust, and unequal thrust with hover total.
    thrust = jnp.array([[1, 0, 2, 4], [1, 0, 2, 0], [1, 0, 2, 0], [1, 0, 2, 0]]) * (9.81 / 4)  # (4, B)
    args = (initial_state(4), parameters(), thrust, jnp.zeros((3, 1)), 0.01)
    enabled = integrate_euler(*args, steps=2, delay=False, power_threshold=1.0)
    disabled = integrate_euler(*args, steps=2, delay=False)
    for actual, expected in zip(jax.tree.leaves(enabled.state), jax.tree.leaves(disabled.state), strict=True):
        np.testing.assert_array_equal(jax.device_get(actual), jax.device_get(expected))
    usage = jax.device_get(enabled.power_usage)
    assert all(leaf.shape == (4,) for leaf in usage)
    power = np.array([1, 0, 2**1.5, 2])  # (B,)
    np.testing.assert_allclose(usage.energy, power * 0.02, rtol=1e-6)
    np.testing.assert_allclose(usage.excess, np.maximum(power - 1, 0) ** 2 * 0.02, rtol=1e-6)


def test_power_integrates_motor_lag_and_control_duration() -> None:
    desired = jnp.full((4, 1), 9.81 / 4)  # (4, B)
    state = initial_state(1)
    result = integrate_euler(
        state, parameters(), desired, jnp.zeros((3, 1)), 0.01, steps=2, delay=True, power_threshold=0.2
    )
    usage = jax.device_get(result.power_usage)
    assert all(leaf.shape == (1,) for leaf in usage)
    # Rotor speed reaches 0.5 and 0.75 of target; thrust power scales with speed cubed.
    np.testing.assert_allclose(usage.energy, (0.5**3 + 0.75**3) * 0.01, rtol=1e-6)
    np.testing.assert_allclose(usage.excess, (0.75**3 - 0.2) ** 2 * 0.01, rtol=1e-6)
    first = integrate_euler(
        state, parameters(), desired, jnp.zeros((3, 1)), 0.01, steps=1, delay=True, power_threshold=0.2
    )
    second = integrate_euler(
        first.state, parameters(), desired, jnp.zeros((3, 1)), 0.01, steps=1, delay=True, power_threshold=0.2
    )
    np.testing.assert_allclose(
        usage.energy, jax.device_get(first.power_usage.energy + second.power_usage.energy), rtol=1e-6
    )
    # Integrating below-threshold and above-threshold samples differs from penalizing mean power.
    assert usage.excess[0] > max(usage.energy[0] / 0.02 - 0.2, 0) ** 2 * 0.02


class NativeDynamics:
    """Minimal upstream motor loop with lag and a nonzero thrust-map offset."""

    def __init__(self) -> None:
        self.m = torch.tensor(1.0)  # ()
        self.dt = 0.01
        self.omega = torch.zeros((4, 1))  # (4, B)

    def _run_motors(self, desired: torch.Tensor) -> torch.Tensor:  # (4, B) -> (4, B)
        self.omega = 0.5 * self.omega + 0.5 * torch.sqrt(desired - 0.1)  # (4, B)
        return self.omega.square() + 0.1  # (4, B)

    def step(self, action: torch.Tensor) -> torch.Tensor:  # (B, 4) -> (B, 13)
        for _ in range(2):
            self._run_motors(action.T)
        return torch.zeros((action.shape[0], 13))  # (B, 13)


def test_native_power_matches_compiled_substeps_and_clears_each_action() -> None:
    dynamics = NativeDynamics()
    meter = NativePowerMeter(dynamics, PowerConfig(power_threshold=0.2))
    desired = jnp.full((4, 1), 9.81 / 4)  # (4, B)
    state = initial_state(1)
    physics = parameters()._replace(thrust_map=jnp.array([1.0, 0.0, 0.1]))
    assert meter.power_usage is None
    for _ in range(2):
        actual = meter.step(torch.full((1, 4), 9.81 / 4))  # (B, 13)
        assert actual.shape == (1, 13)
        result = integrate_euler(
            state, physics, desired, jnp.zeros((3, 1)), dynamics.dt, steps=2, delay=True, power_threshold=0.2
        )
        for actual, expected in zip(meter.power_usage, jax.device_get(result.power_usage), strict=True):
            assert actual.shape == (1,)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)
        state = result.state
