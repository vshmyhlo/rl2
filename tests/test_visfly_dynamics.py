"""Shapes and analytic limits of the compiled CPU flight substeps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from rl2.envs.visfly.dynamics import CompiledEuler, FlightState, PhysicsParameters, integrate_euler, rotate
from rl2.envs.visfly.power import MotorPowerConfig, NativePowerMeter, PowerConfig


def test_rotation_matches_quaternion_matrix_and_broadcasts() -> None:
    # A nonunit quaternion also checks the q v q* scaling used by upstream.
    orientation = jnp.array([[2.0], [1.0], [-2.0], [3.0]])  # (4, 1)
    vectors = jnp.eye(3)  # (3, B)
    expected = np.array([[-8.0, -16.0, -2.0], [8.0, -2.0, -16.0], [14.0, -8.0, 8.0]])  # (3, B)
    rotated = jax.device_get(jax.jit(rotate)(orientation, vectors))  # (3, B)
    assert rotated.shape == (3, 3)
    np.testing.assert_allclose(rotated, expected)
    # Each column is an independent rotation, including the zero quaternion.
    orientations = orientation * jnp.array([0.0, 0.5, 1.0])  # (4, B)
    rotated = jax.device_get(jax.jit(rotate)(orientations, vectors[:, :1]))  # (3, B)
    assert rotated.shape == (3, 3)
    np.testing.assert_allclose(rotated, expected[:, :1] * np.array([0.0, 0.25, 1.0]))


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


def test_euler_gyroscopic_torque_with_coupled_inertia() -> None:
    angular_velocity = jnp.array([[1.0, -1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 0.0, 0.0]])  # (3, B)
    state = initial_state(3)._replace(angular_velocity=angular_velocity)
    physics = parameters()._replace(
        inertia=jnp.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),  # (3, 3)
        inertia_inverse=jnp.array([[0.6, -0.2, 0.0], [-0.2, 0.4, 0.0], [0.0, 0.0, 0.25]]),  # (3, 3)
    )
    result = jax.device_get(
        integrate_euler(state, physics, jnp.zeros((4, 3)), jnp.zeros((3, 1)), 0.01, steps=1, delay=False)
    )
    expected = np.array([[-1.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.25, 1.25, 0.0]])  # (3, B)
    assert result.angular_acceleration.shape == result.state.angular_velocity.shape == (3, 3)
    np.testing.assert_allclose(result.angular_acceleration, expected, atol=1e-7)
    np.testing.assert_allclose(result.state.angular_velocity, jax.device_get(angular_velocity) + 0.01 * expected)


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

    def __init__(self, *, batch: int = 1, delay: bool = True) -> None:
        self.m = torch.tensor([1.0])  # (1,)
        self.dt = 0.01
        self._ctrl_delay = delay
        self._thrust_map = torch.tensor([1.0, 0.0, 0.1])  # (3,)
        self._motor_omega = torch.zeros((4, batch))  # (4, B)

    def _run_motors(self, desired: torch.Tensor) -> torch.Tensor:  # (4, B) -> (4, B)
        if not self._ctrl_delay:
            return desired  # (4, B)
        # An in-place update also checks that the meter snapshots the previous speed.
        self._motor_omega.mul_(0.5).add_(0.5 * torch.sqrt(desired - 0.1))
        return self._motor_omega.square() + 0.1  # (4, B)

    def step(self, action: torch.Tensor) -> torch.Tensor:  # (B, 4) -> (B, 13)
        desired = action.T.to(self._motor_omega.device)  # (4, B), upstream accepts host actions
        for _ in range(2):
            self._run_motors(desired)
        return self._motor_omega.new_zeros((action.shape[0], 13))  # (B, 13)


@pytest.mark.parametrize("device", ["cpu", "meta"], ids=["action_dtype", "host_action_device"])
def test_native_power_accumulates_on_dynamics_device_and_dtype(device: str) -> None:
    # Meta tensors exercise device placement without requiring a GPU.
    with torch.device(device):
        dynamics = NativeDynamics(batch=3)
    meter = NativePowerMeter(dynamics, PowerConfig(motor=MotorPowerConfig()))
    meter.step(torch.ones((3, 4), dtype=torch.float64))
    assert meter.power_usage is not None
    for value in meter.power_usage:
        assert value.shape == (3,)
        assert value.device == dynamics._motor_omega.device
        assert value.dtype == dynamics._motor_omega.dtype


@pytest.mark.parametrize(
    ("batch", "delay", "motor"),
    [(1, True, None), (3, True, MotorPowerConfig()), (3, False, None)],
    ids=["single_thrust", "batched_motor", "batched_no_delay"],
)
def test_native_power_matches_compiled_substeps_and_clears_each_action(
    batch: int, delay: bool, motor: MotorPowerConfig | None
) -> None:
    dynamics = NativeDynamics(batch=batch, delay=delay)
    meter = NativePowerMeter(dynamics, PowerConfig(power_threshold=0.2, motor=motor))
    state = initial_state(batch)
    physics = parameters()._replace(mass=jnp.ones(1), thrust_map=jnp.array([1.0, 0.0, 0.1]))
    assert meter.power_usage is None
    for thrust in (9.81 / 4, 0.1, 9.81 / 4):
        desired = jnp.broadcast_to(0.1 + (thrust - 0.1) * (1 + jnp.arange(batch)), (4, batch))  # (4, B)
        actual = meter.step(torch.tensor(jax.device_get(desired.T)))  # (B, 13)
        assert actual.shape == (batch, 13)
        result = integrate_euler(
            state,
            physics,
            desired,
            jnp.zeros((3, 1)),
            dynamics.dt,
            steps=2,
            delay=delay,
            power_threshold=0.2,
            power_motor=motor,
        )
        for actual, expected in zip(meter.power_usage, jax.device_get(result.power_usage), strict=True):
            assert actual.shape == (batch,)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)
        state = result.state
        # Reset only one environment; the others must retain their rotor history.
        if thrust == 0.1:
            dynamics._motor_omega[:, 0] = 0
            state = state._replace(motor_omega=state.motor_omega.at[:, 0].set(0))


def test_motor_power_hover_transients_and_per_rotor_braking() -> None:
    # Three environments: steady hover, mixed acceleration/braking, and motors off.
    previous = jnp.array([[1, 0, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0]], dtype=jnp.float32)  # (4, B)
    desired_speed = jnp.array([[1, 2, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=jnp.float32)  # (4, B)
    state = initial_state(3)._replace(motor_omega=previous * (9.81 / 4) ** 0.5)
    desired = desired_speed**2 * (9.81 / 4)  # (4, B)
    motor = MotorPowerConfig(inertia_seconds=0.01, copper_loss_ratio=0.1)
    args = (state, parameters(), desired, jnp.zeros((3, 1)), 0.01)
    enabled = integrate_euler(*args, steps=1, delay=True, power_threshold=0.5, power_motor=motor)
    disabled = integrate_euler(*args, steps=1, delay=True)
    for actual, expected in zip(jax.tree.leaves(enabled.state), jax.tree.leaves(disabled.state), strict=True):
        np.testing.assert_array_equal(jax.device_get(actual), jax.device_get(expected))
    usage = jax.device_get(enabled.power_usage)
    assert usage.energy.shape == usage.excess.shape == (3,)
    # Accelerating motor: aero=1, stored=0.5, copper=0.4, hover normalization=1.1.
    # Braking motor: aero=1, stored=-1.5, copper=0; no regeneration credit.
    expected_power = np.array([1, (1.9 / 1.1 + 0 + 1 + 0) / 4, 0])  # (B,)
    np.testing.assert_allclose(usage.energy, expected_power * 0.01, rtol=1e-6)
    np.testing.assert_allclose(usage.excess, np.maximum(expected_power - 0.5, 0) ** 2 * 0.01, rtol=1e-6)


def test_motor_power_requires_finite_motor_response() -> None:
    dynamics = NativeDynamics()
    dynamics._ctrl_delay = False
    config = PowerConfig(motor=MotorPowerConfig())
    for meter in (NativePowerMeter, CompiledEuler):
        with pytest.raises(ValueError, match="ctrl_delay=True"):
            meter(dynamics, config)
    with pytest.raises(ValueError, match="ctrl_delay=True"):
        integrate_euler(
            initial_state(1),
            parameters(),
            jnp.ones((4, 1)),
            jnp.zeros((3, 1)),
            0.01,
            steps=1,
            delay=False,
            power_threshold=1.5,
            power_motor=config.motor,
        )
