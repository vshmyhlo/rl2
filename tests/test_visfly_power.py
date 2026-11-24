"""Analytic limits and backend agreement of hover-relative power accounting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from rl2.envs.visfly.power import (
    MotorPowerConfig,
    PowerUsage,
    accumulate_power,
    hover_rotor_speed,
    motor_power,
    thrust_power,
)


def test_hover_reference_inverts_all_thrust_map_coefficients() -> None:
    coefficients = jnp.array([2.0, 3.0, 0.1])  # (3,)
    speed = hover_rotor_speed(jnp.array(14.1), coefficients)  # ()
    assert speed.shape == ()
    np.testing.assert_allclose(jax.device_get(speed), 2.0)
    native = hover_rotor_speed(torch.tensor([14.1]), torch.tensor([2.0, 3.0, 0.1]))  # (1,)
    assert native.shape == (1,)
    torch.testing.assert_close(native, torch.tensor([2.0]))


def test_thrust_power_preserves_zero_slope_with_autograd() -> None:
    thrust = torch.tensor([[0.0, 1.0, 4.0]] * 4, requires_grad=True)  # (4, B)
    power = thrust_power(thrust)  # (4, B)
    power.sum().backward()
    torch.testing.assert_close(power, torch.tensor([[0.0, 1.0, 8.0]] * 4))
    torch.testing.assert_close(thrust.grad, torch.tensor([[0.0, 1.5, 3.0]] * 4))
    compiled = jax.jit(thrust_power)(jax.device_put(thrust.detach().numpy()))  # (4, B)
    assert compiled.shape == (4, 3)
    np.testing.assert_allclose(jax.device_get(compiled), power.detach().numpy())

    def total_power(ratio: jax.Array) -> jax.Array:  # (4, B) -> ()
        return thrust_power(ratio).sum()  # ()

    gradient = jax.jit(jax.grad(total_power))(jax.device_put(thrust.detach().numpy()))  # (4, B)
    assert gradient.shape == (4, 3)
    np.testing.assert_allclose(jax.device_get(gradient), thrust.grad.numpy())


def test_accumulation_clips_each_motor_before_averaging_and_thresholding() -> None:
    power = jnp.array([[4.0, -4.0, 0.0], [-4.0, -4.0, 0.0], [0.0, -4.0, 0.0], [0.0, -4.0, 0.0]])  # (4, B)
    usage = PowerUsage(energy=jnp.ones(3), excess=jnp.ones(3))  # leaves: (B,)
    result = jax.jit(accumulate_power)(usage, power, 0.02, 0.5)
    native = accumulate_power(
        PowerUsage(energy=torch.ones(3), excess=torch.ones(3)),
        torch.tensor(jax.device_get(power)),
        0.02,
        0.5,
    )
    for actual, expected in zip(jax.device_get(result), native, strict=True):
        assert actual.shape == expected.shape == (3,)
        np.testing.assert_allclose(actual, expected.numpy())
    np.testing.assert_allclose(jax.device_get(result.energy), [1.02, 1.0, 1.0])
    np.testing.assert_allclose(jax.device_get(result.excess), [1.005, 1.0, 1.0])
    np.testing.assert_array_equal(jax.device_get(usage.energy), 1.0)
    np.testing.assert_array_equal(jax.device_get(usage.excess), 1.0)


def test_motor_kinetic_energy_telescopes_and_heating_depends_on_duration() -> None:
    previous = jnp.zeros((4, 1))  # (4, B)
    halfway = jnp.full((4, 1), 0.5)  # (4, B)
    speed = jnp.ones((4, 1))  # (4, B)
    no_load = jnp.zeros((4, 1))  # (4, B)
    lossless = MotorPowerConfig(inertia_seconds=0.01, copper_loss_ratio=0)
    up = motor_power(previous, speed, no_load, 0.02, lossless) * 0.02  # (4, B)
    split = (
        motor_power(previous, halfway, no_load, 0.01, lossless) + motor_power(halfway, speed, no_load, 0.01, lossless)
    ) * 0.01  # (4, B)
    down = motor_power(speed, previous, no_load, 0.02, lossless) * 0.02  # (4, B)
    np.testing.assert_allclose(jax.device_get(up), 0.005)
    np.testing.assert_allclose(jax.device_get(split), jax.device_get(up))
    np.testing.assert_allclose(jax.device_get(up + down), 0, atol=1e-8)
    lossy = MotorPowerConfig(inertia_seconds=0.01, copper_loss_ratio=0.1)
    fast = motor_power(previous, speed, no_load, 0.01, lossy) * 0.01  # (4, B)
    slow = motor_power(previous, speed, no_load, 0.02, lossy) * 0.02  # (4, B)
    # Same stored energy; halving duration doubles integrated winding heat.
    np.testing.assert_allclose(jax.device_get(fast), (0.005 + 0.001) / 1.1)
    np.testing.assert_allclose(jax.device_get(slow), (0.005 + 0.0005) / 1.1)


def test_motor_power_charges_repeated_speed_changes_at_equal_aerodynamic_work() -> None:
    low = jnp.full((4, 1), 0.9)  # (4, B)
    high = jnp.full((4, 1), 1.1)  # (4, B)
    motor = MotorPowerConfig()
    # Both traces spend equal substeps at each speed/thrust. Only transitions differ.
    alternating = motor_power(low, high, high**2, 0.01, motor) + motor_power(high, low, low**2, 0.01, motor)  # (4, B)
    steady = motor_power(high, high, high**2, 0.01, motor) + motor_power(low, low, low**2, 0.01, motor)  # (4, B)
    # Closed-cycle inertial work cancels. For these samples, extra copper loss is 0.024/1.1.
    np.testing.assert_allclose(jax.device_get(alternating - steady), 0.024 / 1.1, rtol=2e-5)


@pytest.mark.parametrize(
    "values",
    [{"inertia_seconds": 0}, {"copper_loss_ratio": -0.1}, {"inertia_seconds": float("inf")}],
)
def test_motor_power_config_rejects_invalid_parameters(values: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MotorPowerConfig(**values)
