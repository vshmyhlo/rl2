"""Uncalibrated hover-relative propulsion power and hover-equivalent seconds.

Without motor settings, power scales as thrust**1.5. With motor settings, use
torque proportional to thrust and actual rotor speed. In hover-normalized units,
motor torque is q + h * dx/dt, where q=T/T_hover, x=omega/omega_hover and
h=J*omega_hover/Q_hover (seconds). Winding loss is c*(q+h*dx/dt)**2, with
c=R*(Q_hover/k_t)**2/(Q_hover*omega_hover). These are lumped physical parameters,
not action-smoothing weights; their defaults are illustrative, not identified.

Aerodynamic power q*x and winding heat use end-of-substep thrust/speed and the
mean rotor acceleration; only the rotor kinetic-energy change is integrated
exactly. Clamp each motor's net draw at zero (active braking, no supply
regeneration), then average over motors and divide by 1+c so hover remains one.
The excess penalty integrates squared excess of the total power ratio, not
per-motor excess or excess of the interval mean. Energy is in hover-equivalent
seconds, not joules. Inflow, inductance, idle current, ESC losses and
current/voltage limits are omitted; this meter does not change flight dynamics.
"""

from typing import Annotated, Any, NamedTuple

import jax.numpy as jnp
import torch
from jax import Array as JaxArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MotorPowerConfig:
    inertia_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.01
    copper_loss_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.1


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PowerConfig:
    energy_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    excess_power_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    power_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.5
    motor: MotorPowerConfig | None = None


class PowerUsage[Array](NamedTuple):
    energy: Array  # (B,) integral of hover-relative power, in hover-equivalent seconds
    excess: Array  # (B,) integral of squared above-threshold power ratio, in seconds


def hover_rotor_speed[Array: (JaxArray, torch.Tensor)](
    hover_thrust: Array,  # () or (1,), per-motor hover thrust in newtons
    thrust_map: Array,  # (3,), quadratic coefficients in descending order
) -> Array:  # () or (1,)
    """Return ()/(1,) hover speed using VisFly's positive quadratic root."""
    a, b, c = thrust_map  # each ()
    return (-b + (b * b - 4 * a * (c - hover_thrust)) ** 0.5) / (2 * a)  # ()/(1,)


def thrust_power[Array: (JaxArray, torch.Tensor)](
    thrust_ratio: Array,  # (4, B), nonnegative T/T_hover
) -> Array:  # (4, B)
    """Return (4, B) hover-relative ideal power at fixed air density and rotor area."""
    if isinstance(thrust_ratio, torch.Tensor):
        return thrust_ratio.pow(1.5)  # (4, B), preserve the zero slope for native autograd
    return thrust_ratio * jnp.sqrt(thrust_ratio)  # (4, B), avoid a general power operation in XLA


def motor_power[Array: (JaxArray, torch.Tensor)](
    previous_speed: Array,  # (4, B), hover-normalized rotor speed
    speed: Array,  # (4, B), hover-normalized rotor speed
    thrust_ratio: Array,  # (4, B), nonnegative T/T_hover
    dt: float,
    config: MotorPowerConfig,
) -> Array:  # (4, B)
    """Return (4, B) signed hover-relative power before per-motor regeneration clipping."""
    acceleration = (speed - previous_speed) * (1 / dt)  # (4, B), scalar reciprocal can be hoisted by XLA
    torque = thrust_ratio + config.inertia_seconds * acceleration  # (4, B)
    kinetic_power = config.inertia_seconds * (speed + previous_speed) * acceleration / 2  # (4, B)
    return (thrust_ratio * speed + kinetic_power + config.copper_loss_ratio * torque**2) / (
        1 + config.copper_loss_ratio
    )  # (4, B)


def accumulate_power[Array: (JaxArray, torch.Tensor)](
    usage: PowerUsage[Array],  # leaves: (B,)
    rotor_power: Array,  # (4, B), signed hover-relative power
    dt: float,
    threshold: float,
) -> PowerUsage[Array]:
    """Return (B,) accumulated energy/excess; braking cannot offset other motors' draw."""
    power = rotor_power.clip(min=0).mean(axis=0)  # (B,)
    return PowerUsage(
        energy=usage.energy + power * dt,  # (B,)
        excess=usage.excess + (power - threshold).clip(min=0) ** 2 * dt,  # (B,)
    )


class NativePowerMeter:
    """Observe upstream motor substeps without changing the native integrator."""

    def __init__(self, dynamics: Any, config: PowerConfig) -> None:
        if config.motor is not None and not dynamics._ctrl_delay:
            raise ValueError("Motor power requires ctrl_delay=True to model finite rotor acceleration")
        self.dynamics = dynamics
        self.config = config
        self.power_usage: PowerUsage[torch.Tensor] | None = None
        self._step = dynamics.step
        self._run_motors = dynamics._run_motors
        self.hover_thrust = dynamics.m * 9.81 / 4  # ()/(1,)
        if config.motor is not None:
            self.hover_omega = hover_rotor_speed(self.hover_thrust, dynamics._thrust_map)  # ()/(1,)
        dynamics._run_motors = self.run_motors

    def run_motors(self, desired: torch.Tensor) -> torch.Tensor:  # (4, B) -> (4, B)
        motor = self.config.motor
        if motor is not None:
            previous_speed = self.dynamics._motor_omega / self.hover_omega  # (4, B), snapshot before update
        thrust = self._run_motors(desired)  # (4, B)
        thrust_ratio = thrust.clamp_min(0) / self.hover_thrust  # (4, B)
        if motor is None:
            power = thrust_power(thrust_ratio)  # (4, B)
        else:
            speed = self.dynamics._motor_omega / self.hover_omega  # (4, B)
            power = motor_power(previous_speed, speed, thrust_ratio, self.dynamics.dt, motor)  # (4, B)
        usage = self.power_usage
        if usage is not None:
            self.power_usage = accumulate_power(usage, power, self.dynamics.dt, self.config.power_threshold)
        return thrust  # (4, B)

    def step(self, action: torch.Tensor) -> torch.Tensor:  # (B, 4) -> (B, 13)
        # Upstream accepts host actions and moves them to the dynamics device.
        zero = torch.zeros_like(self.dynamics._motor_omega[0])  # (B,)
        self.power_usage = PowerUsage(energy=zero, excess=zero)
        return self._step(action)  # (B, 13)
