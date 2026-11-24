"""Uncalibrated rotor effort: hover-relative power and hover-equivalent seconds."""

from typing import Annotated, Any, NamedTuple

import torch
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PowerConfig:
    energy_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    excess_power_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    power_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.5


class PowerUsage[Array](NamedTuple):
    energy: Array  # (B,) integral of hover-relative power, in hover-equivalent seconds
    excess: Array  # (B,) integral of squared above-threshold power ratio, in seconds


class NativePowerMeter:
    """Observe upstream motor substeps without changing the native integrator."""

    def __init__(self, dynamics: Any, config: PowerConfig) -> None:
        self.dynamics = dynamics
        self.config = config
        self.power_usage: PowerUsage[torch.Tensor] | None = None
        self._step = dynamics.step
        self._run_motors = dynamics._run_motors
        dynamics._run_motors = self.run_motors

    def run_motors(self, desired: torch.Tensor) -> torch.Tensor:  # (4, B) -> (4, B)
        thrust = self._run_motors(desired)  # (4, B)
        ratio = (thrust.clamp_min(0) / (self.dynamics.m * 9.81 / 4)).pow(1.5).mean(dim=0)  # (B,)
        usage = self.power_usage
        if usage is not None:
            self.power_usage = PowerUsage(
                energy=usage.energy + ratio * self.dynamics.dt,  # (B,)
                excess=usage.excess
                + (ratio - self.config.power_threshold).clamp_min(0).square() * self.dynamics.dt,  # (B,)
            )
        return thrust  # (4, B)

    def step(self, action: torch.Tensor) -> torch.Tensor:  # (B, 4) -> (B, 13)
        zero = torch.zeros_like(action[:, 0])  # (B,)
        self.power_usage = PowerUsage(energy=zero, excess=zero)
        return self._step(action)  # (B, 13)
