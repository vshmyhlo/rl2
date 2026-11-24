"""Compiled CPU Euler substeps; VisFly retains control, wind, delays, and resets."""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from numpy.typing import NDArray

from rl2.envs.visfly.power import (
    MotorPowerConfig,
    PowerConfig,
    PowerUsage,
    accumulate_power,
    hover_rotor_speed,
    motor_power,
    thrust_power,
)


class FlightState[Array](NamedTuple):
    position: Array  # (3, B)
    orientation: Array  # (4, B), wxyz
    velocity: Array  # (3, B)
    angular_velocity: Array  # (3, B)
    motor_omega: Array  # (4, B)


class PhysicsParameters[Array](NamedTuple):
    allocation: Array  # (4, 4)
    inertia: Array  # (3, 3)
    inertia_inverse: Array  # (3, 3)
    mass: Array  # () or (1,)
    motor_decay: Array  # () or (1,)
    thrust_map: Array  # (3,)
    linear_drag: Array  # (3, 1) or (3, B)
    quadratic_drag: Array  # (3, 1) or (3, B)


class PhysicsResult[Array](NamedTuple):
    state: FlightState[Array]  # leaves: (3, B) or (4, B)
    thrust: Array  # (4, B)
    acceleration: Array  # (3, B)
    angular_acceleration: Array  # (3, B)
    power_usage: PowerUsage[Array] | None = None  # leaves: (B,), absent when disabled


def quaternion_product(left: jax.Array, right: jax.Array) -> jax.Array:  # (4, B), (4, B) -> (4, B)
    w, x, y, z = left  # each (B,)
    a, b, c, d = right  # each (B,)
    return jnp.stack(
        (
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        )
    )  # (4, B)


def rotate(orientation: jax.Array, vector: jax.Array) -> jax.Array:  # (4, B), (3, B) -> (3, B)
    """Return (3, B) vectors transformed by q v q*, including nonunit input quaternions."""
    w, x, y, z = orientation  # each (B,)
    a, b, c = vector  # each (B,)
    return jnp.stack(
        (
            (w * w + x * x - y * y - z * z) * a + 2 * (x * y - w * z) * b + 2 * (x * z + w * y) * c,
            2 * (x * y + w * z) * a + (w * w - x * x + y * y - z * z) * b + 2 * (y * z - w * x) * c,
            2 * (x * z - w * y) * a + 2 * (y * z + w * x) * b + (w * w - x * x - y * y + z * z) * c,
        )
    )  # (3, B)


@partial(jax.jit, static_argnames=("steps", "delay", "power_motor"))
def integrate_euler(
    state: FlightState[jax.Array],  # leaves: (3, B) or (4, B)
    parameters: PhysicsParameters[jax.Array],
    desired_thrust: jax.Array,  # (4, B)
    wind: jax.Array,  # (3, 1) or (3, B)
    dt: float,
    *,
    steps: int,
    delay: bool,
    power_threshold: float | None = None,
    power_motor: MotorPowerConfig | None = None,
) -> PhysicsResult[jax.Array]:
    """Advance the unchanged motor/Euler equations through all control-interval substeps.

    state: (3, B)/(4, B) state leaves in VisFly's component-major layout.
    desired_thrust: (4, B) clamped controller output held over the interval.
    wind: (3, 1) or (3, B) additive position velocity, matching upstream.
    Returns: (3, B)/(4, B) state and final-substep thrust/accelerations, optional
        (B,) integrated energy/excess over the control interval. Motor power requires
        finite motor lag; rotor history comes from state, including after resets.
    """
    if power_motor is not None and not delay:
        raise ValueError("Motor power requires ctrl_delay=True to model finite rotor acceleration")
    if power_threshold is not None:
        hover_thrust = parameters.mass * 9.81 / 4  # ()/(1,)
        inverse_hover_thrust = 1 / hover_thrust  # ()/(1,), keep scalar division outside the substeps
        if power_motor is not None:
            hover_omega = hover_rotor_speed(hover_thrust, parameters.thrust_map)  # ()/(1,)
            inverse_hover_omega = 1 / hover_omega  # ()/(1,), hoist normalization out of the substep loop
    if delay:
        a, b, c = parameters.thrust_map  # each ()
        desired_omega = (-b + jnp.sqrt(b * b - 4 * a * (c - desired_thrust))) / (2 * a)  # (4, B)

    def substep(index: int, result: PhysicsResult[jax.Array]) -> PhysicsResult[jax.Array]:
        position, orientation, velocity, angular_velocity, omega = result.state  # (3/4, B)
        if delay:
            omega = parameters.motor_decay * omega + (1 - parameters.motor_decay) * desired_omega  # (4, B)
            thrust = a * omega**2 + b * omega + c  # (4, B)
        else:
            thrust = desired_thrust  # (4, B)
        usage = result.power_usage
        if power_threshold is not None and usage is not None:
            ratio = jnp.maximum(thrust, 0) * inverse_hover_thrust  # (4, B)
            if power_motor is None:
                power = thrust_power(ratio)  # (4, B)
            else:
                power = motor_power(
                    result.state.motor_omega * inverse_hover_omega,
                    omega * inverse_hover_omega,
                    ratio,
                    dt,
                    power_motor,
                )  # (4, B)
            usage = accumulate_power(usage, power, dt, power_threshold)
        force_torque = parameters.allocation @ thrust  # (4, B)
        conjugate = orientation * jnp.array([1, -1, -1, -1], dtype=orientation.dtype)[:, None]  # (4, B)
        body_velocity = rotate(conjugate, velocity)  # (3, B)
        drag = parameters.linear_drag * body_velocity + parameters.quadratic_drag * body_velocity * jnp.abs(
            body_velocity
        )  # (3, B)
        force = jnp.array([0, 0, 1], dtype=velocity.dtype)[:, None] * force_torque[:1] - drag  # (3, B)
        acceleration = (
            rotate(orientation, force) / parameters.mass + jnp.array([0, 0, -9.81], dtype=velocity.dtype)[:, None]
        )  # (3, B)
        angular_momentum = parameters.inertia @ angular_velocity  # (3, B)
        x, y, z = angular_velocity  # each (B,)
        lx, ly, lz = angular_momentum  # each (B,)
        # Keep the component-major layout; jnp.cross introduces a transposed copy on CPU.
        gyroscopic_torque = jnp.stack((y * lz - z * ly, z * lx - x * lz, x * ly - y * lx))  # (3, B)
        angular_acceleration = parameters.inertia_inverse @ (force_torque[1:] - gyroscopic_torque)  # (3, B)
        angular_quaternion = jnp.concatenate((jnp.zeros_like(angular_velocity[:1]), angular_velocity))  # (4, B)
        orientation = orientation + quaternion_product(orientation, angular_quaternion) * 0.5 * dt  # (4, B)
        orientation = orientation / jnp.sqrt(jnp.sum(orientation**2, axis=0))  # (4, B)
        return PhysicsResult(
            state=FlightState(
                position=position + (velocity + wind) * dt,
                orientation=orientation,
                velocity=velocity + acceleration * dt,
                angular_velocity=angular_velocity + angular_acceleration * dt,
                motor_omega=omega,
            ),
            thrust=thrust,
            acceleration=acceleration,
            angular_acceleration=angular_acceleration,
            power_usage=usage,
        )

    initial = PhysicsResult(
        state=state,
        thrust=desired_thrust,
        acceleration=jnp.zeros_like(state.velocity),
        angular_acceleration=jnp.zeros_like(state.angular_velocity),
        power_usage=(
            PowerUsage(
                energy=jnp.zeros_like(state.position[0]),  # (B,)
                excess=jnp.zeros_like(state.position[0]),  # (B,)
            )
            if power_threshold is not None
            else None
        ),
    )
    return jax.lax.fori_loop(0, steps, substep, initial)


def host_array(value: torch.Tensor) -> NDArray[np.float32]:  # (*S,) -> (*S,)
    return value.detach().numpy()  # (*S,)


def writable_tensor(value: NDArray[np.float32]) -> torch.Tensor:  # (*S,) -> (*S,)
    # Upstream partial resets mutate these arrays; JAX host buffers are read-only.
    return torch.from_numpy(value.copy())  # (*S,)


class CompiledEuler:
    """Bridge CPU Torch state to one fused control-interval executable."""

    def __init__(self, dynamics: Any, power: PowerConfig | None = None) -> None:
        if power is not None and power.motor is not None and not dynamics._ctrl_delay:
            raise ValueError("Motor power requires ctrl_delay=True to model finite rotor acceleration")
        self.dynamics = dynamics
        self.power = power
        self.power_usage: PowerUsage[torch.Tensor] | None = None
        self.device = jax.devices("cpu")[0]
        self.mesh = Mesh(devices=[self.device], axis_names=("physics",))
        self.parameters = jax.device_put(
            jax.tree.map(
                host_array,
                PhysicsParameters(
                    allocation=dynamics._B_allocation,
                    inertia=dynamics._inertia,
                    inertia_inverse=dynamics._inertia_inv,
                    mass=dynamics.m,
                    motor_decay=dynamics._c,
                    thrust_map=dynamics._thrust_map,
                    linear_drag=dynamics._linear_drag_coeffs,
                    quadratic_drag=dynamics._quad_drag_coeffs,
                ),
            ),
            self.device,
        )

    def step(self, action: torch.Tensor) -> torch.Tensor:  # (B, 4) -> (B, 13)
        dynamics = self.dynamics
        dynamics.update_wind()
        if dynamics._comm_delay_steps:
            dynamics._pre_action.append(action.T.clone())
            action = dynamics._pre_action.pop(0).T  # (B, 4)
        desired = dynamics._get_thrust_from_cmd(dynamics._de_normalize(action))  # (4, B)
        state = FlightState(
            position=dynamics._position,
            orientation=dynamics._orientation.toTensor(),
            velocity=dynamics._velocity,
            angular_velocity=dynamics._angular_velocity,
            motor_omega=dynamics._motor_omega,
        )
        state, desired, wind, linear_drag, quadratic_drag = jax.device_put(
            jax.tree.map(
                host_array,
                (
                    state,
                    desired,
                    dynamics.wind_velocity,
                    dynamics._linear_drag_coeffs,
                    dynamics._quad_drag_coeffs,
                ),
            ),
            self.device,
        )
        parameters = self.parameters._replace(linear_drag=linear_drag, quadratic_drag=quadratic_drag)
        # Training's policy mesh may use GPUs; the simulator's host state stays on CPU.
        with jax.set_mesh(self.mesh):
            result = jax.tree.map(
                writable_tensor,
                jax.device_get(
                    integrate_euler(
                        state,
                        parameters,
                        desired,
                        wind,
                        dynamics.dt,
                        steps=dynamics._interval_steps,
                        delay=dynamics._ctrl_delay,
                        power_threshold=self.power.power_threshold if self.power is not None else None,
                        power_motor=self.power.motor if self.power is not None else None,
                    )
                ),
            )
        dynamics._position = result.state.position  # (3, B)
        self.power_usage = result.power_usage  # leaves: (B,), retained through autoresets
        w, x, y, z = result.state.orientation  # each (B,)
        dynamics._orientation = type(dynamics._orientation)(w=w, x=x, y=y, z=z)
        dynamics._velocity = result.state.velocity  # (3, B)
        dynamics._angular_velocity = result.state.angular_velocity  # (3, B)
        dynamics._motor_omega = result.state.motor_omega  # (4, B)
        dynamics._thrusts = result.thrust  # (4, B)
        dynamics._acc = result.acceleration  # (3, B)
        dynamics._angular_acc = result.angular_acceleration  # (3, B)
        dynamics._t += dynamics.ctrl_dt
        dynamics._ugly_fix()
        return dynamics.state  # (B, 13)
