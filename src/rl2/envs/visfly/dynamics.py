"""Compiled CPU Euler substeps; VisFly retains control, wind, delays, and resets."""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpy.typing import NDArray


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
    pure = jnp.concatenate((jnp.zeros_like(vector[:1]), vector))  # (4, B)
    conjugate = orientation * jnp.array([1, -1, -1, -1], dtype=orientation.dtype)[:, None]  # (4, B)
    return quaternion_product(quaternion_product(orientation, pure), conjugate)[1:]  # (3, B)


@partial(jax.jit, backend="cpu", static_argnames=("steps", "delay"))
def integrate_euler(
    state: FlightState[jax.Array],  # leaves: (3, B) or (4, B)
    parameters: PhysicsParameters[jax.Array],
    desired_thrust: jax.Array,  # (4, B)
    wind: jax.Array,  # (3, 1) or (3, B)
    dt: float,
    *,
    steps: int,
    delay: bool,
) -> PhysicsResult[jax.Array]:
    """Advance the unchanged motor/Euler equations through all control-interval substeps.

    state: (3, B)/(4, B) state leaves in VisFly's component-major layout.
    desired_thrust: (4, B) clamped controller output held over the interval.
    wind: (3, 1) or (3, B) additive position velocity, matching upstream.
    Returns: (3, B)/(4, B) state and final-substep thrust/accelerations.
    """
    a, b, c = parameters.thrust_map  # each ()
    desired_omega = (-b + jnp.sqrt(b * b - 4 * a * (c - desired_thrust))) / (2 * a)  # (4, B)

    def substep(index: int, result: PhysicsResult[jax.Array]) -> PhysicsResult[jax.Array]:
        position, orientation, velocity, angular_velocity, omega = result.state  # (3/4, B)
        if delay:
            omega = parameters.motor_decay * omega + (1 - parameters.motor_decay) * desired_omega  # (4, B)
            thrust = a * omega**2 + b * omega + c  # (4, B)
        else:
            thrust = desired_thrust  # (4, B)
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
        angular_acceleration = parameters.inertia_inverse @ (
            force_torque[1:] - jnp.cross(angular_velocity, parameters.inertia @ angular_velocity, axis=0)
        )  # (3, B)
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
        )

    initial = PhysicsResult(
        state=state,
        thrust=desired_thrust,
        acceleration=jnp.zeros_like(state.velocity),
        angular_acceleration=jnp.zeros_like(state.angular_velocity),
    )
    return jax.lax.fori_loop(0, steps, substep, initial)


def host_array(value: torch.Tensor) -> NDArray[np.float32]:  # (*S,) -> (*S,)
    return value.detach().numpy()  # (*S,)


def writable_tensor(value: NDArray[np.float32]) -> torch.Tensor:  # (*S,) -> (*S,)
    # Upstream partial resets mutate these arrays; JAX host buffers are read-only.
    return torch.from_numpy(value.copy())  # (*S,)


class CompiledEuler:
    """Bridge CPU Torch state to one fused control-interval executable."""

    def __init__(self, dynamics: Any) -> None:
        self.dynamics = dynamics
        self.device = jax.devices("cpu")[0]
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
                )
            ),
        )
        dynamics._position = result.state.position  # (3, B)
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
