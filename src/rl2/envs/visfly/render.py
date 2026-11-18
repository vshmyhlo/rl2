"""Host-side diagnostics for VisFly videos; pixels and navigation state stay on the CPU."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from rl2.envs.gym_style_env import STOP_REASON_NAMES, StopReason


@dataclass(frozen=True)
class RenderContext:
    step: int
    episode_return: float
    reward: float | None = None
    action: NDArray[Any] | None = None  # (A,) last applied normalized action
    stop_reason: int | None = None


class NavigationDiagnostics(NamedTuple):
    distance: float
    closing_speed: float
    speed: float
    tilt_degrees: float
    body_goal: NDArray[np.float64]  # (3,) forward, left, up displacement in metres


def depth_panel(depth: NDArray[Any], *, height: int) -> NDArray[np.uint8]:  # (H_depth, W_depth, 1) -> (H, W, 3)
    """Return (H, W, 3) RGB depth with preserved aspect ratio and a fixed 0–20 m inferno scale.

    Near surfaces are dark; distances at or above 20 m (including upstream
    background readings) are pale yellow. Labels are drawn at display resolution.
    """
    pixels = np.rint(np.clip(depth[..., 0] / 20.0, 0.0, 1.0) * 255).astype(np.uint8)  # (H_depth, W_depth)
    colors = cv2.cvtColor(cv2.applyColorMap(pixels, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)  # (H_depth, W_depth, 3)
    width = max(1, round(depth.shape[1] * height / depth.shape[0]))
    panel = cv2.resize(colors, (width, height), interpolation=cv2.INTER_NEAREST)  # (H, W, 3)
    cv2.putText(panel, "Depth: 0-20 m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, "Depth: 0-20 m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel  # (H, W, 3)


def navigation_diagnostics(
    state: NDArray[Any],  # (S,) S=12 Euler or S=13 wxyz quaternion
    target: NDArray[Any],  # (3,)
) -> NavigationDiagnostics:
    """Compute host navigation diagnostics for the OpenCV panel, outside training paths."""
    rotation = (
        Rotation.from_quat(np.roll(state[3:7], -1)) if state.shape[0] == 13 else Rotation.from_euler("xyz", state[3:6])
    )
    goal = target - state[:3]  # (3,)
    distance = float(np.linalg.norm(goal))
    return NavigationDiagnostics(
        distance=distance,
        closing_speed=float(np.dot(state[-6:-3], goal) / distance) if distance > 0 else 0.0,
        speed=float(np.linalg.norm(state[-6:-3])),
        tilt_degrees=float(np.degrees(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))),
        body_goal=rotation.apply(goal, inverse=True),  # (3,)
    )


def diagnostics_panel(
    state: NDArray[Any],  # (S,)
    target: NDArray[Any],  # (3,)
    *,
    height: int,
    action_low: NDArray[Any],  # (A,)
    action_high: NDArray[Any],  # (A,)
    action_type: str,
    context: RenderContext | None = None,
    max_episode_steps: int | None = None,
) -> NDArray[np.uint8]:  # (H, round(H * 1.5), 3)
    """Return (H, round(H * 1.5), 3) RGB diagnostics, scaled from a 384x256 canvas.

    Closing speed is velocity projected toward the goal, positive on approach.
    Tilt is the angle between body and world up. The arrow projects the goal
    onto the body XY plane (forward up, left left); vertical displacement is numeric.
    Rewards/actions describe the previous transition. Final cards retain the last recorded views.
    """
    panel = np.full((256, 384, 3), (20, 25, 34), dtype=np.uint8)  # (H_panel, W_panel, 3)
    foreground = (225, 232, 240)
    accent = (90, 215, 190)

    def text(label: str, x: int, y: int, color: tuple[int, int, int] = foreground) -> None:
        cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    data = navigation_diagnostics(state, target)
    title = "FLIGHT DIAGNOSTICS"
    final = context is not None and context.stop_reason not in (None, StopReason.ONGOING)
    if final:
        title = STOP_REASON_NAMES[context.stop_reason].replace("_", " ").upper()
    text(title, 10, 18, accent)
    if context is not None:
        limit = f" / {max_episode_steps}" if max_episode_steps is not None else ""
        text(f"Step {context.step}{limit}" + (" | views: last recorded" if final else ""), 10, 36)
    text(f"Goal distance  {data.distance:.2f} m", 10, 56)
    text(f"Closing speed  {data.closing_speed:+.2f} m/s", 10, 74)
    text(f"Speed {data.speed:.2f} | vz {state[-4]:+.2f} m/s", 10, 92)
    text(f"Tilt from upright  {data.tilt_degrees:.1f} deg", 10, 110)
    forward, left, up = data.body_goal
    text(f"Goal F {forward:+.1f}  L {left:+.1f}  U {up:+.1f} m", 10, 130)
    cv2.circle(panel, (323, 81), 27, (75, 85, 100), 1, cv2.LINE_AA)
    text("F", 319, 48)
    text("L", 281, 85)
    horizontal = float(np.linalg.norm(data.body_goal[:2]))
    if horizontal > 1e-6:
        end = (round(323 - 25 * left / horizontal), round(81 - 25 * forward / horizontal))
        cv2.arrowedLine(panel, (323, 81), end, accent, 2, cv2.LINE_AA, tipLength=0.3)
    else:
        cv2.circle(panel, (323, 81), 3, accent, -1)
    if context is not None:
        reward = f"Prev reward {context.reward:+.3f} | " if context.reward is not None else ""
        text(f"{reward}Return {context.episode_return:+.3f}", 10, 150)
    if context is not None and context.action is not None:
        text("Last action (normalized; bars show bounds)", 10, 170)
        labels = ("Thrust", "Rate X", "Rate Y", "Rate Z") if action_type == "bodyrate" else ()
        for index, (action, low, high) in enumerate(zip(context.action, action_low, action_high, strict=True)):
            y = 188 + index * 18
            label = labels[index] if labels else f"Action {index}"
            text(label, 10, y)
            cv2.rectangle(panel, (108, y - 9), (252, y), (65, 75, 90), -1)
            fraction = float(np.clip((action - low) / (high - low), 0, 1))
            color = (255, 170, 80) if fraction <= 0.02 or fraction >= 0.98 else accent
            cv2.rectangle(panel, (108, y - 9), (108 + round(144 * fraction), y), color, -1)
            text(f"{action:+.2f} [{low:g},{high:g}]", 263, y, color)
    return cv2.resize(panel, (max(1, round(height * 1.5)), height), interpolation=cv2.INTER_AREA)
