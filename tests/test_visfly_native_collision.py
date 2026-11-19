"""Exercise the production C++ probe loop against OMPL without a renderer."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl2.envs.visfly.collision import SceneCollision
from rl2.envs.visfly.planning import make_state, space_information


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required; run in the VisFly image")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("native_collision")
    source = directory / "test.cpp"
    source.write_text("""
#include "route_collision.h"
extern "C" bool point(const double* position, double radius, const float* hit,
                     bool outside, float* transformed) {
  return rl2::routePointCollides({position[0], position[1], position[2]}, radius,
      [&](const std::array<float, 3>& p, float r) {
        for (int i = 0; i < 3; ++i) transformed[i] = p[i];
        return rl2::CollisionHit{{hit[0], hit[1], hit[2]}, outside};
      });
}
extern "C" bool motion(const double* start, const double* end, unsigned int count,
                      double radius, int blocked, float* probes, int* size) {
  *size = 0;
  return rl2::routeMotionIsValid({start[0], start[1], start[2]},
      {end[0], end[1], end[2]}, count, radius,
      [&](const std::array<float, 3>& p, float r) {
        for (int i = 0; i < 3; ++i) probes[*size * 3 + i] = p[i];
        return rl2::CollisionHit{p, (*size)++ == blocked};
      });
}
""")
    library = directory / "test.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(root / "infra/k3s/training/visfly"),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    result = ctypes.CDLL(str(library))
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    float_array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    result.point.argtypes = [double_array, ctypes.c_double, float_array, ctypes.c_bool, float_array]
    result.point.restype = ctypes.c_bool
    result.motion.argtypes = [
        double_array,
        double_array,
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_int,
        float_array,
        ctypes.POINTER(ctypes.c_int),
    ]
    result.motion.restype = ctypes.c_bool
    return result


def test_native_point_precision_and_boundaries(native: ctypes.CDLL) -> None:
    position = np.array([1 + 2**-25, 2, 3], dtype=np.float64)  # (3,)
    habitat = np.array([-2, 3, -1], dtype=np.float32)  # (3,)
    transformed = np.empty(3, dtype=np.float32)  # (3,)
    radius = 0.5
    for distance, outside, expected in (
        (0.5 - 2**-24, False, True),
        (0.5, False, False),
        (0.75, False, False),
        (0.75, True, True),
    ):
        # Offset the axis with smaller magnitude to retain the boundary ULP.
        hit = habitat.copy()  # (3,)
        hit[2] += distance
        assert native.point(position, radius, hit, outside, transformed) == expected
        np.testing.assert_array_equal(transformed, habitat)


@pytest.mark.parametrize("blocked", [-1, 0, 2], ids=["clear", "end_blocked", "interior_blocked"])
def test_native_motion_matches_ompl_probe_order(native: ctypes.CDLL, blocked: int) -> None:
    probes = np.empty((100, 3), dtype=np.float32)  # (P, 3), Habitat coordinates
    size = ctypes.c_int()
    reference: list[NDArray[np.float32]] = []  # each (3,)

    def collision(points: NDArray[np.float32], radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        reference.append(points[0].copy())
        return np.array([len(reference) - 1 == blocked])  # (1,)

    def motion(start: tuple[float, ...], end: tuple[float, ...], count: int, radius: float) -> bool:  # (3,), (3,)
        # With radius zero, the fake geometry's coincident hit is clear except
        # when its out-of-bounds flag marks the selected probe.
        return native.motion(np.array(start), np.array(end), count, 0, blocked, probes, ctypes.byref(size))

    lower, upper = torch.zeros(3), torch.full((3,), 10.0)  # each (3,)
    standard = space_information(lower, upper, collision, radius=0.1, step=0.2)
    accelerated = space_information(
        lower, upper, SceneCollision(scene=SimpleNamespace(route_motion_is_valid=motion)), radius=0.1, step=0.2
    )
    for start, end in (
        ((1.00000004, 2.0, 3.0), (2.31, 2.27, 3.14)),  # several subdivision levels
        ((1.0, 2.0, 3.0), (1.05, 2.0, 3.0)),  # one segment
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),  # zero-length motion
    ):
        reference.clear()
        expected = standard.checkMotion(make_state(standard, start), make_state(standard, end))
        actual = accelerated.checkMotion(make_state(accelerated, start), make_state(accelerated, end))
        assert actual == expected
        expected_points = np.stack(reference)[:, [1, 2, 0]] * np.array([-1, 1, -1], dtype=np.float32)  # (P, 3)
        np.testing.assert_array_equal(probes[: size.value], expected_points)


def test_native_collision_batch_layout() -> None:
    def query(points: NDArray[np.float32], radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        assert points.flags.c_contiguous
        return points[:, 0] < radius  # (N,)

    collision = SceneCollision(scene=SimpleNamespace(route_points_collide=query))
    points = np.arange(18, dtype=np.float32).reshape(3, 6)[:, ::2]  # (3, 3), strided
    result = collision(points, 1)  # (3,)
    assert result.shape == (3,) and result.dtype == np.bool_
    assert result.tolist() == [True, False, False]
    assert collision(points[:0], 1).shape == (0,)
