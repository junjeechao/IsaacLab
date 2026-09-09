# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for the bridge's tracking and camera contracts (no Kit required)."""

import __future__

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_class(filename, name):
    tree = ast.parse(Path(__file__).with_name(filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"np": np, "torch": torch, "Camera": object, "contextlib": contextlib}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), filename, "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


def gate():
    cls = load_class("run_g1_bridge_teleop.py", "_TeleopEngageGate")
    obj = cls.__new__(cls)
    obj._teleop = SimpleNamespace(last_buttons_left=True, last_buttons_right=True)
    obj._tracked = bool
    obj._edges = lambda *_: (False, False)
    obj.engaged = True
    obj._input_lost = False
    obj._need_clutch = False
    obj._warned_untracked = False
    obj._hold_action = torch.zeros(1, 32)
    obj._refresh_hold = lambda: setattr(obj, "_hold_action", torch.ones(1, 32))
    obj._clutch_mode = "full"
    obj._capture_clutch = lambda _: setattr(obj, "captured", True)
    obj._apply_clutch = lambda a: a
    obj._apply_locomotion = lambda a, *_: a
    return obj


def action():
    value = torch.zeros(1, 32)
    value[:, 6] = value[:, 13] = 1
    return value


@pytest.mark.parametrize("loss", ["missing", "untracked", "nan", "zero_quaternion"])
def test_loss_holds_current_pose_and_reclutches(loss):
    obj = gate()
    value = action()
    if loss == "missing":
        value = None
    elif loss == "untracked":
        obj._teleop.last_buttons_left = False
    elif loss == "nan":
        value[0, 0] = float("nan")
    else:
        value[0, 3:7] = 0
    assert torch.equal(obj.process(value), torch.ones(1, 32))
    assert obj._need_clutch
    # Repeated loss must freeze the captured hold, not chase a drifting robot.
    obj._refresh_hold = lambda: pytest.fail("hold recaptured during the same outage")
    obj.process(value)
    obj._teleop.last_buttons_left = True
    obj.process(action())
    assert obj.captured
    assert not obj._need_clutch


def test_full_clutch_preserves_initial_wrist_pose():
    obj = gate()
    del obj._capture_clutch
    del obj._apply_clutch
    wrists = (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    )
    obj._read_wrist_world = lambda: wrists
    value = action()[0]
    obj._capture_clutch(value)
    result = obj._apply_clutch(value)
    np.testing.assert_allclose(result[:14], np.concatenate(wrists), atol=1e-6)


def test_stereo_matches_physics_epoch_not_update_counter():
    cls = load_class("bridge_camera.py", "BridgeCamera")
    left = cls.__new__(cls)
    left.cfg = SimpleNamespace(stereo_role="left")
    left._capture_step = 10
    left._render_step = 20
    right = SimpleNamespace(_capture_step=9, _render_step=20)
    globals_ = cls._get_right_eye_camera.__globals__
    globals_["_STEREO_REGISTRY"] = {"right": right}
    globals_["logger"] = SimpleNamespace(warning=lambda *_: None)
    assert left._get_right_eye_camera() is None
    right._capture_step = 10
    assert left._get_right_eye_camera() is right


@pytest.mark.parametrize("head_enabled", [True, False])
def test_camera_engage_head_rotation_torso_translation(monkeypatch, head_enabled):
    import sys

    from scipy.spatial.transform import Rotation as R

    basis = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    monkeypatch.setitem(sys.modules, "bridge_camera", SimpleNamespace(_R_USDCAM_TO_BODY=basis))
    cls = load_class("run_g1_bridge_teleop.py", "_StabilizedCameraDriver")
    poses = []
    right_poses = []
    torso = np.array([2.0, 3.0, 1.0])
    cam = SimpleNamespace(
        cfg=SimpleNamespace(
            offset=SimpleNamespace(pos=(0.1, 0.0, 0.5), rot=(0.0, 0.0, 0.0, 1.0)), stereo_baseline_m=0.063
        ),
        _read_prim_world_pose_from_fabric=lambda _: (torso.copy(), np.array([0.0, 0.0, 0.0, 1.0])),
        set_world_poses=lambda **kw: poses.append(kw),
    )
    right = SimpleNamespace(set_world_poses=lambda **kw: right_poses.append(kw))
    teleop = SimpleNamespace(head_quat_world_xyzw=None)
    state = SimpleNamespace(engaged=False)
    driver = cls(SimpleNamespace(device="cpu"), cam, right, teleop, state, 0.25, head_enabled=head_enabled)
    driver.step()
    initial = poses[-1]
    state.engaged = True
    teleop.head_quat_world_xyzw = R.from_euler("z", 70, degrees=True).as_quat()
    driver.step()
    np.testing.assert_allclose(poses[-1]["orientations"], initial["orientations"], atol=1e-6)
    teleop.head_quat_world_xyzw = R.from_euler("z", 90, degrees=True).as_quat()
    driver.step()
    q = poses[-1]["orientations"][0].numpy()
    expected = 20 if head_enabled else 0
    assert R.from_quat(q).magnitude() == pytest.approx(np.deg2rad(expected), abs=1e-6)
    np.testing.assert_allclose(poses[-1]["positions"], initial["positions"])
    delta = right_poses[-1]["positions"][0] - poses[-1]["positions"][0]
    np.testing.assert_allclose(delta, R.from_quat(q).apply([0.0, -0.063, 0.0]), atol=1e-6)
    teleop.head_quat_world_xyzw = None
    driver.step()
    np.testing.assert_allclose(poses[-1]["orientations"][0], q, atol=1e-6)
    teleop.head_quat_world_xyzw = R.from_euler("z", 150, degrees=True).as_quat()
    driver.step()
    np.testing.assert_allclose(poses[-1]["orientations"][0], q, atol=1e-6)
