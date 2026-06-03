# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomanipulation runner with standalone OpenXR teleop + bridge publishing.

Coexists with another OpenXR client (the realtime-3d-reconstruction
``isaaclab_3d_recon`` app in ``render_mode: xr``) by opening its own
:class:`isaacteleop.teleop_session_manager.TeleopSession` directly -- not
through Kit's XR system. The container owns the foreground CloudXR session
and renders the nvblox mesh to the headset; this runner is an input-only
second OpenXR client polling head + controller pose. Pattern mirrors
``gr00t/external_dependencies/isaac_teleop_app/groot_teleop/``.

The retargeting pipeline (``_build_g1_locomanipulation_pipeline``) is imported
verbatim from the upstream env so behavior matches
``Isaac-PickPlace-Locomanipulation-G1-Abs-v0``.

Usage from the IsaacLab repo root::

    ./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --enable_cameras

Do **not** pass ``--xr``: Kit must stay out of OpenXR.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Make sibling bridge modules importable both when this script is run via
# ``./isaaclab.sh -p`` and when imported by Isaac Lab's class_type loader.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description=(
            "Drive the G1 locomanipulation env from a standalone OpenXR client"
            " while streaming RGB-D frames to the realtime-3d-reconstruction"
            " Holoscan app."
        )
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="If > 0, exit after this many env steps. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--bridge_endpoint", type=str, default=None,
        help="Override BridgeCameraCfg.endpoint (default tcp://127.0.0.1:5570).",
    )
    parser.add_argument(
        "--heartbeat_every", type=int, default=60,
        help="Print a status line every N env steps.",
    )
    parser.add_argument(
        "--xr_retry_seconds", type=float, default=2.0,
        help="Retry TeleopSession every N seconds while no XR client is connected.",
    )
    parser.add_argument(
        "--xr_recenter_endpoint", type=str, default="tcp://127.0.0.1:5560",
        help="ZMQ REQ endpoint of XrComposeOp's recenter listener.",
    )
    parser.add_argument(
        "--xr_recenter_delay_steps", type=int, default=10,
        help="Steps to wait after TeleopSession connects before sending the recenter request.",
    )
    parser.add_argument(
        "--left_quat_offset_rpy_deg", type=str, default="0,0,0",
        help="roll,pitch,yaw degrees right-multiplied onto the left wrist target quat.",
    )
    parser.add_argument(
        "--right_quat_offset_rpy_deg", type=str, default="0,0,0",
        help="roll,pitch,yaw degrees right-multiplied onto the right wrist target quat.",
    )
    parser.add_argument(
        "--camera_pitch_down_deg", type=float, default=30.0,
        help=(
            "Tilt the bridge camera down by this many degrees from horizontal."
            " Applied as an extra rotation around the parent's +Y (lateral)"
            " axis in OffsetCfg.rot, so both the rendered RGB-D and the"
            " published T_cam_world reflect it."
        ),
    )
    parser.add_argument(
        "--demo", action="store_true",
        help=(
            "Programmatically explore the scene: after 3 s, yaw left 30 deg,"
            " yaw right back to -30 deg, then move back 0.5 m. Teleports the"
            " robot root each step; locomotion command is zeroed and the"
            " hold-pose action is recomputed so wrists follow the body."
        ),
    )
    parser.add_argument(
        "--demo_start_delay_s", type=float, default=6.0,
        help="Seconds to wait before starting the --demo sequence.",
    )
    parser.add_argument(
        "--keyboard", action="store_true",
        help=(
            "Manually drive the rendering camera with the keyboard"
            " (replaces --demo's automated trajectory; can be combined with"
            " --demo, in which case the trajectory is disabled and only the"
            " initial-delay anchor capture is reused). Bindings: W/S forward"
            "/back, A/D strafe, Q/E down/up, arrow keys yaw + pitch, shift"
            " 5x speed, space recenter."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports that depend on Kit being live must come after AppLauncher.
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from locomanipulation_g1_bridge_env_cfg import LocomanipulationG1BridgeEnvCfg

    env_cfg = LocomanipulationG1BridgeEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.bridge_endpoint is not None:
        env_cfg.scene.bridge_camera.endpoint = args.bridge_endpoint
    if abs(args.camera_pitch_down_deg) > 0.0:
        # Rotate around parent's +Y axis (lateral) so +X (forward) tilts
        # toward -Z (down). xyzw quaternion.
        import math
        half = math.radians(args.camera_pitch_down_deg) * 0.5
        env_cfg.scene.bridge_camera.offset.rot = (
            0.0, math.sin(half), 0.0, math.cos(half)
        )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(
        f"[bridge-teleop] env built; action_dim={env.action_manager.total_action_dim}"
        f" device={env.device}",
        flush=True,
    )

    try:
        # ``AgileBasedLowerBodyAction`` runs a TorchScript locomotion policy
        # whose outputs carry ``requires_grad=True``. Warp's kernel launcher
        # rejects grad-tracking tensors. Wrap the whole loop in no_grad so the
        # policy forward pass and every downstream tensor stay out of autograd.
        torch_no_grad = torch.no_grad()
        torch_no_grad.__enter__()
        no_grad_active = True

        env.reset()

        teleop = _StandaloneG1TeleopSession(
            env=env,
            left_quat_offset_rpy_deg=_parse_rpy(args.left_quat_offset_rpy_deg),
            right_quat_offset_rpy_deg=_parse_rpy(args.right_quat_offset_rpy_deg),
        )
        hold_action = _build_hold_pose_action(env)
        if hold_action is None:
            hold_action = torch.zeros(
                (args.num_envs, env.action_manager.total_action_dim), device=env.device
            )

        demo_controller = None
        keyboard_controller = None
        if args.keyboard:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                # --keyboard replaces the automated --demo trajectory. The
                # initial-delay anchor capture is reused so the first pose is
                # stable, but no scripted motion runs.
                delay = args.demo_start_delay_s if args.demo else 0.0
                keyboard_controller = _KeyboardController(
                    bridge_cam_handle, env, delay
                )
                print(
                    f"[bridge-teleop] keyboard control enabled"
                    f" (initial delay {delay}s)",
                    flush=True,
                )
        elif args.demo:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                demo_controller = _DemoController(
                    bridge_cam_handle, env, args.demo_start_delay_s
                )
                print(
                    f"[bridge-teleop] demo enabled, starting in"
                    f" {args.demo_start_delay_s}s",
                    flush=True,
                )

        step = 0
        last_retry = 0.0
        recenter_sent = False
        connected_at_step = -1
        while simulation_app.is_running():
            now = time.monotonic()
            if teleop.session is None and (now - last_retry) >= args.xr_retry_seconds:
                last_retry = now
                teleop.try_connect()
                if teleop.session is not None:
                    connected_at_step = step

            action = teleop.step_action()
            if action is None:
                action = hold_action

            if keyboard_controller is not None:
                keyboard_controller.step()
            elif demo_controller is not None:
                demo_controller.step()

            try:
                env.step(action)
            except Exception:
                print("[bridge-teleop] env.step() raised; aborting", flush=True)
                traceback.print_exc()
                break

            # # One-shot: send "re-center_xr_poses" to the realtime-3d-recon
            # # XrComposeOp once the bridge has had a few frames to seed it
            # # with a T_cam_world reference. After this, headset=identity →
            # # robot head_link pose.
            # if (
            #     not recenter_sent
            #     and args.xr_recenter_endpoint
            #     and connected_at_step >= 0
            #     and step - connected_at_step >= args.xr_recenter_delay_steps
            # ):
            #     _send_recenter(args.xr_recenter_endpoint)
            #     recenter_sent = True

            step += 1
            if args.heartbeat_every > 0 and step % args.heartbeat_every == 0:
                print(
                    f"[bridge-teleop] step={step}"
                    f" xr_connected={teleop.session is not None}"
                    f" recenter_sent={recenter_sent}",
                    flush=True,
                )
            if args.max_steps > 0 and step >= args.max_steps:
                break

        print(f"[bridge-teleop] total steps={step}", flush=True)
    finally:
        if "no_grad_active" in locals() and no_grad_active:
            torch_no_grad.__exit__(None, None, None)
        if "teleop" in locals():
            teleop.close()
        if "keyboard_controller" in locals() and keyboard_controller is not None:
            keyboard_controller.close()
        bridge_cam = _get_bridge_camera(env)
        if bridge_cam is not None and hasattr(bridge_cam, "close"):
            bridge_cam.close()
        env.close()
        simulation_app.close()


# ---------------------------------------------------------------------------
# Standalone teleop wrapper
# ---------------------------------------------------------------------------


class _StandaloneG1TeleopSession:
    """Wraps :class:`isaacteleop.teleop_session_manager.TeleopSession` for our env.

    Builds the upstream locomanipulation pipeline, opens its own OpenXR client
    connection (no Kit handles), and exposes :meth:`step_action` returning a
    ``(num_envs, 32)`` action tensor ready for ``env.step``. A missing XR
    client is a soft failure: :meth:`step_action` returns ``None`` so the
    caller can fall back to a hold-pose action.
    """

    def __init__(
        self,
        env,
        left_quat_offset_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        right_quat_offset_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._env = env
        self._session = None
        self._session_ctx = None
        self._left_quat_offset_xyzw = _rpy_deg_to_quat_xyzw(left_quat_offset_rpy_deg)
        self._right_quat_offset_xyzw = _rpy_deg_to_quat_xyzw(right_quat_offset_rpy_deg)
        self._build_session_config()

    @property
    def session(self):
        return self._session

    def _build_session_config(self) -> None:
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
            _build_g1_locomanipulation_pipeline,
        )
        from isaacteleop.teleop_session_manager import TeleopSessionConfig

        self._pipeline = _build_g1_locomanipulation_pipeline()
        self._session_config = TeleopSessionConfig(
            app_name="IsaacLab3DReconBridgeTeleop",
            pipeline=self._pipeline,
        )

    def try_connect(self) -> bool:
        """Open the standalone TeleopSession. Returns True on success."""
        if self._session is not None:
            return True
        from isaacteleop.teleop_session_manager import TeleopSession

        try:
            self._session_ctx = TeleopSession(self._session_config)
            self._session = self._session_ctx.__enter__()
            print("[bridge-teleop] TeleopSession connected", flush=True)
            return True
        except RuntimeError as e:
            if "Failed to get OpenXR system" in str(e) or "OpenXR" in str(e):
                # CloudXR client not connected yet; will retry from the loop.
                self._session = None
                self._session_ctx = None
                return False
            raise

    def step_action(self):
        """Advance the teleop session by one tick and return a batched action.

        ``ValueError`` (scipy "zero norm quaternion" when OpenXR reports an
        uninitialized / out-of-tracking pose) is treated as transient: keep
        the session, return ``None``, throttled log. Other errors tear down
        the session so the retry loop creates a fresh one.
        """
        import torch

        if self._session is None:
            return None

        external_inputs = self._build_external_inputs()
        try:
            result = self._session.step(external_inputs=external_inputs)
        except ValueError as e:
            self._transient_error_count = getattr(self, "_transient_error_count", 0) + 1
            if (
                self._transient_error_count == 1
                or self._transient_error_count % 120 == 0
            ):
                print(
                    f"[bridge-teleop] TeleopSession.step() transient ValueError"
                    f" (count={self._transient_error_count}): {e}.",
                    flush=True,
                )
            return None
        except Exception as e:  # noqa: BLE001
            print(
                f"[bridge-teleop] TeleopSession.step() failed: {e!r}; closing session",
                flush=True,
            )
            self.close()
            return None

        if getattr(self, "_transient_error_count", 0) > 0:
            print(
                f"[bridge-teleop] TeleopSession recovered after"
                f" {self._transient_error_count} transient failures",
                flush=True,
            )
            self._transient_error_count = 0

        if result is None or "action" not in result:
            return None

        action_1d = torch.from_dlpack(result["action"][0]).to(
            dtype=torch.float32, device=self._env.device
        )

        # Per-wrist corrective rotation offsets (CLI-tunable).
        action_1d = _apply_wrist_quat_offsets(
            action_1d,
            self._left_quat_offset_xyzw,
            self._right_quat_offset_xyzw,
        )

        return action_1d.unsqueeze(0).repeat(self._env.num_envs, 1)

    def close(self) -> None:
        if self._session_ctx is not None:
            try:
                self._session_ctx.__exit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                print(f"[bridge-teleop] error closing session: {e!r}", flush=True)
        self._session = None
        self._session_ctx = None

    def _build_external_inputs(self):
        """Compute ``world_T_anchor`` via upstream's XrAnchorSynchronizer.

        Instantiates the upstream synchronizer with a no-op ``xr_core`` so the
        Fabric read of the pelvis + initial-quat capture + FOLLOW_PRIM_SMOOTHED
        yaw delta + ``anchor_pos`` offset + ``_OXR_TO_USD_ROTATION`` basis
        change all run identically to the ``--xr`` path. The only divergence:
        we don't drive Kit's renderer (the container's XrComposeOp owns that).
        """
        import numpy as np
        from isaacteleop.retargeting_engine.interface import TensorGroup, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        if self._session is None or not self._session.has_external_inputs():
            return None
        ext_specs = self._session.get_external_input_specs()
        if not ext_specs:
            return None

        if not hasattr(self, "_anchor_synchronizer"):
            from isaaclab_teleop.xr_anchor_utils import XrAnchorSynchronizer

            class _NoOpXrCore:
                def set_world_transform_matrix(self, *args, **kwargs):
                    return None

            self._anchor_synchronizer = XrAnchorSynchronizer(
                xr_core=_NoOpXrCore(),
                xr_cfg=self._env.cfg.xr,
                xr_anchor_headset_path="/World/_bridge_unused_xr_anchor",
            )

        self._anchor_synchronizer.sync_headset_to_anchor()
        transform = self._anchor_synchronizer.get_world_transform()
        if transform is None:
            return None
        anchor_pos, anchor_quat_xyzw = transform

        # Final 4x4 via upstream's pure-math builder. __new__ bypasses the
        # constructor's USD anchor-prim creation + XRCore wiring.
        from isaaclab_teleop.xr_anchor_manager import XrAnchorManager

        _builder = XrAnchorManager.__new__(XrAnchorManager)
        world_T_anchor = _builder._build_matrix(anchor_pos, anchor_quat_xyzw).astype(np.float32)

        external_inputs: dict = {}
        for leaf_name in ext_specs:
            if leaf_name == "world_T_anchor":
                xform_tg = TensorGroup(TransformMatrix())
                xform_tg[0] = world_T_anchor
                external_inputs[leaf_name] = {ValueInput.VALUE: xform_tg}
        return external_inputs if external_inputs else None


# ---------------------------------------------------------------------------
# Demo: programmatic camera-only exploration
# ---------------------------------------------------------------------------


class _DemoController:
    """Move the rendering camera only (robot untouched) on a fixed schedule.

    After ``initial_delay_s`` seconds, plays the sequence:

    * Phase 1: yaw left 30 deg @ 3.33 deg/s (= 1 deg / 0.3 s)        ->  9 s
    * Phase 2: yaw right 60 deg @ 3.33 deg/s (ends at -30 deg net)   -> 18 s
    * Phase 3: move back 0.5 m @ 3.33 cm/s (= 1 cm / 0.3 s)          -> 15 s

    Each step it computes a new camera world pose relative to the initial
    pose captured at demo start, calls ``Camera.set_world_poses(...)`` so the
    renderer moves accordingly, and sets a world-pose override on the bridge
    camera so the published ``T_cam_world`` matches what the renderer used.

    The yaw is around world +Z (so it preserves any pitch baked into
    OffsetCfg.rot, e.g. ``--camera_pitch_down_deg``). The "back" translation
    is along the yawed initial forward direction projected to horizontal.
    """

    # Rates are 3x slower than the original 10 deg/s and 10 cm/s, so motion
    # is gentle enough to keep nvblox integrations clean.
    _YAW_RATE_DEG_PER_S = 10.0 / 3.0
    _BACK_RATE_M_PER_S = 0.1 / 3.0

    # (phase_type, magnitude). magnitude is degrees for yaw_*, meters for back.
    _PHASES = [
        ("yaw_left", 30.0),
        ("yaw_right", 60.0),
        ("back", 0.5),
    ]

    def __init__(self, bridge_camera, env, initial_delay_s: float = 3.0) -> None:
        self._cam = bridge_camera
        self._env = env
        self._initial_delay_s = initial_delay_s
        self._start_time = time.monotonic()
        self._initial_pos_w: "np.ndarray | None" = None  # noqa: UP037
        self._initial_quat_xyzw: "np.ndarray | None" = None  # noqa: UP037
        self._initial_forward_horiz: "np.ndarray | None" = None  # noqa: UP037
        self._demo_start_time: float | None = None

    def step(self) -> None:
        if (time.monotonic() - self._start_time) < self._initial_delay_s:
            return
        if self._initial_pos_w is None and not self._capture_initial():
            return  # try again next step

        t = time.monotonic() - self._demo_start_time  # type: ignore[operator]
        yaw_deg, back_m = self._phase_progress(t)
        self._apply_pose(yaw_deg, back_m)

    def _capture_initial(self) -> bool:
        """Capture the camera's current world pose (Fabric + cfg.offset)."""
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        parent_pose = self._cam._read_parent_world_pose_from_fabric()
        if parent_pose is None:
            return False
        parent_pos_w, parent_quat_xyzw = parent_pose
        r_parent = R.from_quat(parent_quat_xyzw)
        offset_pos = np.array(list(self._cam.cfg.offset.pos), dtype=np.float64)
        offset_rot_xyzw = np.array(list(self._cam.cfg.offset.rot), dtype=np.float64)

        self._initial_pos_w = parent_pos_w + r_parent.as_matrix() @ offset_pos
        self._initial_quat_xyzw = (
            r_parent * R.from_quat(offset_rot_xyzw)
        ).as_quat()  # xyzw

        # Initial forward in world = camera body-frame +X rotated to world.
        forward_world = R.from_quat(self._initial_quat_xyzw).as_matrix() @ np.array(
            [1.0, 0.0, 0.0]
        )
        forward_world[2] = 0.0  # project to horizontal
        n = np.linalg.norm(forward_world)
        self._initial_forward_horiz = (
            forward_world / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        )

        self._demo_start_time = time.monotonic()
        return True

    def _phase_progress(self, t: float) -> tuple[float, float]:
        """Return cumulative ``(yaw_deg, back_m)`` at elapsed-demo-time ``t``."""
        yaw_deg = 0.0
        back_m = 0.0
        phase_elapsed = 0.0
        for phase_type, magnitude in self._PHASES:
            if phase_type in ("yaw_left", "yaw_right"):
                rate = self._YAW_RATE_DEG_PER_S
            else:
                rate = self._BACK_RATE_M_PER_S
            duration = magnitude / rate

            if t < phase_elapsed + duration:
                progress = t - phase_elapsed
                if phase_type == "yaw_left":
                    yaw_deg += progress * rate
                elif phase_type == "yaw_right":
                    yaw_deg -= progress * rate
                else:
                    back_m += progress * rate
                return yaw_deg, back_m
            if phase_type == "yaw_left":
                yaw_deg += magnitude
            elif phase_type == "yaw_right":
                yaw_deg -= magnitude
            else:
                back_m += magnitude
            phase_elapsed += duration
        return yaw_deg, back_m  # all phases done, hold at final values

    def _apply_pose(self, yaw_deg: float, back_m: float) -> None:
        import math
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R

        yaw_rad = math.radians(yaw_deg)
        # World-frame yaw delta applied on top of the initial orientation.
        delta_q = R.from_euler("z", yaw_rad).as_quat()  # xyzw
        new_quat_xyzw = (
            R.from_quat(delta_q) * R.from_quat(self._initial_quat_xyzw)
        ).as_quat()

        # Move back along the yawed initial-forward (horizontal projection).
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        fx, fy, _ = self._initial_forward_horiz  # type: ignore[misc]
        new_forward = np.array([c * fx - s * fy, s * fx + c * fy, 0.0])
        new_pos_w = self._initial_pos_w - back_m * new_forward  # type: ignore[operator]

        # 1. Move the rendered camera prim.
        device = self._env.device
        pos_t = torch.tensor(new_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
        quat_t = torch.tensor(new_quat_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
        self._cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="world")
        # The published T_cam_world is read back from the moved camera prim's
        # Fabric world matrix after the render, so it tracks this move
        # automatically -- no separate publisher override needed.


class _KeyboardController:
    """Move the rendering camera with the keyboard. Robot is untouched.

    Mirrors :class:`_DemoController`'s anchor-and-delta model: captures the
    bridge camera's pose at startup, then on each step composes the current
    accumulated yaw + pitch + translation deltas (driven by key-hold state)
    on top of the captured initial pose and pushes the result via
    ``Camera.set_world_poses``. The published ``T_cam_world`` is read back from
    the moved camera prim after the render, so it stays in sync automatically.

    Bindings:
      W / S        forward / back along yawed initial-forward (horizontal)
      A / D        strafe left / right
      Q / E        down / up (world +Z)
      Left/Right   yaw left / right (around world +Z)
      Up / Down    pitch up / down (around camera body +Y)
      Shift        5x speed multiplier while held
      Space        reset deltas (back to anchor pose)
    """

    _TRANSLATE_RATE_M_PER_S = 0.5
    _YAW_RATE_DEG_PER_S = 30.0
    _PITCH_RATE_DEG_PER_S = 30.0
    _FAST_MULTIPLIER = 5.0

    def __init__(self, bridge_camera, env, initial_delay_s: float = 0.0) -> None:
        self._cam = bridge_camera
        self._env = env
        self._initial_delay_s = initial_delay_s
        self._start_time = time.monotonic()
        self._initial_pos_w: "np.ndarray | None" = None  # noqa: UP037
        self._initial_quat_xyzw: "np.ndarray | None" = None  # noqa: UP037
        self._initial_forward_horiz: "np.ndarray | None" = None  # noqa: UP037

        import numpy as np
        self._pos_delta = np.zeros(3, dtype=np.float64)
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._last_step_t: float | None = None

        import threading
        self._pressed: set[str] = set()
        self._reset_requested = False
        self._lock = threading.Lock()

        from pynput import keyboard as _kb
        self._kb = _kb
        self._listener = _kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _key_name(key) -> str:
        """Normalize pynput key to a lowercase string we can compare on."""
        from pynput import keyboard as _kb
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        if isinstance(key, _kb.Key):
            return key.name.lower()
        return str(key).lower()

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            self._pressed.add(name)
            if name == "space":
                self._reset_requested = True
        # Treat both shift_l / shift_r as just "shift" too.
        if name.startswith("shift"):
            with self._lock:
                self._pressed.add("shift")

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        with self._lock:
            self._pressed.discard(name)
        if name.startswith("shift"):
            with self._lock:
                self._pressed.discard("shift")

    def step(self) -> None:
        if (time.monotonic() - self._start_time) < self._initial_delay_s:
            return
        if self._initial_pos_w is None and not self._capture_initial():
            return

        now = time.monotonic()
        if self._last_step_t is None:
            self._last_step_t = now
            return
        dt = now - self._last_step_t
        self._last_step_t = now

        with self._lock:
            pressed = set(self._pressed)
            reset = self._reset_requested
            self._reset_requested = False

        import math
        import numpy as np
        if reset:
            self._pos_delta = np.zeros(3, dtype=np.float64)
            self._yaw_deg = 0.0
            self._pitch_deg = 0.0

        fast = self._FAST_MULTIPLIER if "shift" in pressed else 1.0
        trans_step = self._TRANSLATE_RATE_M_PER_S * fast * dt
        yaw_step = self._YAW_RATE_DEG_PER_S * fast * dt
        pitch_step = self._PITCH_RATE_DEG_PER_S * fast * dt

        if "left" in pressed:
            self._yaw_deg += yaw_step
        if "right" in pressed:
            self._yaw_deg -= yaw_step
        if "up" in pressed:
            self._pitch_deg += pitch_step
        if "down" in pressed:
            self._pitch_deg -= pitch_step

        # Build yawed forward / right basis for translation in world frame.
        yaw_rad = math.radians(self._yaw_deg)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        fx, fy, _ = self._initial_forward_horiz  # type: ignore[misc]
        forward = np.array([c * fx - s * fy, s * fx + c * fy, 0.0])
        right = np.array([forward[1], -forward[0], 0.0])  # 90deg clockwise of forward
        up = np.array([0.0, 0.0, 1.0])

        if "w" in pressed:
            self._pos_delta += forward * trans_step
        if "s" in pressed:
            self._pos_delta -= forward * trans_step
        if "d" in pressed:
            self._pos_delta += right * trans_step
        if "a" in pressed:
            self._pos_delta -= right * trans_step
        if "e" in pressed:
            self._pos_delta += up * trans_step
        if "q" in pressed:
            self._pos_delta -= up * trans_step

        self._apply_pose(self._yaw_deg, self._pitch_deg, self._pos_delta)

    def _capture_initial(self) -> bool:
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        parent_pose = self._cam._read_parent_world_pose_from_fabric()
        if parent_pose is None:
            return False
        parent_pos_w, parent_quat_xyzw = parent_pose
        r_parent = R.from_quat(parent_quat_xyzw)
        offset_pos = np.array(list(self._cam.cfg.offset.pos), dtype=np.float64)
        offset_rot_xyzw = np.array(list(self._cam.cfg.offset.rot), dtype=np.float64)

        self._initial_pos_w = parent_pos_w + r_parent.as_matrix() @ offset_pos
        self._initial_quat_xyzw = (
            r_parent * R.from_quat(offset_rot_xyzw)
        ).as_quat()

        forward_world = R.from_quat(self._initial_quat_xyzw).as_matrix() @ np.array(
            [1.0, 0.0, 0.0]
        )
        forward_world[2] = 0.0
        n = np.linalg.norm(forward_world)
        self._initial_forward_horiz = (
            forward_world / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        )
        print(
            "[bridge-teleop] keyboard control armed; bindings:"
            " W/S fwd/back, A/D strafe, Q/E down/up, arrows yaw+pitch,"
            " shift=5x, space=recenter",
            flush=True,
        )
        return True

    def _apply_pose(self, yaw_deg: float, pitch_deg: float, pos_delta) -> None:
        import math
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R

        # Pitch is local (camera body +Y), applied first; yaw is world (+Z),
        # applied last. Composition: R_world_new = R_yaw * R_initial * R_pitch_body.
        yaw_q = R.from_euler("z", math.radians(yaw_deg))
        pitch_q = R.from_euler("y", math.radians(pitch_deg))
        new_quat_xyzw = (
            yaw_q * R.from_quat(self._initial_quat_xyzw) * pitch_q
        ).as_quat()
        new_pos_w = (self._initial_pos_w + pos_delta).astype(np.float64)  # type: ignore[operator]

        device = self._env.device
        pos_t = torch.tensor(new_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
        quat_t = torch.tensor(new_quat_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
        self._cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="world")
        # Published pose is read back from the moved camera prim post-render,
        # so it tracks this move without a separate publisher override.

    def close(self) -> None:
        try:
            self._listener.stop()
        except Exception:  # pragma: no cover - best-effort
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_bridge_camera(env):
    scene = env.scene
    try:
        return scene["bridge_camera"]
    except (KeyError, TypeError):
        pass
    return getattr(scene, "bridge_camera", None)


def _build_hold_pose_action(env):
    """Keep the wrists at their current world poses. Used while XR is idle."""
    import torch

    robot = env.scene["robot"]
    body_names = robot.body_names
    left_idx = _find_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_link"))
    right_idx = _find_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_link"))
    if left_idx is None or right_idx is None:
        return None

    pos = robot.data.body_pos_w
    quat_wxyz = robot.data.body_quat_w
    left_pos = pos[:, left_idx]
    right_pos = pos[:, right_idx]
    left_q = _wxyz_to_xyzw(quat_wxyz[:, left_idx])
    right_q = _wxyz_to_xyzw(quat_wxyz[:, right_idx])

    num_envs = pos.shape[0]
    hand_joints = torch.zeros((num_envs, 14), device=env.device)
    locomotion = torch.zeros((num_envs, 4), device=env.device)
    action = torch.cat([left_pos, left_q, right_pos, right_q, hand_joints, locomotion], dim=-1)
    return action.detach().clone()


def _find_body_index(body_names, candidates):
    for name in candidates:
        if name in body_names:
            return body_names.index(name)
    return None


def _wxyz_to_xyzw(quat_wxyz):
    import torch

    return torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)


def _parse_rpy(s: str) -> tuple[float, float, float]:
    """Parse a ``"roll,pitch,yaw"`` comma-separated string of degrees."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'roll,pitch,yaw', got {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _rpy_deg_to_quat_xyzw(rpy_deg: tuple[float, float, float]) -> "np.ndarray | None":
    """Convert an intrinsic XYZ Euler triple (degrees) to a quat ``(x, y, z, w)``.

    Returns ``None`` when all three angles are zero so the caller can skip
    the multiplication entirely.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    if all(abs(a) < 1e-9 for a in rpy_deg):
        return None
    return R.from_euler("xyz", rpy_deg, degrees=True).as_quat().astype(np.float32)


def _apply_wrist_quat_offsets(
    action_1d,
    left_offset_xyzw,
    right_offset_xyzw,
):
    """Right-multiply per-side offset quats onto the wrist targets in the action."""
    if left_offset_xyzw is None and right_offset_xyzw is None:
        return action_1d

    import torch

    action_1d = action_1d.clone()
    if left_offset_xyzw is not None:
        left_q = action_1d[3:7].detach().cpu().numpy()
        new_q = _quat_xyzw_mul(left_q, left_offset_xyzw)
        action_1d[3:7] = torch.from_numpy(new_q).to(action_1d.device, action_1d.dtype)
    if right_offset_xyzw is not None:
        right_q = action_1d[10:14].detach().cpu().numpy()
        new_q = _quat_xyzw_mul(right_q, right_offset_xyzw)
        action_1d[10:14] = torch.from_numpy(new_q).to(action_1d.device, action_1d.dtype)
    return action_1d


def _quat_xyzw_mul(a, b):
    """Quaternion multiplication ``a * b``, both in ``(x, y, z, w)`` order."""
    import numpy as np

    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _send_recenter(endpoint: str) -> None:
    """Send the one-shot ``re-center_xr_poses`` ZMQ request to XrComposeOp.

    Best-effort: a missing or unreachable endpoint is logged and ignored so
    the sim loop never stalls on it. Currently unused -- the auto-recenter
    call site in ``main()`` is commented out; re-enable when needed.
    """
    try:
        import zmq
    except ImportError:
        print("[bridge-teleop] pyzmq not available; skipping XR recenter", flush=True)
        return

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 250)
    sock.setsockopt(zmq.SNDTIMEO, 250)
    try:
        sock.connect(endpoint)
        sock.send_string("re-center_xr_poses")
        reply = sock.recv_string()
        print(
            f"[bridge-teleop] XR recenter sent to {endpoint} reply={reply!r}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[bridge-teleop] XR recenter to {endpoint} failed: {e!r}",
            flush=True,
        )
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
