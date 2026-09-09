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
import contextlib
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, exit after this many env steps. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--bridge_endpoint",
        type=str,
        default=None,
        help="Override BridgeCameraCfg.endpoint (default tcp://127.0.0.1:5570).",
    )
    parser.add_argument(
        "--heartbeat_every",
        type=int,
        default=60,
        help="Print a status line every N env steps.",
    )
    parser.add_argument(
        "--xr_retry_seconds",
        type=float,
        default=2.0,
        help="Retry TeleopSession every N seconds while no XR client is connected.",
    )
    parser.add_argument(
        "--xr_recenter_endpoint",
        type=str,
        default="tcp://127.0.0.1:5560",
        help="ZMQ REQ endpoint of XrComposeOp's recenter listener.",
    )
    parser.add_argument(
        "--xr_recenter_delay_steps",
        type=int,
        default=10,
        help="Steps to wait after TeleopSession connects before sending the recenter request.",
    )
    parser.add_argument(
        "--left_quat_offset_rpy_deg",
        type=str,
        default="0,0,0",
        help="roll,pitch,yaw degrees right-multiplied onto the left wrist target quat.",
    )
    parser.add_argument(
        "--right_quat_offset_rpy_deg",
        type=str,
        default="0,0,0",
        help="roll,pitch,yaw degrees right-multiplied onto the right wrist target quat.",
    )
    parser.add_argument(
        "--camera_pitch_down_deg",
        type=float,
        default=0.0,
        help=(
            "Tilt the bridge camera down by this many degrees from horizontal."
            " Applied as an extra rotation around the parent's +Y (lateral)"
            " axis in OffsetCfg.rot, so both the rendered RGB-D and the"
            " published T_cam_world reflect it."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Programmatically explore the scene: after 3 s, yaw left 30 deg,"
            " yaw right back to -30 deg, then move back 0.5 m. Teleports the"
            " robot root each step; locomotion command is zeroed and the"
            " hold-pose action is recomputed so wrists follow the body."
        ),
    )
    parser.add_argument(
        "--demo_start_delay_s",
        type=float,
        default=6.0,
        help="Seconds to wait before starting the --demo sequence.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help=(
            "Manually drive the rendering camera with the keyboard"
            " (replaces --demo's automated trajectory; can be combined with"
            " --demo, in which case the trajectory is disabled and only the"
            " initial-delay anchor capture is reused). Bindings: W/S forward"
            "/back, A/D strafe, Q/E down/up, arrow keys yaw + pitch, shift"
            " 5x speed, space recenter."
        ),
    )
    parser.add_argument(
        "--camera_stabilize_s",
        type=float,
        default=0.25,
        help=(
            "Gimbal-style camera stabilization time constant [s]. The G1's"
            " balance policy micro-sways the torso ~0.2 deg / ~0.6 mm EVERY"
            " step (measured), which a rigidly mounted camera sees as ~3 px"
            " of image jitter per frame. With this > 0 the bridge cameras are"
            " parented to the WORLD instead of the torso and driven each step"
            " toward the torso mount point through a critically-damped"
            " low-pass with this time constant — smooth image, still follows"
            " the robot (and the XR head rotation when engaged). The"
            " published GT pose remains exact (read back from the camera"
            " prim). 0 = rigid torso mount (the physically-faithful shaky"
            " view). --keyboard/--demo take over the camera and pair best"
            " with 0."
        ),
    )
    parser.add_argument(
        "--pose_diag",
        action="store_true",
        help=(
            "Diagnostic: every heartbeat, print the bridge camera's motion"
            " statistics — camera-relative-to-torso (must be ~0 when no"
            " controller is driving it: any non-zero value is a self-inflicted"
            " pose write) and torso world motion (the robot's physical sway,"
            " which a torso-mounted camera legitimately sees)."
        ),
    )
    parser.add_argument(
        "--no_engage_gate",
        action="store_true",
        help=(
            "Disable the teleop engage gate and revert to the old always-on"
            " behavior (the robot follows the controllers the moment the XR"
            " session connects). By default the robot HOLDS its pose until"
            " you press the engage chord (X+Y+A+B together), the wrist"
            " targets are clutched to the robot's current hand poses at every"
            " engage (no jump), untracked controllers auto-hold, and the"
            " thumbsticks get a deadzone so drift can't walk/crouch the robot."
        ),
    )
    parser.add_argument(
        "--engage_chord",
        type=str,
        default="x+y+a+b",
        help=(
            "Button chord that toggles engage/hold (rising edge of ALL"
            " buttons held together). '+'-separated tokens: x/y = left"
            " primary/secondary, a/b = right primary/secondary, lsqueeze/"
            "rsqueeze, ltrigger/rtrigger, lmenu/rmenu, lstick/rstick."
            " Default: all four face buttons (x+y+a+b) — hard to hit by"
            " accident."
        ),
    )
    parser.add_argument(
        "--reset_chord",
        type=str,
        default="lsqueeze+a",
        help=(
            "Button chord (rising edge): while engaged, re-clutch the wrist"
            " targets to the robot's current hand poses and reset the hip"
            " height; while held (disengaged), re-capture the hold pose at"
            " the robot's current configuration. Same token syntax as"
            " --engage_chord. Default: left squeeze + right A."
        ),
    )
    parser.add_argument(
        "--engage_clutch",
        type=str,
        default="full",
        choices=("pos", "full", "off"),
        help=(
            "How wrist targets are anchored at engage so the hand continues"
            " from where it is instead of jumping to the controller's absolute"
            " pose. pos: offset positions only, rotations stay"
            " absolute (predictable 1:1 translation mapping; small rotation"
            " snap remains). full (default): offset the full SE(3) pose (zero snap, but"
            " your translations are re-expressed through the captured rotation"
            " offset). off: raw absolute targets (old behavior)."
        ),
    )
    parser.add_argument(
        "--stick_deadzone",
        type=float,
        default=0.15,
        help=(
            "Thumbstick deadzone for the locomotion command (0..1). Values"
            " below it read as zero (rescaled above it), so stick drift can't"
            " slowly walk, turn, or crouch the robot. The gate also owns the"
            " hip-height integration: it only accumulates while ENGAGED"
            " (the stock retargeter integrates drift even when idle)."
        ),
    )
    parser.add_argument(
        "--no_xr_head_camera",
        action="store_true",
        help=(
            "Disable the XR-head-driven virtual camera. By default (when an XR"
            " client is connected and neither --keyboard nor --demo is active)"
            " the bridge camera's ROTATION follows the raw XR headset"
            " orientation while its TRANSLATION stays bound to the robot torso"
            " (parent + cfg offset). The published T_cam_world reflects the"
            " XR-driven pose because it is read back from the moved camera prim"
            " after each render."
        ),
    )
    parser.add_argument(
        "--no_aa",
        action="store_true",
        help=(
            "Diagnostic: disable post-process anti-aliasing (default DLSS) on"
            " the bridge camera RGB. AA blends foreground/background colors at"
            " silhouettes, which leaks fg color onto bg surfaces in the nvblox"
            " color layer. Use to isolate the sim-side AA contribution to the"
            " boundary color bleed (does not affect real-camera leak)."
        ),
    )
    parser.add_argument(
        "--aa_off_spp",
        type=int,
        default=64,
        help=(
            "Samples per pixel to use when --no_aa is set. Disabling AA also"
            " disables the DLSS/TAA denoiser, so RTX RealTime's stochastic"
            " sampled lighting shows through as noise/holes. Raising"
            " samplesPerPixel makes the raw frame dense without a denoiser"
            " (higher = cleaner but slower). Ignored unless --no_aa is set."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:  # noqa: C901 - launch variants kept together for this integration checkpoint
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.enable_cameras = True
    if args.num_envs != 1:
        parser.error("The stereo teleop bridge supports exactly one environment.")
    if getattr(args, "xr", False):
        parser.error("Do not use --xr: televiz owns the foreground XR session.")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports that depend on Kit being live must come after AppLauncher.
    import torch
    from locomanipulation_g1_bridge_env_cfg import LocomanipulationG1BridgeEnvCfg

    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = LocomanipulationG1BridgeEnvCfg()
    env_cfg.scene.bridge_camera.publish_enabled = False
    # Continuous teleop must not inherit training episode resets.
    env_cfg.terminations = {}
    env_cfg.scene.bridge_camera.debug_window = False
    env_cfg.scene.num_envs = args.num_envs
    if args.bridge_endpoint is not None:
        env_cfg.scene.bridge_camera.endpoint = args.bridge_endpoint

    stabilize_s = max(0.0, float(args.camera_stabilize_s))
    if stabilize_s > 0.0 and (args.keyboard or args.demo):
        print(
            "[bridge-teleop] --keyboard/--demo drive the camera directly —"
            " disabling camera stabilization (rigid-mount paths)",
            flush=True,
        )
        stabilize_s = 0.0
    if stabilize_s > 0.0:
        # Stabilized (gimbal) mount: parent the cameras to the WORLD so the
        # torso's balance micro-sway cannot leak into them through the scene
        # graph; _StabilizedCameraDriver then follows the torso mount point
        # through a low-pass every step. cfg offsets stay as the nominal
        # mount definition the driver reads.
        env_cfg.scene.bridge_camera.prim_path = "/World/envs/env_.*/bridge_camera"
        if hasattr(env_cfg.scene, "bridge_camera_right"):
            env_cfg.scene.bridge_camera_right.prim_path = "/World/envs/env_.*/bridge_camera_right"
        print(
            f"[bridge-teleop] camera stabilization ON (tau={stabilize_s}s):"
            " cameras world-parented, low-pass follow of the torso mount"
            " (--camera_stabilize_s 0 for the rigid mount)",
            flush=True,
        )
    if abs(args.camera_pitch_down_deg) > 0.0:
        # Rotate around parent's +Y axis (lateral) so +X (forward) tilts
        # toward -Z (down). xyzw quaternion. Applied to both eyes so the
        # stereo pair stays rigid in the no-controller fallback.
        import math

        half = math.radians(args.camera_pitch_down_deg) * 0.5
        pitch_rot = (0.0, math.sin(half), 0.0, math.cos(half))
        env_cfg.scene.bridge_camera.offset.rot = pitch_rot
        if hasattr(env_cfg.scene, "bridge_camera_right"):
            env_cfg.scene.bridge_camera_right.offset.rot = pitch_rot

    if args.no_aa:
        # Hard-edged RGB silhouettes (no DLSS/TAA temporal fg/bg blend) so the
        # nvblox color layer isn't contaminated by anti-aliased edge pixels.
        env_cfg.sim.render.antialiasing_mode = "Off"
        env_cfg.sim.render.enable_dlssg = False
        # Disabling AA also disables the denoiser, so RTX RealTime's stochastic
        # sampled lighting would show through as noise/holes. Raise the per-pixel
        # sample count so the raw (un-denoised) frame is dense.
        env_cfg.sim.render.enable_direct_lighting = True
        env_cfg.sim.render.samples_per_pixel = args.aa_off_spp
        print(
            f"[bridge-teleop] anti-aliasing disabled (--no_aa); samples_per_pixel={args.aa_off_spp}",
            flush=True,
        )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(
        f"[bridge-teleop] env built; action_dim={env.action_manager.total_action_dim} device={env.device}",
        flush=True,
    )

    if args.no_aa:
        # Belt-and-suspenders, run AFTER the sim/render is up so the render
        # init can't overwrite it: hard-set the carb AA op to 0 (no AA) and
        # verify the active render mode. Hard edges + pixel-matched rgb/depth
        # require RaytracedLighting; RealTimePathTracing (e.g. --deterministic)
        # accumulates + denoises and re-softens edges.
        import carb

        _s = carb.settings.get_settings()
        _s.set_int("/rtx/post/aa/op", 0)  # 0 = no AA (off DLSS/TAA/FXAA)
        _s.set_bool("/rtx-transient/dlssg/enabled", False)
        _rendermode = _s.get("/rtx/rendermode")
        _aa_op = _s.get("/rtx/post/aa/op")
        print(
            f"[bridge-teleop] --no_aa enforced: /rtx/rendermode={_rendermode} /rtx/post/aa/op={_aa_op}",
            flush=True,
        )
        if _rendermode and _rendermode != "RaytracedLighting":
            print(
                f"[bridge-teleop] WARNING: /rtx/rendermode={_rendermode!r}, expected"
                " 'RaytracedLighting'. PathTracing accumulates + denoises and re-softens"
                " edges; do NOT pass --deterministic if you want hard, pixel-matched"
                " rgb/depth.",
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
            hold_action = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)

        engage_gate = None
        if not args.no_engage_gate:
            engage_gate = _TeleopEngageGate(
                env,
                teleop,
                initial_hold_action=hold_action,
                engage_chord=args.engage_chord,
                reset_chord=args.reset_chord,
                clutch_mode=args.engage_clutch,
                stick_deadzone=args.stick_deadzone,
            )

        demo_controller = None
        keyboard_controller = None
        xr_head_controller = None
        if args.keyboard:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                # --keyboard replaces the automated --demo trajectory. The
                # initial-delay anchor capture is reused so the first pose is
                # stable, but no scripted motion runs.
                delay = args.demo_start_delay_s if args.demo else 0.0
                keyboard_controller = _KeyboardController(bridge_cam_handle, env, delay)
                print(
                    f"[bridge-teleop] keyboard control enabled (initial delay {delay}s)",
                    flush=True,
                )
        elif args.demo:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                demo_controller = _DemoController(bridge_cam_handle, env, args.demo_start_delay_s)
                print(
                    f"[bridge-teleop] demo enabled, starting in {args.demo_start_delay_s}s",
                    flush=True,
                )

        camera_stabilizer = None
        if stabilize_s > 0.0:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                camera_stabilizer = _StabilizedCameraDriver(
                    env,
                    bridge_cam_handle,
                    _get_bridge_camera_right(env),
                    teleop,
                    engage_gate,
                    tau_s=stabilize_s,
                    head_enabled=not args.no_xr_head_camera,
                )
                # Warm-up drive so the FIRST rendered/published frame (which
                # seeds the bridge's pose-rebase anchor) already sits at the
                # torso mount, not at the world-parented spawn pose.
                camera_stabilizer.step()
        elif not args.no_xr_head_camera and keyboard_controller is None and demo_controller is None:
            bridge_cam_handle = _get_bridge_camera(env)
            if bridge_cam_handle is not None:
                xr_head_controller = _XrHeadCameraController(bridge_cam_handle, env, teleop)
                print(
                    "[bridge-teleop] XR head camera enabled: rotation follows"
                    " the headset, translation stays bound to the torso",
                    flush=True,
                )

        pose_diag = None
        bridge_cam = _get_bridge_camera(env)
        if bridge_cam is not None:
            bridge_cam.cfg.publish_enabled = True
        if args.pose_diag:
            diag_cam = _get_bridge_camera(env)
            if diag_cam is not None:
                pose_diag = _PoseDiag(diag_cam)

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

            raw_action = teleop.step_action()
            if engage_gate is not None:
                action = engage_gate.process(raw_action)
            else:
                action = raw_action if raw_action is not None else hold_action

            if keyboard_controller is not None:
                keyboard_controller.step()
            elif demo_controller is not None:
                demo_controller.step()
            elif camera_stabilizer is not None:
                # World-parented cameras MUST be driven every step (they no
                # longer ride the torso); head rotation inside is gated on
                # ENGAGED.
                camera_stabilizer.step()
            elif xr_head_controller is not None and (engage_gate is None or engage_gate.engaged):
                # Head look-around is gated on ENGAGED too: before the engage
                # chord the cameras must stay on their rigid torso mounts (no
                # pose writes at all — an XR session merely being connected
                # must not move anything).
                xr_head_controller.step()

            try:
                env.step(action)
            except Exception:
                print("[bridge-teleop] env.step() raised; aborting", flush=True)
                traceback.print_exc()
                break

            if pose_diag is not None:
                pose_diag.step()

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
                    f" engaged={engage_gate.engaged if engage_gate is not None else 'gate-off'}"
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
        bridge_cam_right = _get_bridge_camera_right(env)
        if bridge_cam_right is not None and hasattr(bridge_cam_right, "close"):
            bridge_cam_right.close()
        env.close()
        simulation_app.close()


# ---------------------------------------------------------------------------
# Standalone teleop wrapper
# ---------------------------------------------------------------------------


def _build_pipeline_with_head():
    """Wrap the upstream G1 locomanipulation pipeline with a raw-head-pose tap.

    The upstream pipeline exposes only the retargeted 32D ``action``; the raw
    headset orientation never surfaces (the anchor synchronizer's yaw comes
    from the pelvis prim, not the headset). Adding a :class:`HeadSource` --
    transformed by the same ``world_T_anchor`` matrix the controllers use --
    makes ``session.step()`` also return a ``"head"`` output: the headset pose
    in the Isaac world frame (OpenXR view-frame axes: -Z forward, +Y up). The
    head branch uses its own ``ValueInput`` leaf, ``world_T_anchor_head``; the
    caller feeds it the identical matrix (see ``_build_external_inputs``).
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import (
        ControllersSource,
        HeadSource,
    )
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
        _build_g1_locomanipulation_pipeline,
    )

    base = _build_g1_locomanipulation_pipeline()

    head_source = HeadSource(name="bridge_head")
    head_xform_input = ValueInput("world_T_anchor_head", TransformMatrix())
    transformed_head = head_source.transformed(head_xform_input.output(ValueInput.VALUE))

    # Parallel button/tracking tap (the SessionLifecycle "_button_controllers"
    # pattern): raw, untransformed ControllerInput groups so the runner's
    # engage gate can read button edges + grip validity. The main pipeline's
    # own ControllersSource is unreachable from its OutputCombiner, and a
    # second source is the supported idiom (each gets its own tracker).
    buttons = ControllersSource(name="bridge_buttons")

    return OutputCombiner(
        {
            **base.output_mapping,
            "head": transformed_head.output("head"),
            "buttons_left": buttons.output(ControllersSource.LEFT),
            "buttons_right": buttons.output(ControllersSource.RIGHT),
        }
    )


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
        # Latest raw headset orientation mapped into the Isaac world frame
        # (world_T_anchor applied; still in the OpenXR view-frame convention:
        # -Z forward, +Y up). ``None`` until the first valid head pose arrives.
        self.head_quat_world_xyzw = None
        # Latest raw ControllerInput tensor groups (Optional; may be None or
        # ``is_none`` when a controller is untracked). Consumed by the
        # engage gate for button edges, grip validity, and thumbstick values.
        self.last_buttons_left = None
        self.last_buttons_right = None
        self._build_session_config()

    @property
    def session(self):
        return self._session

    def _build_session_config(self) -> None:
        from isaacteleop.teleop_session_manager import TeleopSessionConfig

        self._pipeline = _build_pipeline_with_head()
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

        self.head_quat_world_xyzw = None
        self.last_buttons_left = None
        self.last_buttons_right = None
        if self._session is None:
            return None

        external_inputs = self._build_external_inputs()
        try:
            result = self._session.step(external_inputs=external_inputs)
        except ValueError as e:
            self._transient_error_count = getattr(self, "_transient_error_count", 0) + 1
            if self._transient_error_count == 1 or self._transient_error_count % 120 == 0:
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
                f"[bridge-teleop] TeleopSession recovered after {self._transient_error_count} transient failures",
                flush=True,
            )
            self._transient_error_count = 0

        if result is None or "action" not in result:
            return None

        self._update_head_pose(result)
        # Session output supports membership/indexing; do not assume dict.get.
        self.last_buttons_left = result["buttons_left"] if "buttons_left" in result else None  # noqa: SIM401
        self.last_buttons_right = result["buttons_right"] if "buttons_right" in result else None  # noqa: SIM401

        action_1d = torch.from_dlpack(result["action"][0]).to(dtype=torch.float32, device=self._env.device)

        # Per-wrist corrective rotation offsets (CLI-tunable).
        action_1d = _apply_wrist_quat_offsets(
            action_1d,
            self._left_quat_offset_xyzw,
            self._right_quat_offset_xyzw,
        )

        return action_1d.unsqueeze(0).repeat(self._env.num_envs, 1)

    def _update_head_pose(self, result) -> None:
        """Cache the latest world-frame headset orientation from the step result.

        Missing/invalid head output leaves this step's cache empty. The camera
        driver owns freezing the last view and re-seating on tracking recovery.
        """
        import numpy as np

        try:
            head_tg = result["head"] if "head" in result else None  # noqa: SIM401
            if head_tg is None or getattr(head_tg, "is_none", False):
                return
            from isaacteleop.retargeting_engine.tensor_types import HeadPoseIndex

            # BoolType slots come back as a plain Python bool (verified);
            # handle a dlpack tensor too in case that changes.
            valid_slot = head_tg[HeadPoseIndex.IS_VALID]
            if isinstance(valid_slot, (bool, int, np.bool_)):
                valid = bool(valid_slot)
            else:
                try:
                    valid = bool(np.asarray(np.from_dlpack(valid_slot)))
                except Exception:  # noqa: BLE001 - unreadable validity fails closed
                    valid = False
            if not valid:
                return
            quat = np.from_dlpack(head_tg[HeadPoseIndex.ORIENTATION])
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            if np.dot(quat, quat) < 1e-9 or not np.all(np.isfinite(quat)):
                return
            self.head_quat_world_xyzw = quat
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_head_pose_warned", False):
                self._head_pose_warned = True
                print(
                    f"[bridge-teleop] head pose extraction failed ({e!r}); XR head camera will stay torso-bound",
                    flush=True,
                )

    def close(self) -> None:
        if self._session_ctx is not None:
            try:
                self._session_ctx.__exit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                print(f"[bridge-teleop] error closing session: {e!r}", flush=True)
        self._session = None
        self._session_ctx = None
        self.head_quat_world_xyzw = None
        self.last_buttons_left = None
        self.last_buttons_right = None

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
            # Feed the same matrix to the upstream pipeline's "world_T_anchor"
            # leaf and to our head branch's "world_T_anchor_head" leaf (see
            # _build_pipeline_with_head).
            if leaf_name.startswith("world_T_anchor"):
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
        self._initial_quat_xyzw = (r_parent * R.from_quat(offset_rot_xyzw)).as_quat()  # xyzw

        # Initial forward in world = camera body-frame +X rotated to world.
        forward_world = R.from_quat(self._initial_quat_xyzw).as_matrix() @ np.array([1.0, 0.0, 0.0])
        forward_world[2] = 0.0  # project to horizontal
        n = np.linalg.norm(forward_world)
        self._initial_forward_horiz = forward_world / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

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
        new_quat_xyzw = (R.from_quat(delta_q) * R.from_quat(self._initial_quat_xyzw)).as_quat()

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
        _sync_right_camera(self._env, self._cam, new_pos_w, new_quat_xyzw)
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
        self._initial_quat_xyzw = (r_parent * R.from_quat(offset_rot_xyzw)).as_quat()

        forward_world = R.from_quat(self._initial_quat_xyzw).as_matrix() @ np.array([1.0, 0.0, 0.0])
        forward_world[2] = 0.0
        n = np.linalg.norm(forward_world)
        self._initial_forward_horiz = forward_world / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
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
        new_quat_xyzw = (yaw_q * R.from_quat(self._initial_quat_xyzw) * pitch_q).as_quat()
        new_pos_w = (self._initial_pos_w + pos_delta).astype(np.float64)  # type: ignore[operator]

        device = self._env.device
        pos_t = torch.tensor(new_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
        quat_t = torch.tensor(new_quat_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
        self._cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="world")
        _sync_right_camera(self._env, self._cam, new_pos_w, new_quat_xyzw)
        # Published pose is read back from the moved camera prim post-render,
        # so it tracks this move without a separate publisher override.

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._listener.stop()


# ---------------------------------------------------------------------------
# Pose diagnostics (--pose_diag)
# ---------------------------------------------------------------------------


class _PoseDiag:
    """Per-step camera/torso motion statistics (--pose_diag).

    Separates the two possible sources of image motion:
    * ``cam_in_parent``: the camera's pose RELATIVE to its parent (torso).
      Must be ~0 when nothing drives the camera — any nonzero delta is a
      self-inflicted pose write (a bug).
    * ``parent_world``: the torso's world motion per step — the balance
      policy's physical sway, which a torso-mounted camera legitimately sees.
    """

    def __init__(self, bridge_camera, window: int = 60) -> None:
        self._cam = bridge_camera
        self._window = window
        self._prev_rel = None
        self._prev_parent = None
        self._rel_dpos = []
        self._rel_drot = []
        self._par_dpos = []
        self._par_drot = []

    def step(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        cam_path = self._cam.cfg.prim_path.replace("env_.*", "env_0")
        cam_pose = self._cam._read_prim_world_pose_from_fabric(cam_path)
        parent_pose = self._cam._read_parent_world_pose_from_fabric()
        if cam_pose is None or parent_pose is None:
            return
        cam_pos, cam_quat = cam_pose
        par_pos, par_quat = parent_pose
        r_par = R.from_quat(par_quat)
        rel_pos = r_par.inv().apply(np.asarray(cam_pos) - np.asarray(par_pos))
        rel_rot = r_par.inv() * R.from_quat(cam_quat)

        if self._prev_rel is not None:
            p0, r0 = self._prev_rel
            self._rel_dpos.append(float(np.linalg.norm(rel_pos - p0)) * 1000.0)
            self._rel_drot.append(float(np.degrees((r0.inv() * rel_rot).magnitude())))
            pp0, pr0 = self._prev_parent
            self._par_dpos.append(float(np.linalg.norm(np.asarray(par_pos) - pp0)) * 1000.0)
            self._par_drot.append(float(np.degrees((pr0.inv() * r_par).magnitude())))
        self._prev_rel = (rel_pos, rel_rot)
        self._prev_parent = (np.asarray(par_pos), r_par)

        if len(self._rel_dpos) >= self._window:
            import numpy as np

            def s(v):
                return f"mean={np.mean(v):.3f} max={np.max(v):.3f}"

            print(
                f"[pose-diag] per-step deltas over {len(self._rel_dpos)} steps:"
                f" cam_in_parent pos[mm] {s(self._rel_dpos)}"
                f" rot[deg] {s(self._rel_drot)}"
                f" | parent_world pos[mm] {s(self._par_dpos)}"
                f" rot[deg] {s(self._par_drot)}",
                flush=True,
            )
            self._rel_dpos.clear()
            self._rel_drot.clear()
            self._par_dpos.clear()
            self._par_drot.clear()


# ---------------------------------------------------------------------------
# Stabilized (gimbal) camera mount — --camera_stabilize_s > 0
# ---------------------------------------------------------------------------


class _StabilizedCameraDriver:
    """Drive WORLD-parented bridge cameras as a stabilized virtual gimbal.

    Why: the G1 balance policy micro-sways the torso ~0.2 deg / ~0.6 mm per
    step (measured with --pose_diag). A camera parented under the torso
    inherits that sway rigidly — ~3 px of image jitter per frame at 1280 px.
    Reparenting the cameras to the world (done in main() when
    --camera_stabilize_s > 0) removes the scene-graph coupling entirely; this
    driver then follows the torso mount point through a first-order low-pass
    (time constant ``tau_s``), so the camera tracks the robot's real motion
    but not its dither.

    Rotation source: the XR head orientation when the engage gate is ENGAGED
    (same convention chain as _XrHeadCameraController), otherwise the smoothed
    torso-mount orientation. The right eye rides at -baseline along the
    smoothed body +Y. Published GT pose stays exact — it is read back from
    the camera prim after each render.
    """

    def __init__(
        self,
        env,
        left_cam,
        right_cam,
        teleop,
        engage_gate,
        tau_s: float,
        head_enabled: bool = True,
    ) -> None:
        import numpy as np

        self._env = env
        self._cam = left_cam
        self._right_cam = right_cam
        self._teleop = teleop
        self._gate = engage_gate
        self._tau = float(tau_s)
        self._head_enabled = head_enabled
        self._held_head_quat = None
        self._head_rotation_offset = None
        self._head_tracking_lost = False
        self._torso_path = "/World/envs/env_0/Robot/torso_link"
        self._offset_pos = np.array(list(left_cam.cfg.offset.pos), dtype=np.float64)
        self._offset_rot = np.array(list(left_cam.cfg.offset.rot), dtype=np.float64)
        self._baseline = float(getattr(left_cam.cfg, "stereo_baseline_m", 0.0))
        self._state_pos = None  # smoothed camera world position
        self._state_quat = None  # smoothed camera world body-quat (xyzw)
        self._last_t = None
        self._logged_head = False

    def step(self) -> None:
        import math
        import time as _time

        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R

        torso = self._cam._read_prim_world_pose_from_fabric(self._torso_path)
        if torso is None:
            return
        torso_pos, torso_quat = torso
        r_torso = R.from_quat(torso_quat)
        target_pos = np.asarray(torso_pos) + r_torso.as_matrix() @ self._offset_pos

        head_quat = getattr(self._teleop, "head_quat_world_xyzw", None)
        engaged = self._head_enabled and (self._gate is None or self._gate.engaged)
        if engaged and head_quat is not None:
            if self._head_tracking_lost:
                self._head_rotation_offset = None
            self._head_tracking_lost = False
            self._held_head_quat = head_quat.copy()
        elif engaged:
            self._head_tracking_lost = True
        elif not engaged:
            self._held_head_quat = None
            self._head_rotation_offset = None
        head_quat = self._held_head_quat
        head_active = engaged and head_quat is not None
        if head_active:
            from bridge_camera import _R_USDCAM_TO_BODY

            head_body = R.from_quat(head_quat) * R.from_matrix(_R_USDCAM_TO_BODY)
            if self._head_rotation_offset is None:
                # The operator looks forward at engage: seat that head pose
                # at the current camera orientation without an absolute snap.
                reference = self._state_quat
                if reference is None:
                    fwd = r_torso.as_matrix()[:, 0]
                    reference = (
                        R.from_euler("z", math.atan2(fwd[1], fwd[0])) * R.from_quat(self._offset_rot)
                    ).as_quat()
                self._head_rotation_offset = R.from_quat(reference) * head_body.inv()
            target_quat = (self._head_rotation_offset * head_body).as_quat()
            if not self._logged_head:
                self._logged_head = True
                print(
                    "[bridge-teleop] stabilized camera: rotation now follows the XR headset",
                    flush=True,
                )
        else:
            # Gimbal leveling: what shakes the IMAGE is rotation (position
            # jitter of ~2 mm is sub-pixel at scene distance), and the sway
            # is a ~1 Hz oscillation a practical low-pass barely attenuates.
            # So kill roll/pitch OUTRIGHT — follow only the torso's YAW
            # (smoothed below) and compose the cfg mount rotation (which
            # carries any --camera_pitch_down_deg). The camera horizon is
            # level by construction, exactly like a real stabilized gimbal.
            import math as _math

            fwd = r_torso.as_matrix() @ np.array([1.0, 0.0, 0.0])
            yaw = _math.atan2(fwd[1], fwd[0])
            target_quat = (R.from_euler("z", yaw) * R.from_quat(self._offset_rot)).as_quat()

        now = _time.monotonic()
        if self._state_pos is None:
            self._state_pos = target_pos
            self._state_quat = np.asarray(target_quat, dtype=np.float64)
        else:
            dt = max(1e-4, now - (self._last_t or now))
            alpha = 1.0 - math.exp(-dt / max(self._tau, 1e-3))
            # Head rotation is a COMMAND, not sway — follow it unfiltered so
            # look-around stays latency-free; smooth only the torso follow.
            rot_alpha = 1.0 if head_active else alpha
            self._state_pos = self._state_pos + alpha * (target_pos - self._state_pos)
            q_t = np.asarray(target_quat, dtype=np.float64)
            if float(np.dot(q_t, self._state_quat)) < 0.0:
                q_t = -q_t  # shortest-arc nlerp
            q = self._state_quat + rot_alpha * (q_t - self._state_quat)
            self._state_quat = q / max(1e-9, float(np.linalg.norm(q)))
        self._last_t = now

        device = self._env.device
        pos_t = torch.tensor(self._state_pos, dtype=torch.float32, device=device).unsqueeze(0)
        quat_t = torch.tensor(self._state_quat, dtype=torch.float32, device=device).unsqueeze(0)
        self._cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="world")
        if self._right_cam is not None:
            right_pos = self._state_pos + R.from_quat(self._state_quat).as_matrix() @ np.array(
                [0.0, -self._baseline, 0.0]
            )
            rpos_t = torch.tensor(right_pos, dtype=torch.float32, device=device).unsqueeze(0)
            self._right_cam.set_world_poses(positions=rpos_t, orientations=quat_t, convention="world")


# ---------------------------------------------------------------------------
# XR head camera: rotation from the headset, translation bound to the torso
# ---------------------------------------------------------------------------


class _XrHeadCameraController:
    """Drive the bridge camera(s) as a rotation-only virtual head.

    * **Translation** is never touched: the cameras stay parented under the
      torso at their cfg offsets, so the scene graph binds them rigidly with
      zero lag. (An earlier version re-set the full world pose every step
      from a one-render-stale Fabric read of the parent — that
      double-differences the balance policy's micro-motion and makes the
      rendered image jitter. Do not reintroduce it.)
    * **Rotation** comes from the raw XR headset orientation (already mapped
      into the Isaac world frame by ``world_T_anchor`` inside the teleop
      pipeline), converted from the OpenXR view-frame convention (-Z forward,
      +Y up) to the world/ROS body convention (+X forward, +Y left, +Z up)
      that ``Camera.set_world_poses(convention="world")`` expects, and set
      with ``positions=None`` (orientation-only). Until the first valid head
      pose arrives, the controller does NOTHING — the plain parented mount
      (including any --camera_pitch offset) is already correct.

    The right-eye camera (if present) is placed rigidly at ``-baseline`` along
    the head's body +Y each step, so the stereo pair rotates as one unit.
    The published ``T_cam_world`` needs no extra plumbing: it is read back
    from the moved camera prim's Fabric matrix after each render, so the GT
    pose always matches the XR-driven render pose.
    """

    def __init__(self, bridge_camera, env, teleop) -> None:
        self._cam = bridge_camera
        self._env = env
        self._teleop = teleop
        self._logged_active = False

    def step(self) -> None:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R

        head_quat = getattr(self._teleop, "head_quat_world_xyzw", None)
        if head_quat is None:
            # No headset fix yet: DO NOTHING. The cameras are parented under
            # the torso and follow it rigidly through the scene graph — that
            # IS the torso binding. Re-setting the world pose every step from
            # a one-render-stale Fabric read of the parent double-differences
            # the balance policy's micro-motion and makes the image jitter
            # (observed live), so the fallback must not touch the prims.
            return

        from bridge_camera import _R_USDCAM_TO_BODY

        # World-frame OpenXR view orientation -> world/ROS body convention.
        quat_xyzw = (R.from_quat(head_quat) * R.from_matrix(_R_USDCAM_TO_BODY)).as_quat()
        if not self._logged_active:
            self._logged_active = True
            print(
                "[bridge-teleop] XR head camera active: first headset pose"
                " received, camera rotation now follows the headset",
                flush=True,
            )

        device = self._env.device
        quat_t = torch.tensor(quat_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
        # Rotation-only virtual head: set ONLY the orientation. positions=None
        # leaves the camera's parented local offset untouched, so translation
        # stays rigidly bound to the torso with zero re-set lag/jitter.
        self._cam.set_world_poses(orientations=quat_t, convention="world")

        # The right eye's POSITION does depend on the head rotation (it orbits
        # the left eye at -baseline along the head's +Y), so it needs a world
        # position; derive the left eye position from the parent Fabric pose +
        # cfg offset (one consistent snapshot; sub-mm staleness vs the 63 mm
        # baseline).
        parent_pose = self._cam._read_parent_world_pose_from_fabric()
        if parent_pose is None:
            return
        parent_pos_w, parent_quat_xyzw = parent_pose
        offset_pos = np.array(list(self._cam.cfg.offset.pos), dtype=np.float64)
        left_pos_w = parent_pos_w + R.from_quat(parent_quat_xyzw).as_matrix() @ offset_pos
        _sync_right_camera(self._env, self._cam, left_pos_w, quat_xyzw)


# ---------------------------------------------------------------------------
# Teleop engage gate: hold until engaged, clutch on engage, guard bad inputs
# ---------------------------------------------------------------------------


def _tg_scalar(group, index) -> float:
    """Read one scalar slot from a ControllerInput TensorGroup.

    Slots may come back as plain Python numbers/bools or as dlpack-capable
    0-d tensors depending on the type (BoolType returns bool, FloatType may
    return either); normalize to float.
    """
    import numpy as np

    value = group[index]
    if isinstance(value, (bool, int, float)):
        return float(value)
    try:
        return float(np.asarray(np.from_dlpack(value)))
    except Exception:  # noqa: BLE001 — unreadable slot reads as released
        return 0.0


class _TeleopEngageGate:
    """Explicit engage/hold state machine around the retargeted action.

    Fixes three failure modes of the bare (ungated) pipeline:

    1. **No engagement**: the stock path commands the robot the moment the
       XR session connects. Here the robot HOLDS its current pose until the
       engage button (right controller, ``--engage_button``) is pressed;
       pressing it again disengages and re-captures the hold pose at the
       robot's CURRENT configuration (no snap back to the startup pose).
    2. **Engage jump**: ``Se3AbsRetargeter`` is absolute with no clutch — at
       engage the wrist target snaps to wherever your physical hand maps
       into the anchored frame. On every engage / reset / tracking recovery
       this gate captures ``robot_wrist - target`` and composes it onto all
       subsequent targets, so the first commanded pose IS the robot's
       current hand pose and your motion is applied as deltas from there
       (``--engage_clutch pos|full|off``).
    3. **Weird targets**: when a controller is untracked the retargeter
       emits its last pose — initialized at the WORLD ORIGIN — which yanks
       both arms toward the pelvis. The gate reads grip validity from the
       parallel button tap and auto-holds while either controller is
       untracked (re-clutching on recovery). It also recomputes the
       locomotion command from raw thumbsticks with a deadzone, and owns the
       hip-height integration so stick drift can't walk or crouch the robot
       (the stock retargeter integrates hip drift even while idle).

    ``process(raw_action)`` never returns None — it substitutes the held
    pose whenever the pipeline output must not reach the robot.
    """

    # Action layout: [l_pos(0:3), l_quat(3:7), r_pos(7:10), r_quat(10:14),
    #                 hands(14:28), locomotion(28:32)].
    _LOCO = slice(28, 32)

    # Locomotion constants — mirror LocomotionRootCmdRetargeterConfig in
    # _build_g1_locomanipulation_pipeline (initial_hip_height 0.72,
    # movement_scale 0.5, rotation_scale 0.35, dt 1/100) and the stock
    # retargeter's [0.4, 1.0] hip clamp.
    _HIP_INITIAL = 0.72
    _HIP_RANGE = (0.4, 1.0)
    _MOVEMENT_SCALE = 0.5
    _HIP_RATE = 0.35
    _DT = 1.0 / 100.0

    def __init__(
        self,
        env,
        teleop,
        initial_hold_action,
        engage_chord: str = "x+y+a+b",
        reset_chord: str = "lsqueeze+a",
        clutch_mode: str = "full",
        stick_deadzone: float = 0.15,
    ) -> None:
        from isaacteleop.retargeting_engine.tensor_types import ControllerInputIndex

        self._idx = ControllerInputIndex
        self._env = env
        self._teleop = teleop
        self._hold_action = initial_hold_action
        self._last_hand_targets = initial_hold_action[:, 14:28].clone()
        self._engage_chord = self._parse_chord(engage_chord)
        self._reset_chord = self._parse_chord(reset_chord)
        self._clutch_mode = clutch_mode
        self._deadzone = float(stick_deadzone)
        self.engaged = False
        self._need_clutch = True
        self._hip = self._HIP_INITIAL
        # Init True: a chord already held when the session connects must
        # not read as a rising edge (DefaultTeleopStateManager's fail-safe).
        self._prev_engage = True
        self._prev_reset = True
        self._warned_untracked = False
        self._input_lost = False
        # pos clutch: torch offset tensors; full clutch: scipy anchors.
        self._clutch_l = None
        self._clutch_r = None
        print(
            f"[bridge-teleop] engage gate armed: [{engage_chord}] ="
            f" engage/hold toggle, [{reset_chord}] = re-clutch/reset,"
            f" clutch={clutch_mode}, stick deadzone={stick_deadzone}",
            flush=True,
        )

    def _parse_chord(self, spec: str):
        """'+'-separated chord tokens -> list of (side, ControllerInputIndex).

        Tokens: x/y = left primary/secondary, a/b = right primary/secondary,
        [lr]squeeze, [lr]trigger, [lr]menu, [lr]stick.
        """
        idx = self._idx
        token_map = {
            "x": ("left", idx.PRIMARY_CLICK),
            "y": ("left", idx.SECONDARY_CLICK),
            "a": ("right", idx.PRIMARY_CLICK),
            "b": ("right", idx.SECONDARY_CLICK),
            "lsqueeze": ("left", idx.SQUEEZE_VALUE),
            "rsqueeze": ("right", idx.SQUEEZE_VALUE),
            "ltrigger": ("left", idx.TRIGGER_VALUE),
            "rtrigger": ("right", idx.TRIGGER_VALUE),
            "lmenu": ("left", idx.MENU_CLICK),
            "rmenu": ("right", idx.MENU_CLICK),
            "lstick": ("left", idx.THUMBSTICK_CLICK),
            "rstick": ("right", idx.THUMBSTICK_CLICK),
        }
        chord = []
        for token in spec.replace(",", "+").split("+"):
            token = token.strip().lower()
            if not token:
                continue
            if token not in token_map:
                raise ValueError(f"unknown chord token {token!r}; valid: {sorted(token_map)}")
            chord.append(token_map[token])
        if not chord:
            raise ValueError(f"empty chord spec {spec!r}")
        return chord

    # -- input reading -------------------------------------------------------

    @staticmethod
    def _group_present(group) -> bool:
        return group is not None and not getattr(group, "is_none", False)

    def _tracked(self, group) -> bool:
        if not self._group_present(group):
            return False
        return _tg_scalar(group, self._idx.GRIP_IS_VALID) > 0.5

    def _chord_active(self, chord, left_group, right_group):
        """True when ALL chord buttons read pressed; None when any needed
        controller group is absent this frame (indeterminate)."""
        for side, button_idx in chord:
            group = left_group if side == "left" else right_group
            if not self._group_present(group):
                return None
            if _tg_scalar(group, button_idx) <= 0.5:
                return False
        return True

    def _edges(self, left_group, right_group):
        """Rising edges of the engage/reset chords (all buttons together)."""
        edges = []
        for chord, prev_attr in (
            (self._engage_chord, "_prev_engage"),
            (self._reset_chord, "_prev_reset"),
        ):
            active = self._chord_active(chord, left_group, right_group)
            if active is None:
                # Controller vanished mid-frame: hold prev=True so a chord
                # still held when it reappears can't fire a synthetic edge.
                setattr(self, prev_attr, True)
                edges.append(False)
                continue
            edge = active and not getattr(self, prev_attr)
            setattr(self, prev_attr, active)
            edges.append(edge)
        engage_edge, reset_edge = edges
        if engage_edge:
            # The chords can overlap (e.g. right A in both): an engage
            # activation wins outright that frame.
            reset_edge = False
        return engage_edge, reset_edge

    def _dz(self, value: float) -> float:
        """Deadzone with rescaling above it (continuous at the edge)."""
        a = abs(value)
        if a < self._deadzone:
            return 0.0
        return (1.0 if value > 0 else -1.0) * (a - self._deadzone) / (1.0 - self._deadzone)

    # -- hold pose -----------------------------------------------------------

    def _refresh_hold(self) -> None:
        new_hold = _build_hold_pose_action(self._env)
        if new_hold is not None:
            new_hold[:, 14:28] = self._last_hand_targets
            new_hold[:, 31] = self._hip
            self._hold_action = new_hold

    def _hold(self):
        return self._hold_action

    # -- clutch --------------------------------------------------------------

    def _read_wrist_world(self):
        """Current (l_pos, l_quat_xyzw, r_pos, r_quat_xyzw) in the action's
        frame (env-origin frame, xyzw quats — see _build_hold_pose_action's
        convention notes), or None."""
        import numpy as np

        robot = self._env.scene["robot"]
        body_names = robot.body_names
        li = _find_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_link"))
        ri = _find_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_link"))
        if li is None or ri is None:
            return None
        origin = self._env.scene.env_origins[0]
        pos = robot.data.body_pos_w[0]
        quat_xyzw = robot.data.body_quat_w[0]  # already (x, y, z, w)
        return (
            (pos[li] - origin).detach().cpu().numpy().astype(np.float64),
            quat_xyzw[li].detach().cpu().numpy().astype(np.float64),
            (pos[ri] - origin).detach().cpu().numpy().astype(np.float64),
            quat_xyzw[ri].detach().cpu().numpy().astype(np.float64),
        )

    def _capture_clutch(self, action_1d) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        wrists = self._read_wrist_world()
        if wrists is None:
            print(
                "[bridge-teleop] engage gate: wrist links not found — clutch DISABLED (raw absolute targets)",
                flush=True,
            )
            self._clutch_mode = "off"
            return
        l_pos, l_quat, r_pos, r_quat = wrists
        a = action_1d.detach().cpu().numpy().astype(np.float64)
        clutches = []
        for robot_pos, robot_quat, tp, tq in (
            (l_pos, l_quat, a[0:3], a[3:7]),
            (r_pos, r_quat, a[7:10], a[10:14]),
        ):
            if self._clutch_mode == "pos":
                clutches.append({"dpos": robot_pos - tp})
            else:  # full SE(3)
                r_off = R.from_quat(robot_quat) * R.from_quat(tq).inv()
                clutches.append({"r_off": r_off, "anchor_t": tp.copy(), "anchor_r": robot_pos})
        self._clutch_l, self._clutch_r = clutches
        jump_l = float(np.linalg.norm(l_pos - a[0:3]))
        jump_r = float(np.linalg.norm(r_pos - a[7:10]))
        print(
            f"[bridge-teleop] clutch captured ({self._clutch_mode}): absorbed"
            f" offsets L={jump_l * 100:.1f}cm R={jump_r * 100:.1f}cm",
            flush=True,
        )

    def _apply_clutch(self, action_1d):
        if self._clutch_mode == "off" or self._clutch_l is None:
            return action_1d
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation as R

        a = action_1d.detach().cpu().numpy().astype(np.float64)
        for clutch, ps, qs in (
            (self._clutch_l, slice(0, 3), slice(3, 7)),
            (self._clutch_r, slice(7, 10), slice(10, 14)),
        ):
            if self._clutch_mode == "pos":
                a[ps] = a[ps] + clutch["dpos"]
            else:
                r_off = clutch["r_off"]
                a[ps] = clutch["anchor_r"] + r_off.apply(a[ps] - clutch["anchor_t"])
                a[qs] = (r_off * R.from_quat(a[qs])).as_quat()
        return torch.from_numpy(a).to(action_1d.device, action_1d.dtype)

    # -- locomotion ------------------------------------------------------------

    def _apply_locomotion(self, action_1d, left_group, right_group):
        """Recompute [vel_x, vel_y, rot_vel_z, hip] from raw sticks with a
        deadzone; integrate hip height HERE (only while engaged) using the
        stock retargeter's constants and clamp."""
        lx = ly = rx = ry = 0.0
        if self._group_present(left_group):
            lx = self._dz(_tg_scalar(left_group, self._idx.THUMBSTICK_X))
            ly = self._dz(_tg_scalar(left_group, self._idx.THUMBSTICK_Y))
        if self._group_present(right_group):
            rx = self._dz(_tg_scalar(right_group, self._idx.THUMBSTICK_X))
            ry = self._dz(_tg_scalar(right_group, self._idx.THUMBSTICK_Y))
        self._hip += ry * self._DT * self._HIP_RATE
        self._hip = max(self._HIP_RANGE[0], min(self._HIP_RANGE[1], self._hip))
        # Same mapping/signs as LocomotionRootCmdRetargeter._compute_fn.
        action_1d[self._LOCO] = action_1d.new_tensor(
            [ly * self._MOVEMENT_SCALE, -lx * self._MOVEMENT_SCALE, -rx, self._hip]
        )
        return action_1d

    # -- main entry ------------------------------------------------------------

    def process(self, raw_action):
        """Gate one step. ``raw_action`` is the (num_envs, 32) pipeline action
        or None (session down / transient error). ALWAYS returns a valid
        action tensor — the held pose whenever the pipeline output must not
        reach the robot."""
        import torch

        left = self._teleop.last_buttons_left
        right = self._teleop.last_buttons_right
        engage_edge, reset_edge = self._edges(left, right)
        tracked = self._tracked(left) and self._tracked(right)
        valid_action = raw_action is not None
        if valid_action:
            valid_action = bool(torch.isfinite(raw_action).all()) and all(
                bool(torch.linalg.vector_norm(raw_action[:, qs], dim=-1).gt(1e-6).all())
                for qs in (slice(3, 7), slice(10, 14))
            )
        input_ready = valid_action and tracked
        if self.engaged and not input_ready:
            if not self._input_lost:
                self._refresh_hold()
            self._input_lost = True
            self._need_clutch = True
        elif input_ready:
            self._input_lost = False

        if reset_edge:
            if self.engaged:
                self._need_clutch = True
                self._hip = self._HIP_INITIAL
                print(
                    "[bridge-teleop] RESET: re-clutching wrist targets to the"
                    " robot's current hand poses; hip height reset",
                    flush=True,
                )
            else:
                self._refresh_hold()
                print("[bridge-teleop] RESET (held): hold pose re-captured", flush=True)

        if engage_edge:
            if self.engaged:
                self.engaged = False
                self._refresh_hold()
                print("[bridge-teleop] DISENGAGED: holding current pose", flush=True)
            elif not input_ready:
                print(
                    "[bridge-teleop] engage pressed but controllers are not"
                    " tracked — staying held (try again with both controllers"
                    " visible)",
                    flush=True,
                )
            else:
                self.engaged = True
                self._need_clutch = True
                self._hip = self._HIP_INITIAL
                print("[bridge-teleop] ENGAGED", flush=True)

        if not self.engaged or not valid_action:
            return self._hold()

        if not tracked:
            if not self._warned_untracked:
                self._warned_untracked = True
                print(
                    "[bridge-teleop] controllers untracked — HOLDING; will re-clutch when tracking resumes (no jump)",
                    flush=True,
                )
            self._need_clutch = True  # hand may move while untracked
            return self._hold()
        if self._warned_untracked:
            self._warned_untracked = False
            print("[bridge-teleop] tracking recovered — re-clutching", flush=True)

        action_1d = raw_action[0].clone()
        if self._need_clutch:
            if self._clutch_mode != "off":
                self._capture_clutch(action_1d)
            self._need_clutch = False
        action_1d = self._apply_clutch(action_1d)
        action_1d = self._apply_locomotion(action_1d, left, right)
        self._last_hand_targets = action_1d[14:28].unsqueeze(0).clone()
        return action_1d.unsqueeze(0).repeat(raw_action.shape[0], 1)


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


def _get_bridge_camera_right(env):
    scene = env.scene
    try:
        return scene["bridge_camera_right"]
    except (KeyError, TypeError):
        pass
    return getattr(scene, "bridge_camera_right", None)


def _sync_right_camera(env, left_cam, left_pos_w, left_quat_xyzw) -> None:
    """Place the right-eye camera rigidly relative to the (just-moved) left eye.

    Right eye = ``-stereo_baseline_m`` along the left camera's body +Y (left)
    axis, same orientation. No-op when the scene has no right camera. Must be
    called by every controller that ``set_world_poses``-moves the left camera,
    otherwise the pair de-rigidifies (the right eye would stay on its static
    torso mount).
    """
    right_cam = _get_bridge_camera_right(env)
    if right_cam is None:
        return
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R

    baseline = float(getattr(left_cam.cfg, "stereo_baseline_m", 0.0))
    quat = np.asarray(left_quat_xyzw, dtype=np.float64).reshape(4)
    right_pos = np.asarray(left_pos_w, dtype=np.float64) + R.from_quat(quat).as_matrix() @ np.array(
        [0.0, -baseline, 0.0]
    )

    device = env.device
    pos_t = torch.tensor(right_pos, dtype=torch.float32, device=device).unsqueeze(0)
    quat_t = torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0)
    right_cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="world")


def _build_hold_pose_action(env):
    """Keep the wrists at their current poses. Used while XR is idle/held.

    Conventions (both bit us before — see INTEGRATION_NOTES):
    * ``body_quat_w`` is ALREADY (x, y, z, w) on this Isaac Lab branch
      (repo-wide xyzw convention) — exactly what the Pink IK action expects.
      Do NOT reorder it: the old wxyz->xyzw swizzle scrambled the wrist
      orientation targets and made the IK crank both arms sideways.
    * The action's positions live in the ENV-ORIGIN frame
      (``PinkInverseKinematicsAction`` subtracts ``env_origins`` from the
      base pose only), so subtract the env origins from the world positions.
    * Locomotion is [vel_x, vel_y, rot_vel_z, hip_height]: hip height is an
      ABSOLUTE command clamped to [0.4, 1.0] downstream — a zero would order
      a full crouch and destabilize the balance policy. Hold at 0.72 (the
      pipeline's initial_hip_height).
    """
    import torch

    robot = env.scene["robot"]
    body_names = robot.body_names
    left_idx = _find_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_link"))
    right_idx = _find_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_link"))
    if left_idx is None or right_idx is None:
        return None

    pos = robot.data.body_pos_w
    quat_xyzw = robot.data.body_quat_w  # already (x, y, z, w) on this branch
    env_origins = env.scene.env_origins
    left_pos = pos[:, left_idx] - env_origins
    right_pos = pos[:, right_idx] - env_origins
    left_q = quat_xyzw[:, left_idx]
    right_q = quat_xyzw[:, right_idx]

    num_envs = pos.shape[0]
    hand_joints = torch.zeros((num_envs, 14), device=env.device)
    locomotion = torch.zeros((num_envs, 4), device=env.device)
    locomotion[:, 3] = 0.72  # hold the nominal hip height, never 0
    action = torch.cat([left_pos, left_q, right_pos, right_q, hand_joints, locomotion], dim=-1)
    return action.detach().clone()


def _find_body_index(body_names, candidates):
    for name in candidates:
        if name in body_names:
            return body_names.index(name)
    return None


# NOTE: there is deliberately no wxyz<->xyzw helper here anymore. This Isaac
# Lab branch uses the xyzw quaternion convention REPO-WIDE (body_quat_w,
# matrix_from_quat, the Pink IK action) — a leftover wxyz->xyzw swizzle on
# body_quat_w scrambled the hold-pose wrist orientations and contorted the
# robot. If you need a conversion, you are probably misreading a convention.


def _parse_rpy(s: str) -> tuple[float, float, float]:
    """Parse a ``"roll,pitch,yaw"`` comma-separated string of degrees."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 'roll,pitch,yaw', got {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _rpy_deg_to_quat_xyzw(rpy_deg: tuple[float, float, float]) -> np.ndarray | None:
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
