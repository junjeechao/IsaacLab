# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless runner: G1 locomanipulation env streaming frames to the Holoscan bridge.

Typical usage (from the IsaacLab repo root, with the realtime-3d-reconstruction
container already running ``isaaclab_3d_recon`` on the same host)::

    ./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge.py --enable_cameras

``--enable_cameras`` is required because the bridge attaches a camera sensor
to the scene. Headless is the default; pass ``--viz kit`` to bring up the
Kit viewport instead. Run with ``--help`` to see the full argument list
inherited from :class:`isaaclab.app.AppLauncher`.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Make sibling bridge modules importable both when this script is run via
# ``./isaaclab.sh -p`` and when imported by Isaac Lab's class_type loader.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher  # imported here so --help works pre-launch

    parser = argparse.ArgumentParser(
        description="Stream G1 head-camera frames to the realtime-3d-reconstruction app."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments. Only view 0 is published.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, exit after this many env steps. Default 0 means run until interrupted.",
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
        default=30,
        help="Print a status line every N env steps.",
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

    print(f"[bridge-runner] building env (num_envs={args.num_envs})", flush=True)
    env_cfg = LocomanipulationG1BridgeEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.bridge_endpoint is not None:
        env_cfg.scene.bridge_camera.endpoint = args.bridge_endpoint

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(
        f"[bridge-runner] env built; action_dim={env.action_manager.total_action_dim}"
        f" device={env.device}",
        flush=True,
    )

    bridge_cam_probe = _get_bridge_camera(env)
    if bridge_cam_probe is None:
        print("[bridge-runner] WARNING: env.scene['bridge_camera'] not found", flush=True)
    else:
        print(
            f"[bridge-runner] bridge_camera type={type(bridge_cam_probe).__name__}"
            f" module={type(bridge_cam_probe).__module__}",
            flush=True,
        )

    try:
        # AgileBasedLowerBodyAction internally runs a TorchScript locomotion
        # policy whose outputs carry ``requires_grad=True``. Warp's kernel
        # launcher rejects grad-tracking tensors via ``__cuda_array_interface__``.
        # Wrapping the whole sim loop in ``torch.no_grad()`` keeps the policy
        # forward pass and every downstream tensor out of the autograd graph.
        torch_no_grad = torch.no_grad()
        torch_no_grad.__enter__()
        no_grad_active = True

        env.reset()
        print("[bridge-runner] env.reset() complete", flush=True)

        hold_action = _build_hold_pose_action(env)
        if hold_action is None:
            print(
                "[bridge-runner] could not build hold-pose action; falling back to zeros",
                flush=True,
            )
            hold_action = torch.zeros(
                (args.num_envs, env.action_manager.total_action_dim), device=env.device
            )

        step = 0
        print("[bridge-runner] entering step loop", flush=True)
        while simulation_app.is_running():
            try:
                env.step(hold_action)
            except Exception:  # noqa: BLE001 — surface the trace, do not swallow
                print("[bridge-runner] env.step() raised; aborting", flush=True)
                traceback.print_exc()
                break

            step += 1
            if args.heartbeat_every > 0 and step % args.heartbeat_every == 0:
                print(f"[bridge-runner] step={step}", flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                print(f"[bridge-runner] max_steps={args.max_steps} reached", flush=True)
                break

        if not simulation_app.is_running():
            print("[bridge-runner] simulation_app reported not running; exiting loop", flush=True)
        print(f"[bridge-runner] total steps={step}", flush=True)
    finally:
        if "no_grad_active" in locals() and no_grad_active:
            torch_no_grad.__exit__(None, None, None)
        bridge_cam = _get_bridge_camera(env)
        if bridge_cam is not None and hasattr(bridge_cam, "close"):
            print("[bridge-runner] closing bridge camera publisher", flush=True)
            bridge_cam.close()
        env.close()
        simulation_app.close()


def _get_bridge_camera(env):
    """Return the BridgeCamera sensor from the env scene, or None."""
    scene = env.scene
    # Scene supports dict-style lookup; sensors are usually under ``scene["name"]``.
    try:
        return scene["bridge_camera"]
    except (KeyError, TypeError):
        pass
    return getattr(scene, "bridge_camera", None)


def _build_hold_pose_action(env):
    """Build an action that keeps the wrists at their current world-frame poses.

    The env's ``upper_body_ik`` action expects a 28-D payload of
    ``[left_pos(3), left_quat_xyzw(4), right_pos(3), right_quat_xyzw(4), hand_joints(14)]``,
    and ``lower_body_joint_pos`` expects a 4-D locomotion command. Zero is fine
    for the locomotion command and the hand joints. For the wrist targets we
    read the current body-frame poses from the articulation so the IK has a
    reachable target and stops printing ``Solution to IK contains NaN``.

    Returns ``None`` if the wrist links cannot be located in the articulation.
    """
    import torch

    robot = env.scene["robot"]
    body_names = robot.body_names

    # Try the most common G1 wrist link names. Fall back to a substring search.
    left_idx = _find_body_index(body_names, ("left_wrist_yaw_link", "left_wrist_link"))
    right_idx = _find_body_index(body_names, ("right_wrist_yaw_link", "right_wrist_link"))
    if left_idx is None or right_idx is None:
        print(
            f"[bridge-runner] wrist link lookup failed; left={left_idx} right={right_idx}"
            f" body_names_sample={body_names[:8]}",
            flush=True,
        )
        return None

    pos = robot.data.body_pos_w  # (N, B, 3)
    quat_wxyz = robot.data.body_quat_w  # (N, B, 4) in (w, x, y, z)

    left_pos = pos[:, left_idx]
    right_pos = pos[:, right_idx]
    left_q = _wxyz_to_xyzw(quat_wxyz[:, left_idx])
    right_q = _wxyz_to_xyzw(quat_wxyz[:, right_idx])

    num_envs = pos.shape[0]
    hand_joints = torch.zeros((num_envs, 14), device=env.device)
    locomotion = torch.zeros((num_envs, 4), device=env.device)

    # body_pos_w/body_quat_w are torch views over Warp arrays and may have
    # requires_grad=True. Detach+clone so the action is a static snapshot
    # that won't propagate gradients into the downstream Warp kernels.
    action = torch.cat([left_pos, left_q, right_pos, right_q, hand_joints, locomotion], dim=-1)
    return action.detach().clone()


def _find_body_index(body_names: list[str], candidates: tuple[str, ...]) -> int | None:
    for name in candidates:
        if name in body_names:
            return body_names.index(name)
    return None


def _wxyz_to_xyzw(quat_wxyz):
    """Swap a ``(..., 4)`` quaternion from ``(w, x, y, z)`` to ``(x, y, z, w)``."""
    import torch

    return torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)


if __name__ == "__main__":
    main()
