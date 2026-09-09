# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomanipulation env extended with a bridge camera."""

from __future__ import annotations

from bridge_camera_cfg import BridgeCameraCfg

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
)


@configclass
class LocomanipulationG1BridgeSceneCfg(LocomanipulationG1SceneCfg):
    """Scene config that adds a bridge camera mounted on the G1's torso."""

    # Right eye of the stereo pair: RGB-only, render-only (never publishes;
    # the left camera pulls its RGB at publish time -- see
    # BridgeCameraCfg.stereo_role). Declared BEFORE ``bridge_camera`` so it
    # updates/renders first each step and the stereo pair is time-aligned.
    # Offset matches the left eye shifted by -stereo_baseline_m along the
    # torso's +Y (left) axis: right eye = -Y.
    bridge_camera_right: BridgeCameraCfg = BridgeCameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/bridge_camera_right",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 20.0),
        ),
        offset=BridgeCameraCfg.OffsetCfg(
            pos=(0.10, -0.063, 0.50),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
        stereo_role="right",
        debug_window=False,
    )

    # G1 has no ``head_link`` body; the head is a static visual under
    # ``torso_link``. Parent here and apply a +Z offset to reach head height.
    # ``update_latest_camera_pose=True`` is required for cameras under a
    # moving articulation -- otherwise CameraData.pos_w stays at scene-init.
    bridge_camera: BridgeCameraCfg = BridgeCameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/bridge_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 20.0),
        ),
        # Offset is in torso_link's local frame. pos.z lifts the camera from
        # the torso_link origin (~0.79 m world Z) to ~head height (~1.3 m).
        # ``convention="world"`` aligns camera forward with parent's +X.
        offset=BridgeCameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.50),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
        endpoint="tcp://127.0.0.1:5570",
        stereo_role="left",
        stereo_baseline_m=0.063,
    )


@configclass
class LocomanipulationG1BridgeEnvCfg(LocomanipulationG1EnvCfg):
    """G1 locomanipulation env that streams head-camera frames to the bridge."""

    scene: LocomanipulationG1BridgeSceneCfg = LocomanipulationG1BridgeSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep ``self.xr`` from the upstream env -- our ``_build_external_inputs``
        # reads ``cfg.xr.anchor_pos`` to match the ``--xr`` calibration.
        # Null ``isaac_teleop`` so Kit stays out of OpenXR; we bypass
        # IsaacTeleopDevice with our own standalone TeleopSession.
        self.isaac_teleop = None
