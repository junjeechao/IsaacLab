# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sensor that publishes RGB-D + intrinsics + pose over the bridge."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.camera import Camera

from bridge_publisher import BridgePublisher, compute_t_cam_world

if TYPE_CHECKING:
    from bridge_camera_cfg import BridgeCameraCfg
    from bridge_publisher_ipc import BridgePublisherIPC

logger = logging.getLogger(__name__)


# Rotation taking a USD camera frame (looks down local -Z, +Y up, +X right)
# into the ROS body frame (x-forward, y-left, z-up) that the publisher's
# ``compute_t_cam_world`` expects before it applies the body->optical
# correction. Columns are the body axes expressed in USD-camera coords:
#   x_body (forward) = -z_cam, y_body (left) = -x_cam, z_body (up) = +y_cam.
# Cross-check: for an identity USD-camera world rotation this yields
# ``r_world_body @ ROS_BODY_TO_OPTICAL == diag(1, -1, -1)``, i.e. the same
# optical frame the direct USD-cam->optical mapping produces.
_R_USDCAM_TO_BODY = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


class BridgeCamera(Camera):
    """Camera sensor that publishes each render to the Holoscan bridge after the base updates.

    Hooks into :meth:`_update_buffers_impl` so the framework's existing render
    path runs unchanged. After the base buffers are populated for the current
    step, the selected view is read back, converted to the wire format expected
    by the realtime-3d-reconstruction receiver, and sent via a ``PUSH`` socket.
    """

    cfg: "BridgeCameraCfg"

    def __init__(self, cfg: "BridgeCameraCfg") -> None:
        super().__init__(cfg)
        self._publisher: BridgePublisher | None = None
        self._publisher_ipc: "BridgePublisherIPC | None" = None  # noqa: UP037
        self._step_counter: int = 0
        # Inverse of the first ``T_cam_world`` captured from the sim; used to
        # rebase later frames so the first published pose is identity.
        self._initial_world_T_cam: np.ndarray | None = None
        # Lazy cv2 handle for the optional debug preview window.
        self._cv2 = None
        # Rolling FPS over the last kFpsWindowS seconds, matching the
        # container-side NvbloxOp / IsaacLabBridgeReceiverOp counters.
        self._fps_window_s: float = 3.0
        self._fps_window_start: float = 0.0
        self._fps_window_frames: int = 0
        # Pure RTX render timing: how long ``super()._update_buffers_impl``
        # takes per call (i.e. Kit's render + readback). Independent of the
        # downstream host copy / ZMQ send, so represents the max FPS the
        # renderer alone could sustain.
        self._render_window_start: float = 0.0
        self._render_window_count: int = 0
        self._render_window_time_total: float = 0.0

    def update(self, dt: float, force_recompute: bool = False) -> None:  # type: ignore[override]
        # No observation term reads this camera, so the default lazy update
        # never fires. Force a recompute every step.
        super().update(dt, force_recompute=True)

    def _ensure_publisher(self) -> BridgePublisher:
        if self._publisher is None:
            self._publisher = BridgePublisher(endpoint=self.cfg.endpoint)
        return self._publisher

    def _ensure_publisher_ipc(self, width: int, height: int, device: torch.device | str | None = None):
        if self._publisher_ipc is None:
            from bridge_publisher_ipc import BridgePublisherIPC

            self._publisher_ipc = BridgePublisherIPC(
                endpoint=self.cfg.endpoint,
                width=width,
                height=height,
                ring_size=int(getattr(self.cfg, "ipc_ring_size", 3)),
                device=device if device is not None else "cuda:0",
            )
        return self._publisher_ipc

    def close(self) -> None:
        """Tear down the publisher socket(s) and any cv2 windows."""
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        if self._publisher_ipc is not None:
            self._publisher_ipc.close()
            self._publisher_ipc = None
        if self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:  # pragma: no cover - defensive
                pass
            self._cv2 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def _update_buffers_impl(self, env_mask: wp.array) -> None:  # type: ignore[override]
        # Time only the upstream call so we get a clean RTX render number.
        t_render_start = time.monotonic()
        super()._update_buffers_impl(env_mask)
        render_elapsed = time.monotonic() - t_render_start
        self._update_render_only_fps(render_elapsed)

        self._step_counter += 1
        if self._step_counter % max(1, self.cfg.publish_every_n_steps) != 0:
            return
        try:
            self._publish_current_frame()
        except Exception:
            # A publisher hiccup must never take down the simulation loop.
            logger.exception("BridgeCamera publish failed; continuing sim")

    def _update_render_only_fps(self, render_elapsed_s: float) -> None:
        """Rolling avg of pure-render time / max sustainable FPS, 3 s window."""
        now = time.monotonic()
        if self._render_window_count == 0:
            self._render_window_start = now
            self._render_window_time_total = 0.0
        self._render_window_count += 1
        self._render_window_time_total += render_elapsed_s

        wall_elapsed = now - self._render_window_start
        if wall_elapsed >= self._fps_window_s:
            n = self._render_window_count
            avg_ms = (self._render_window_time_total / n) * 1000.0 if n > 0 else 0.0
            peak_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
            print(
                f"[bridge-camera] render_only_ms={avg_ms:.2f}"
                f" peak_render_fps={peak_fps:.2f}"
                f" (avg over {wall_elapsed:.2f}s, n={n} renders)",
                flush=True,
            )
            self._render_window_count = 0
            self._render_window_time_total = 0.0

    def _publish_current_frame(self) -> None:
        data = self._data
        if data is None or data.image_shape is None or data.output is None:
            return

        view_idx = int(self.cfg.publish_view_index)
        height, width = data.image_shape

        fx, fy, cx, cy = self._extract_intrinsics(view_idx)
        t_cam_world_col_major = self._extract_t_cam_world_column_major(view_idx)
        debug = getattr(self.cfg, "debug_window", False)

        if getattr(self.cfg, "use_cuda_ipc", False):
            # v2 CUDA-IPC path: keep frames on GPU, ship only a small header.
            rgb_gpu = self._extract_rgb_gpu(view_idx, height, width)
            if rgb_gpu is None:
                return
            depth_gpu = self._extract_depth_u16_gpu(view_idx, height, width)
            if depth_gpu is None:
                return
            if debug:
                # cv2 preview still needs host bytes; only pay D2H if asked.
                self._show_debug_window(
                    rgb_gpu.contiguous().cpu().numpy().tobytes(),
                    depth_gpu.contiguous().cpu().numpy().tobytes(),
                    width, height,
                )
            # Pin the publisher's ring + IPC events to the same physical GPU
            # Kit is rendering on. Cross-GPU memory IPC sometimes works via
            # P2P, but event IPC is strictly per-GPU — mixing GPUs causes
            # ``cudaIpcOpenEventHandle`` to fail with
            # ``cudaErrorMapBufferObjectFailed``.
            publisher_ipc = self._ensure_publisher_ipc(width, height, device=rgb_gpu.device)
            publisher_ipc.publish_frame(
                rgb_gpu=rgb_gpu,
                depth_u16_gpu=depth_gpu,
                width=width, height=height,
                fx=fx, fy=fy, cx=cx, cy=cy,
                t_cam_world_column_major=t_cam_world_col_major,
            )
        else:
            # v1 host-staged path.
            rgb_bytes = self._extract_rgb_bytes(view_idx, height, width)
            if rgb_bytes is None:
                return
            depth_bytes = self._extract_depth_u16_mm_bytes(view_idx, height, width)
            if depth_bytes is None:
                return
            if debug:
                self._show_debug_window(rgb_bytes, depth_bytes, width, height)
            publisher = self._ensure_publisher()
            publisher.publish_frame(
                rgb_bytes=rgb_bytes,
                depth_u16_mm_bytes=depth_bytes,
                width=width, height=height,
                fx=fx, fy=fy, cx=cx, cy=cy,
                t_cam_world_column_major=t_cam_world_col_major,
            )

        # Rolling render-FPS over the last ``self._fps_window_s`` seconds.
        now = time.monotonic()
        if self._fps_window_frames == 0:
            self._fps_window_start = now
        self._fps_window_frames += 1
        elapsed = now - self._fps_window_start
        if elapsed >= self._fps_window_s:
            fps = self._fps_window_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"[bridge-camera] render_fps={fps:.2f}"
                f" (avg over {elapsed:.2f}s, n={self._fps_window_frames} frames)",
                flush=True,
            )
            self._fps_window_frames = 0

    def _read_prim_world_pose_from_fabric(self, prim_path: str):
        """Read ``prim_path``'s world transform from its Fabric hierarchy matrix.

        Reads the same Fabric state the RTX renderer consumes: ``render()``
        calls ``physics_manager.forward()`` to sync physics -> Fabric before
        rendering, so a read performed *after* the render reflects exactly the
        transform the frame was rendered from.

        Returns ``(pos_float64, quat_xyzw_float64)`` (quaternion in the prim's
        own frame convention) or ``None`` on failure.
        """
        try:
            from isaaclab.sim.utils.stage import get_current_stage_id
            import usdrt
            from usdrt import Rt
        except ImportError:
            return None

        stage_id = get_current_stage_id()
        rt_stage = usdrt.Usd.Stage.Attach(stage_id)
        if rt_stage is None:
            return None
        rt_prim = rt_stage.GetPrimAtPath(prim_path)
        if rt_prim is None:
            return None
        rt_xformable = Rt.Xformable(rt_prim)
        if rt_xformable is None:
            return None
        world_matrix_attr = rt_xformable.GetFabricHierarchyWorldMatrixAttr()
        if world_matrix_attr is None:
            return None
        rt_matrix = world_matrix_attr.Get()
        if rt_matrix is None:
            return None

        rt_pos = rt_matrix.ExtractTranslation()
        rt_quat = rt_matrix.ExtractRotationQuat()
        pos = np.array([rt_pos[0], rt_pos[1], rt_pos[2]], dtype=np.float64)
        imag = rt_quat.GetImaginary()
        quat_xyzw = np.array(
            [imag[0], imag[1], imag[2], rt_quat.GetReal()], dtype=np.float64
        )
        return pos, quat_xyzw

    def _read_parent_world_pose_from_fabric(self):
        """Read the camera's parent prim world transform from Fabric.

        ``CameraData.pos_w`` / ``.quat_w_world`` do not refresh per step for
        cameras mounted under articulation links, even with
        ``update_latest_camera_pose=True``. Reading the parent prim from
        Fabric is the workaround (same mechanism :class:`XrAnchorSynchronizer`
        uses for the pelvis). Still used by the demo / keyboard controllers to
        capture an initial anchor pose; the published frame pose now comes from
        :meth:`_read_camera_world_pose_body_from_fabric` instead.

        Returns ``(pos_float64, quat_xyzw_float64)`` or ``None`` on failure.
        """
        if not hasattr(self, "_cached_parent_prim_path"):
            parent_path = "/".join(self.cfg.prim_path.split("/")[:-1])
            self._cached_parent_prim_path = parent_path.replace("env_.*", "env_0")
        return self._read_prim_world_pose_from_fabric(self._cached_parent_prim_path)

    def _read_camera_world_pose_body_from_fabric(self):
        """Read the *rendered* camera prim's world pose, in ROS body convention.

        Reads the leaf camera prim's Fabric world matrix -- the exact transform
        the RTX renderer rendered this frame from -- and rotates the USD-camera
        orientation into the x-forward / y-left / z-up body convention expected
        by :func:`compute_t_cam_world`. This is the single source of truth for
        the published pose: it captures both articulation-driven motion and
        ``set_world_poses`` moves (the camera prim transform reflects either)
        without any parent+offset re-derivation, so the published
        ``T_cam_world`` is guaranteed to match the rendered pixels.

        Returns ``(pos_float64, quat_xyzw_body_float64)`` or ``None``.
        """
        if not hasattr(self, "_cached_camera_prim_path"):
            self._cached_camera_prim_path = self.cfg.prim_path.replace("env_.*", "env_0")

        pose = self._read_prim_world_pose_from_fabric(self._cached_camera_prim_path)
        if pose is None:
            return None

        from scipy.spatial.transform import Rotation as R

        pos, quat_usdcam_xyzw = pose
        # r_world_body = r_world_usdcam @ R_usdcam_to_body
        r_world_body = R.from_quat(quat_usdcam_xyzw) * R.from_matrix(_R_USDCAM_TO_BODY)
        return pos, r_world_body.as_quat().astype(np.float64)

    def _maybe_log_pose_sync_check(self, pos_used: np.ndarray, quat_used: np.ndarray) -> None:
        """One-time sanity log: leaf-Fabric pose vs legacy parent+offset.

        Confirms the leaf-camera convention conversion reproduces the pose the
        old parent+offset path published, validating the change against the
        known-good result. Expected to read ~0 mm / ~0 deg on the first frame
        (before any ``set_world_poses`` move offsets the camera from
        parent+offset). Non-fatal: logs once and never blocks the sim loop.
        """
        if getattr(self, "_pose_sync_checked", False):
            return
        self._pose_sync_checked = True
        try:
            parent_pose = self._read_parent_world_pose_from_fabric()
            if parent_pose is None:
                return
            from scipy.spatial.transform import Rotation as R

            parent_pos_w, parent_quat_xyzw = parent_pose
            r_parent = R.from_quat(parent_quat_xyzw)
            offset_pos = np.array(list(self.cfg.offset.pos), dtype=np.float64)
            offset_rot_xyzw = np.array(list(self.cfg.offset.rot), dtype=np.float64)
            legacy_pos = parent_pos_w + r_parent.as_matrix() @ offset_pos
            legacy_quat = (r_parent * R.from_quat(offset_rot_xyzw)).as_quat()

            pos_err_mm = float(
                np.linalg.norm(np.asarray(pos_used, dtype=np.float64) - legacy_pos) * 1000.0
            )
            dot = abs(float(np.dot(np.asarray(quat_used, dtype=np.float64), legacy_quat)))
            ang_err_deg = float(np.degrees(2.0 * np.arccos(min(1.0, dot))))
            print(
                f"[bridge-camera] pose-sync check: leaf-Fabric vs parent+offset"
                f" pos_err={pos_err_mm:.2f}mm ang_err={ang_err_deg:.3f}deg"
                " (both ~0 => convention validated)",
                flush=True,
            )
        except Exception:  # pragma: no cover - diagnostic only
            logger.exception("pose-sync check failed (non-fatal)")

    def _show_debug_window(
        self,
        rgb_bytes: bytes,
        depth_u16_mm_bytes: bytes,
        width: int,
        height: int,
    ) -> None:
        """Refresh the side-by-side cv2 preview with the exact bytes being shipped."""
        if self._cv2 is None:
            try:
                import cv2  # type: ignore
            except ImportError:
                return
            self._cv2 = cv2

        cv2 = self._cv2

        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth_u16 = np.frombuffer(depth_u16_mm_bytes, dtype=np.uint16).reshape(
            height, width
        )
        depth_cap_mm = max(int(self.cfg.max_depth_m * 1000.0), 1)
        depth_u8 = np.clip(
            depth_u16.astype(np.float32) * (255.0 / depth_cap_mm), 0.0, 255.0
        ).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color[depth_u16 == 0] = (0, 0, 0)

        gutter = np.zeros((height, 4, 3), dtype=np.uint8)
        side_by_side = np.hstack([bgr, gutter, depth_color])

        cv2.putText(
            side_by_side, "RGB", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            side_by_side, f"Depth (mm, cap={depth_cap_mm})",
            (width + 4 + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )

        try:
            cv2.imshow("bridge-camera (sent to nvblox)", side_by_side)
            cv2.waitKey(1)
        except cv2.error:
            self._cv2 = None

    def _extract_rgb_gpu(self, view_idx: int, height: int, width: int) -> torch.Tensor | None:
        """Return the (H, W, 3) uint8 CUDA tensor (no host roundtrip).

        Drops alpha if the renderer gave us RGBA. Returns ``None`` on shape /
        availability mismatch.
        """
        data = self._data
        assert data is not None and data.output is not None
        proxy = data.output.get(self.cfg.rgb_data_type)
        if proxy is None:
            return None
        tensor: torch.Tensor = proxy.torch
        if tensor.ndim != 4 or tensor.shape[0] <= view_idx:
            return None
        view = tensor[view_idx]
        if view.shape[-1] == 4:  # drop alpha if RGBA
            view = view[..., :3]
        if view.shape[-1] != 3:
            return None
        if view.dtype != torch.uint8:
            view = view.clamp_(0, 255).to(torch.uint8)
        return view.contiguous()

    def _extract_depth_u16_gpu(
        self, view_idx: int, height: int, width: int
    ) -> torch.Tensor | None:
        """Return the (H, W) uint16 CUDA tensor (millimeters)."""
        data = self._data
        assert data is not None and data.output is not None
        proxy = data.output.get(self.cfg.depth_data_type)
        if proxy is None:
            return None
        tensor: torch.Tensor = proxy.torch
        if tensor.ndim != 4 or tensor.shape[0] <= view_idx:
            return None
        view = tensor[view_idx]
        if view.ndim == 3 and view.shape[-1] == 1:
            view = view.squeeze(-1)
        if view.shape != (height, width):
            return None
        # Mask non-finite + out-of-range to 0 (nvblox treats 0 as invalid),
        # then convert meters → mm and clamp to uint16 range.
        view_m = view.to(torch.float32)
        valid_mask = torch.isfinite(view_m) & (view_m > 0.0) & (view_m <= self.cfg.max_depth_m)
        view_m = torch.where(valid_mask, view_m, torch.zeros_like(view_m))
        depth_mm = (view_m * 1000.0).clamp_(0.0, 65535.0).to(torch.uint16)
        return depth_mm.contiguous()

    def _extract_rgb_bytes(self, view_idx: int, height: int, width: int) -> bytes | None:
        data = self._data
        assert data is not None and data.output is not None
        proxy = data.output.get(self.cfg.rgb_data_type)
        if proxy is None:
            return None
        tensor: torch.Tensor = proxy.torch
        if tensor.ndim != 4 or tensor.shape[0] <= view_idx:
            return None
        view = tensor[view_idx]
        if view.shape[-1] == 4:  # drop alpha if RGBA
            view = view[..., :3]
        if view.shape[-1] != 3:
            return None
        if view.dtype != torch.uint8:
            view = view.clamp_(0, 255).to(torch.uint8)
        view = view.contiguous().cpu().numpy()
        return view.tobytes()

    def _extract_depth_u16_mm_bytes(
        self, view_idx: int, height: int, width: int
    ) -> bytes | None:
        data = self._data
        assert data is not None and data.output is not None
        proxy = data.output.get(self.cfg.depth_data_type)
        if proxy is None:
            return None
        tensor: torch.Tensor = proxy.torch
        if tensor.ndim != 4 or tensor.shape[0] <= view_idx:
            return None
        view = tensor[view_idx]
        if view.ndim == 3 and view.shape[-1] == 1:
            view = view.squeeze(-1)
        if view.shape != (height, width):
            return None
        # Mask non-finite + out-of-range to 0 (nvblox treats 0 as invalid).
        view_m = view.to(torch.float32)
        valid_mask = torch.isfinite(view_m) & (view_m > 0.0) & (view_m <= self.cfg.max_depth_m)
        view_m = torch.where(valid_mask, view_m, torch.zeros_like(view_m))
        depth_mm = (view_m * 1000.0).clamp_(0.0, 65535.0).to(torch.uint16)
        return depth_mm.contiguous().cpu().numpy().tobytes()

    def _extract_intrinsics(self, view_idx: int) -> tuple[float, float, float, float]:
        data = self._data
        assert data is not None and data.intrinsic_matrices is not None
        k = data.intrinsic_matrices.torch[view_idx].detach().cpu().numpy()
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

    def _extract_t_cam_world_column_major(self, view_idx: int) -> np.ndarray:
        data = self._data
        assert data is not None and data.pos_w is not None and data.quat_w_world is not None

        # Publish exactly the pose the renderer used: the leaf camera prim's
        # Fabric world matrix, read here *after* ``super()._update_buffers_impl``
        # rendered (which first ran ``physics_manager.forward()`` to push the
        # latest transforms into Fabric). Pose and pixels therefore come from
        # the same Fabric state at the same instant -> guaranteed in sync.
        cam_pose = self._read_camera_world_pose_body_from_fabric()
        if cam_pose is not None:
            pos_w, quat_xyzw = cam_pose
        else:
            # Fabric unavailable: fall back to the framework snapshot. NOTE this
            # is sampled one Fabric-forward behind the render, so it may lag the
            # pixels by a frame -- only hit when the usdrt read fails.
            pos_w = data.pos_w.torch[view_idx].detach().cpu().numpy().astype(np.float64)
            quat_xyzw = data.quat_w_world.torch[view_idx].detach().cpu().numpy().astype(np.float64)

        self._maybe_log_pose_sync_check(pos_w, quat_xyzw)

        # T_cam_simworld: cam-from-Isaac-Sim-world, camera in optical convention.
        t_cam_simworld = compute_t_cam_world(
            pos_w=pos_w,
            quat_w_xyzw=quat_xyzw,
            apply_optical_correction=self.cfg.apply_optical_frame_correction,
        )

        t_cam_anchor = t_cam_simworld

        # Rebase: anchor frame = camera's optical frame at t=0, so
        # T_cam_anchor(t) = T_cam_simworld(t) @ inv(T_cam_simworld(0)).
        if self.cfg.rebase_to_initial_pose:
            if self._initial_world_T_cam is None:
                self._initial_world_T_cam = np.linalg.inv(t_cam_simworld).astype(
                    np.float64, copy=False
                )
                t_cam_anchor = np.eye(4, dtype=np.float32)
            else:
                t_cam_anchor = (
                    t_cam_simworld.astype(np.float64) @ self._initial_world_T_cam
                ).astype(np.float32)

        # Axis remap: flip X and Z of the anchor frame so it has
        # +X = left, +Y = down (unchanged), +Z = back.
        r_flip = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        t_cam_anchor = (t_cam_anchor.astype(np.float32) @ r_flip).astype(np.float32)

        # Column-major flatten so ``Eigen::Map<Matrix4f, ColMajor>`` on the
        # receiver reconstructs the matrix correctly. NOTE:
        # ``np.asfortranarray(M).ravel()`` is NOT column-major — ravel
        # defaults to ``order='C'``. Use ``order='F'`` explicitly.
        return t_cam_anchor.ravel(order="F").astype(np.float32, copy=False)
