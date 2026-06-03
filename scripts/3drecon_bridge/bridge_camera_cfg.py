# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the bridge camera that publishes frames to the Holoscan receiver."""

from __future__ import annotations

from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class BridgeCameraCfg(CameraCfg):
    """A :class:`~isaaclab.sensors.camera.CameraCfg` that also publishes its frames over ZeroMQ.

    Renders like a regular Isaac Lab camera; every base ``CameraCfg`` field
    (resolution, spawn cfg, offsets, data types) applies as usual. The extra
    fields below configure the side-channel that ships RGB-D + intrinsics +
    pose to the realtime-3d-reconstruction Holoscan app.

    Assumes :attr:`data_types` includes ``"rgb"`` and
    ``"distance_to_image_plane"``.
    """

    class_type: type | str = "bridge_camera:BridgeCamera"

    endpoint: str = "tcp://127.0.0.1:5570"
    """ZeroMQ endpoint of the Holoscan receiver (loopback via ``--network=host``)."""

    depth_data_type: str = "distance_to_image_plane"
    """Key in :attr:`CameraData.output` for the depth payload (perpendicular meters)."""

    rgb_data_type: str = "rgb"
    """Key in :attr:`CameraData.output` for the RGB payload. Alpha is dropped on send."""

    publish_view_index: int = 0
    """Which view to publish when the cfg vectorises. Only one view is supported."""

    max_depth_m: float = 65.0
    """Clamp on the wire. Values > this (and non-finite) become 0 (invalid).
    The uint16 mm wire format caps at 65.535 m anyway."""

    apply_optical_frame_correction: bool = True
    """Rotate the world pose into OpenCV optical convention (``+Z`` through
    the lens, ``+X`` right, ``+Y`` down) so nvblox sees the camera in the
    convention it expects."""

    publish_every_n_steps: int = 1
    """Throttle: publish every N camera updates."""

    debug_window: bool = True
    """Open a side-by-side cv2 preview window showing the exact bytes shipped.

    Requires ``opencv-python``. Silently disables itself if cv2 is missing or
    no display is available. ~1-2 ms host CPU per frame; turn off in production.
    """

    use_cuda_ipc: bool = False
    """Share GPU buffers with the Holoscan receiver via CUDA IPC (v2 wire format).

    .. warning::
        Temporarily set to ``False`` to diagnose frame-boundary geometry jumps.
        The v2 CUDA-IPC ring has no consumer back-pressure (the CUDA-event
        handshake was removed for the CUDA 12.8/13.0 mismatch), so the producer
        can overwrite a ring slot before the receiver reads it -- fusing one
        frame's pose with a later frame's pixels. The v1 host-staged path below
        ships RGB+depth+pose in a single atomic ZMQ multipart message, so the
        triple is bit-exact paired by construction (no shared mutable slot).
        Re-enable once the IPC ring gains frame-id validation / back-pressure.

    When ``True``, RGB and depth stay on GPU: the publisher allocates a ring of
    CUDA buffers + IPC events, sends handles to the receiver in a one-shot
    handshake, then per-frame does a small D2D copy into a ring slot and ships
    only a ~144-byte header (no payload). The receiver D2D-copies from the
    IPC-mapped buffer directly into nvblox's GXF tensor. Eliminates the host
    roundtrip (D2H + ZMQ + H2D) — saves ~8-10 ms per frame at 1280x720.

    Requires:
      * ``cuda-python`` in the Isaac Lab venv (``./isaaclab.sh -p -m pip install cuda-python``)
      * the Holoscan container started with ``--ipc=host`` (already in docker_run.sh)
      * receiver built with ``isaaclab_3d_recon`` >= v2 (the receiver detects
        the v2 magic and routes to the CUDA-IPC path)

    Defaults to ``False`` so v1 host-staged path remains the production default.
    """

    ipc_ring_size: int = 4
    """Number of GPU slots in the IPC ring (ignored when ``use_cuda_ipc=False``).

    The producer cycles slots round-robin without waiting on the consumer
    (event IPC isn't available across the CUDA 12.8 / 13.0 process boundary),
    so the ring must be deep enough that slot ``N`` is fully consumed before
    slot ``N+ring_size`` is overwritten. 4 covers the consumer falling up to
    3 frames behind. Reduce to 3 if GPU memory is tight; increase to 6+ if
    the receiver routinely lags more than 3 frames.
    """
