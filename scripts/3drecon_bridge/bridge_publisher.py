# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ZeroMQ publisher matching the realtime-3d-reconstruction Holoscan receiver wire format.

The C++ receiver (``IsaacLabBridgeReceiverOp``) binds a ZMQ ``PULL`` socket and
expects a three-part multipart message per frame: a 140-byte packed header,
the raw RGB bytes (``uint8``, ``H * W * 3``), and the raw depth bytes
(``uint16`` millimeters, ``H * W * 2``). The header field order and the
``T_cam_world`` column-major layout are mirrored verbatim from the C++ struct.
"""

from __future__ import annotations

import logging
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

# Mirror of the C++ ``IsaacLabRgbdHeader`` (see
# realtime-3d-reconstruction/src/sensor/isaaclab_bridge_receiver.cpp).
_HEADER_FORMAT = "<8sIIQqII5f16fQII"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
assert _HEADER_SIZE == 140, f"Expected 140-byte header, got {_HEADER_SIZE}"

_MAGIC = b"ILRGBD1\0"
_SCHEMA_VERSION = 1
_DEPTH_SCALE_M_PER_UNIT = 0.001  # depth on the wire is uint16 millimeters

# Stereo host-staged variant (wire v3): identical header fields plus a trailing
# ``baseline_m`` float32 (header_size = 144), shipped as a FOUR-part multipart:
# header, rgb_left, rgb_right, depth. Consumers dispatch on the magic.
_HEADER_FORMAT_V3 = "<8sIIQqII5f16fQIIf"
_HEADER_SIZE_V3 = struct.calcsize(_HEADER_FORMAT_V3)
assert _HEADER_SIZE_V3 == 144, f"Expected 144-byte v3 header, got {_HEADER_SIZE_V3}"

_MAGIC_V3 = b"ILRGBD3\0"
_SCHEMA_VERSION_V3 = 3


class BridgePublisher:
    """ZeroMQ ``PUSH`` publisher that ships RGB-D frames to the Holoscan receiver.

    This is a thin wrapper around ``pyzmq``. It owns the socket and reusable
    numpy scratch buffers and exposes :meth:`publish_frame` which packs a
    header plus pre-prepared RGB / depth byte buffers into a 3-part multipart
    message.

    Args:
        endpoint: ZMQ endpoint to connect to (e.g. ``tcp://127.0.0.1:5570``).
        send_high_water_mark: PUSH-side queue depth. Defaults to 1 so the
            publisher drops stale frames rather than queueing them.
        linger_ms: Number of milliseconds to wait for outstanding messages on
            socket close. Defaults to 0 so shutdown is immediate.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5570",
        send_high_water_mark: int = 1,
        linger_ms: int = 0,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - exercised when zmq is missing
            raise RuntimeError(
                "pyzmq is required to publish frames to the Holoscan bridge. "
                "Install it inside the Isaac Lab venv with `pip install pyzmq`."
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, send_high_water_mark)
        self._socket.setsockopt(zmq.LINGER, linger_ms)
        self._socket.connect(endpoint)
        self._endpoint = endpoint
        self._frame_id = 0
        self._reset_sequence = 0
        logger.info("BridgePublisher connected to %s", endpoint)

    def close(self) -> None:
        """Close the underlying socket. Safe to call multiple times."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            finally:
                self._socket = None

    def __enter__(self) -> BridgePublisher:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def bump_reset_sequence(self) -> None:
        """Signal that the simulation has been reset.

        The receiver mirrors this counter through to ``NvbloxOp`` so the
        downstream cuVSLAM-SLAM bookkeeping can drop stale keyframes if the
        publisher ever gets reused across episodes.
        """
        self._reset_sequence += 1

    def publish_frame(
        self,
        rgb_bytes: bytes,
        depth_u16_mm_bytes: bytes,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        t_cam_world_column_major: np.ndarray,
        timestamp_ns: int | None = None,
        rgb_right_bytes: bytes | None = None,
        baseline_m: float = 0.0,
    ) -> None:
        """Pack and send a single RGB-D frame (optionally with a right RGB eye).

        Args:
            rgb_bytes: Contiguous ``height * width * 3`` uint8 RGB payload.
            depth_u16_mm_bytes: Contiguous ``height * width * 2`` uint16
                little-endian depth payload, interpreted by the receiver as
                millimeters.
            width: Image width in pixels.
            height: Image height in pixels.
            fx: Focal length x [pixels].
            fy: Focal length y [pixels].
            cx: Principal point x [pixels].
            cy: Principal point y [pixels].
            t_cam_world_column_major: 16 ``float32`` values describing
                ``T_cam_world`` in **column-major** order (matches the
                Eigen ``ColMajor`` ``Map`` used in ``NvbloxOp``).
            timestamp_ns: Optional capture timestamp [ns]. If ``None``,
                ``time.monotonic_ns()`` is used.
            rgb_right_bytes: Optional right-eye RGB payload (same size as
                ``rgb_bytes``). When given, the message is sent in the stereo
                v3 wire format (144-byte header + 4 parts) instead of v1.
            baseline_m: Stereo baseline [m] (v3 header field; ignored for v1).
        """
        if self._socket is None:
            raise RuntimeError("BridgePublisher socket is already closed")

        expected_rgb = width * height * 3
        expected_depth = width * height * 2
        if len(rgb_bytes) != expected_rgb:
            raise ValueError(f"rgb_bytes size mismatch: got {len(rgb_bytes)}, expected {expected_rgb}")
        if len(depth_u16_mm_bytes) != expected_depth:
            raise ValueError(
                f"depth_u16_mm_bytes size mismatch: got {len(depth_u16_mm_bytes)}, expected {expected_depth}"
            )
        if rgb_right_bytes is not None and len(rgb_right_bytes) != expected_rgb:
            raise ValueError(f"rgb_right_bytes size mismatch: got {len(rgb_right_bytes)}, expected {expected_rgb}")
        if t_cam_world_column_major.size != 16:
            raise ValueError("t_cam_world_column_major must contain 16 floats")

        ts_ns = int(time.monotonic_ns()) if timestamp_ns is None else int(timestamp_ns)
        pose = np.ascontiguousarray(t_cam_world_column_major, dtype=np.float32).ravel()

        common_fields = (
            self._frame_id,
            ts_ns,
            int(width),
            int(height),
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            _DEPTH_SCALE_M_PER_UNIT,
            *pose.tolist(),
            self._reset_sequence,
            expected_rgb,
            expected_depth,
        )
        if rgb_right_bytes is not None:
            header = struct.pack(
                _HEADER_FORMAT_V3,
                _MAGIC_V3,
                _SCHEMA_VERSION_V3,
                _HEADER_SIZE_V3,
                *common_fields,
                float(baseline_m),
            )
        else:
            header = struct.pack(
                _HEADER_FORMAT,
                _MAGIC,
                _SCHEMA_VERSION,
                _HEADER_SIZE,
                *common_fields,
            )

        # PUSH with HWM=1 will drop newest if the receiver is behind, so the
        # Holoscan side always sees the freshest frame.
        flags = self._zmq.NOBLOCK
        try:
            self._socket.send(header, flags=flags | self._zmq.SNDMORE)
            self._socket.send(rgb_bytes, flags=flags | self._zmq.SNDMORE)
            if rgb_right_bytes is not None:
                self._socket.send(rgb_right_bytes, flags=flags | self._zmq.SNDMORE)
            self._socket.send(depth_u16_mm_bytes, flags=flags)
        except self._zmq.Again:
            # Receiver isn't keeping up; drop this frame entirely so we don't
            # send a partial multipart that would desync the framing.
            logger.debug("BridgePublisher dropped frame %d (receiver busy)", self._frame_id)
        else:
            self._frame_id += 1


def quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a ``3x3`` float32 rotation matrix.

    Args:
        quat_xyzw: Length-4 array in ``(x, y, z, w)`` order.

    Returns:
        The corresponding ``3x3`` float32 rotation matrix. Identity is returned
        for near-zero-norm input to avoid division by zero on first-step warmup.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


# ``R_body_optical``: optical basis vectors expressed in body coordinates.
# Columns are: optical +X = body -Y, optical +Y = body -Z, optical +Z = body +X.
# Right-multiply onto ``R_world_body`` to produce ``R_world_optical``.
ROS_BODY_TO_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def compute_t_cam_world(
    pos_w: np.ndarray,
    quat_w_xyzw: np.ndarray,
    apply_optical_correction: bool,
) -> np.ndarray:
    """Compose ``T_cam_world`` from world-frame camera pose components.

    Args:
        pos_w: Camera origin in world frame, shape ``(3,)``, meters.
        quat_w_xyzw: Camera orientation in world frame as an ``(x, y, z, w)``
            quaternion.
        apply_optical_correction: When ``True``, applies the standard ROS
            body-to-optical-frame rotation so the returned transform matches
            the OpenCV convention (``+Z`` through the lens, ``+X`` right,
            ``+Y`` down) that ``NvbloxOp`` and the existing capture operators
            already use.

    Returns:
        A ``4x4`` float32 ``T_cam_world`` (camera-from-world) transform. The
        caller is responsible for packing it column-major before sending.
    """
    r_world_body = quat_xyzw_to_rotation_matrix(quat_w_xyzw)
    if apply_optical_correction:
        r_world_cam = r_world_body @ ROS_BODY_TO_OPTICAL
    else:
        r_world_cam = r_world_body
    t_world_cam = np.eye(4, dtype=np.float32)
    t_world_cam[:3, :3] = r_world_cam
    t_world_cam[:3, 3] = np.asarray(pos_w, dtype=np.float32).reshape(3)
    return np.linalg.inv(t_world_cam).astype(np.float32)
