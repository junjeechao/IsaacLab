# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA-IPC variant of :class:`BridgePublisher` (v2 wire format).

Eliminates the host roundtrip (D2H + ZMQ payload + H2D) by sharing a ring of
GPU buffers with the Holoscan receiver via CUDA IPC memory handles. Per-frame
the producer D2D-copies Kit's tensors into a slot, calls
``torch.cuda.current_stream().synchronize()`` so the bytes are committed before
the consumer reads, then sends a ~144 byte header. The consumer D2D-copies
from the IPC-mapped slot into its GXF tensor and syncs its own upload stream.

CUDA event IPC was *not* used: the host process runs against the CUDA 12.8
runtime (PyTorch) while the container runs against CUDA 13.0 (Holoscan), and
the cudaIpcEventHandle internal format differs between those major versions
(``cudaIpcOpenEventHandle`` returns ``cudaErrorMapBufferObjectFailed``).
Memory IPC handles are interoperable across CUDA 12/13 so they're kept.

Wire format v2
==============

Handshake (single ZMQ message, sent once before the first frame)
----------------------------------------------------------------
``<8sIIIIIII`` (36 bytes) followed by ``ring_size * 128`` bytes of handles::

  magic = "ILRGBD2\\0"
  schema_version    = 2
  message_type      = 0   (HANDSHAKE)
  header_size       = 36
  ring_size
  width, height
  depth_dtype       = 0   (uint16 mm)
  ---- payload, ring_size repeats of: ----
    rgb_mem_handle    64 B  (cudaIpcMemHandle_t)
    depth_mem_handle  64 B  (cudaIpcMemHandle_t)

Wire format v4 (stereo)
=======================

Same as v2 with magic ``"ILRGBD4\\0"`` / schema 4, except: the handshake
header gains a trailing float32 ``baseline_m`` (40 bytes total) and each
ring-slot handle group is ``rgb_left + rgb_right + depth`` (192 bytes/slot).
The per-frame header layout is identical to v2 apart from magic/schema.

Per-frame (single ZMQ message, header only)
-------------------------------------------
``<8sIIIQqII5f16fQII`` (144 bytes)::

  magic = "ILRGBD2\\0"
  schema_version    = 2
  message_type      = 1   (FRAME)
  header_size       = 144
  frame_id          uint64
  timestamp_ns      int64
  width, height
  fx, fy, cx, cy, depth_scale
  T_cam_world[16]   column-major float32
  reset_sequence    uint64
  slot_index        uint32
  depth_dtype       uint32 (echo)

Synchronization
===============

The producer ``cudaStreamSynchronize``s its current stream before sending the
header, so by the time the consumer receives the message the slot bytes are
materialized in GPU memory. The consumer then D2D copies and syncs its
upload stream. A ring of N>=3 slots gives the consumer N frame periods to
drain before the producer wraps around and overwrites; a corrupt read can
only happen if the consumer falls more than ``ring_size`` frames behind, in
which case at most one frame is torn.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

_V2_HANDSHAKE_FORMAT = "<8sIIIIIII"
_V2_HANDSHAKE_SIZE = struct.calcsize(_V2_HANDSHAKE_FORMAT)  # 36

_V2_FRAME_FORMAT = "<8sIIIQqII5f16fQII"
_V2_FRAME_SIZE = struct.calcsize(_V2_FRAME_FORMAT)  # 144

_MAGIC_V2 = b"ILRGBD2\0"
_SCHEMA_VERSION_V2 = 2

# Stereo CUDA-IPC variant (wire v4). Handshake gains a trailing float32
# ``baseline_m`` (40 bytes) and each ring-slot handle triple is
# ``rgb_left + rgb_right + depth`` (192 bytes/slot). The per-frame header
# layout is identical to v2 apart from magic/schema.
_V4_HANDSHAKE_FORMAT = "<8sIIIIIIIf"
_V4_HANDSHAKE_SIZE = struct.calcsize(_V4_HANDSHAKE_FORMAT)  # 40

_MAGIC_V4 = b"ILRGBD4\0"
_SCHEMA_VERSION_V4 = 4

_MESSAGE_TYPE_HANDSHAKE = 0
_MESSAGE_TYPE_FRAME = 1
_DEPTH_DTYPE_UINT16_MM = 0
_DEPTH_SCALE_M_PER_UNIT = 0.001

_IPC_HANDLE_SIZE = 64  # cudaIpc{Mem,Event}Handle_t.reserved size


def _load_cudart():
    """Lazy-load the CUDA runtime via cuda-python."""
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12
    except ImportError as e:
        raise RuntimeError(
            "BridgePublisherIPC requires cuda-python. Install with:\n    ./isaaclab.sh -p -m pip install cuda-python"
        ) from e
    return cudart


def _check(cudart, err, what: str) -> None:
    # cuda-python wraps void-returning functions' result as a ``(error,)`` tuple
    # (e.g. ``cudaEventRecord``, ``cudaEventSynchronize``). Functions with
    # output params return ``(error, out)`` and callers should unpack — but if
    # any caller forgot, unwrap here too.
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        try:
            err_name = err.name
        except AttributeError:
            err_name = str(err)
        raise RuntimeError(f"CUDA: {what} failed: {err_name}")


def _handle_bytes(handle) -> bytes:
    """Extract the 64-byte ``reserved`` field of a cudaIpc{Mem,Event}Handle_t."""
    raw = bytes(handle.reserved)
    if len(raw) != _IPC_HANDLE_SIZE:
        raise RuntimeError(f"Expected {_IPC_HANDLE_SIZE}-byte IPC handle, got {len(raw)}")
    return raw


class BridgePublisherIPC:
    """ZMQ PUSH publisher that shares GPU buffers via CUDA IPC.

    Args:
        endpoint: ZMQ endpoint to connect to (e.g. ``tcp://127.0.0.1:5570``).
        width, height: Image dimensions (fixed at init; matches the camera cfg).
        ring_size: Number of slots in the GPU ring.
        device: Torch CUDA device for the ring buffers.
    """

    def __init__(
        self,
        endpoint: str,
        width: int,
        height: int,
        ring_size: int = 3,
        device: str | torch.device = "cuda:0",
        stereo: bool = False,
        baseline_m: float = 0.0,
    ) -> None:
        try:
            import zmq
        except ImportError as e:
            raise RuntimeError("pyzmq required") from e
        self._zmq = zmq

        self._endpoint = endpoint
        self._width = int(width)
        self._height = int(height)
        self._ring_size = int(ring_size)
        self._device = torch.device(device)
        self._stereo = bool(stereo)
        self._baseline_m = float(baseline_m)
        self._frame_id = 0
        self._reset_sequence = 0
        self._slot = 0
        self._handshake_sent = False

        # --- ZMQ socket ---------------------------------------------------------------
        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

        self._cudart = _load_cudart()  # cudaMalloc + cudaIpcGetMemHandle
        cudart = self._cudart

        # Per-slot byte sizes (uint8 RGB, uint16 depth).
        self._rgb_slot_bytes = self._height * self._width * 3
        self._depth_slot_bytes = self._height * self._width * 2

        # Allocate the ring with RAW cudaMalloc. Each call returns the base
        # pointer of its own allocation, which is what cudaIpcGetMemHandle
        # requires — torch.zeros() returns sub-allocations from the
        # CachingAllocator (one big cudaMalloc block, sliced internally),
        # and passing a non-base pointer to cudaIpcGetMemHandle is undefined
        # behavior — in practice it silently aliases every slot's handle to
        # the base block, so the consumer reads stale memory.
        with torch.cuda.device(self._device):
            self._rgb_slot_ptrs: list[int] = []
            self._rgb_right_slot_ptrs: list[int] = []
            self._depth_slot_ptrs: list[int] = []
            stream_ptr = torch.cuda.current_stream(self._device).cuda_stream
            for _ in range(self._ring_size):
                err, p_rgb = cudart.cudaMalloc(self._rgb_slot_bytes)
                _check(cudart, err, "cudaMalloc(rgb slot)")
                self._rgb_slot_ptrs.append(int(p_rgb))
                _check(
                    cudart,
                    cudart.cudaMemsetAsync(int(p_rgb), 0, self._rgb_slot_bytes, stream_ptr),
                    "cudaMemsetAsync(rgb slot)",
                )

                if self._stereo:
                    err, p_rgb_r = cudart.cudaMalloc(self._rgb_slot_bytes)
                    _check(cudart, err, "cudaMalloc(rgb right slot)")
                    self._rgb_right_slot_ptrs.append(int(p_rgb_r))
                    _check(
                        cudart,
                        cudart.cudaMemsetAsync(int(p_rgb_r), 0, self._rgb_slot_bytes, stream_ptr),
                        "cudaMemsetAsync(rgb right slot)",
                    )

                err, p_depth = cudart.cudaMalloc(self._depth_slot_bytes)
                _check(cudart, err, "cudaMalloc(depth slot)")
                self._depth_slot_ptrs.append(int(p_depth))
                _check(
                    cudart,
                    cudart.cudaMemsetAsync(int(p_depth), 0, self._depth_slot_bytes, stream_ptr),
                    "cudaMemsetAsync(depth slot)",
                )
            torch.cuda.synchronize(self._device)

            self._rgb_mem_handles = [self._get_mem_handle_from_ptr(p) for p in self._rgb_slot_ptrs]
            self._rgb_right_mem_handles = [self._get_mem_handle_from_ptr(p) for p in self._rgb_right_slot_ptrs]
            self._depth_mem_handles = [self._get_mem_handle_from_ptr(p) for p in self._depth_slot_ptrs]

        # Diagnostic so a GPU-index mismatch with the container is obvious.
        gpu_name = torch.cuda.get_device_name(self._device)
        gpu_uuid = ""
        try:
            from cuda.bindings import runtime as _rt

            err, props = _rt.cudaGetDeviceProperties(self._device.index)
            if err == _rt.cudaError_t.cudaSuccess:
                gpu_uuid = bytes(props.uuid.bytes).hex()
        except Exception:  # pragma: no cover - best-effort
            pass
        logger.info(
            "BridgePublisherIPC initialized: %dx%d, ring_size=%d, endpoint=%s, device=%s (%s, uuid=%s)",
            self._width,
            self._height,
            self._ring_size,
            endpoint,
            str(self._device),
            gpu_name,
            gpu_uuid or "?",
        )

    # ---------------------------------------------------------------------------
    # IPC handle accessors
    # ---------------------------------------------------------------------------

    def _get_mem_handle_from_ptr(self, ptr: int) -> bytes:
        err, h = self._cudart.cudaIpcGetMemHandle(int(ptr))
        _check(self._cudart, err, "cudaIpcGetMemHandle")
        return _handle_bytes(h)

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            finally:
                self._socket = None
        # Free the raw cudaMalloc'd slot buffers.
        if hasattr(self, "_cudart") and self._cudart is not None:
            for p in getattr(self, "_rgb_slot_ptrs", []):
                if p:
                    self._cudart.cudaFree(int(p))
            for p in getattr(self, "_rgb_right_slot_ptrs", []):
                if p:
                    self._cudart.cudaFree(int(p))
            for p in getattr(self, "_depth_slot_ptrs", []):
                if p:
                    self._cudart.cudaFree(int(p))
            self._rgb_slot_ptrs = []
            self._rgb_right_slot_ptrs = []
            self._depth_slot_ptrs = []

    def __enter__(self) -> BridgePublisherIPC:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def bump_reset_sequence(self) -> None:
        self._reset_sequence += 1

    # ---------------------------------------------------------------------------
    # Handshake
    # ---------------------------------------------------------------------------

    def _send_handshake(self) -> None:
        if self._stereo:
            header = struct.pack(
                _V4_HANDSHAKE_FORMAT,
                _MAGIC_V4,
                _SCHEMA_VERSION_V4,
                _MESSAGE_TYPE_HANDSHAKE,
                _V4_HANDSHAKE_SIZE,
                self._ring_size,
                self._width,
                self._height,
                _DEPTH_DTYPE_UINT16_MM,
                self._baseline_m,
            )
        else:
            header = struct.pack(
                _V2_HANDSHAKE_FORMAT,
                _MAGIC_V2,
                _SCHEMA_VERSION_V2,
                _MESSAGE_TYPE_HANDSHAKE,
                _V2_HANDSHAKE_SIZE,
                self._ring_size,
                self._width,
                self._height,
                _DEPTH_DTYPE_UINT16_MM,
            )
        body = bytearray()
        for i in range(self._ring_size):
            body += self._rgb_mem_handles[i]
            if self._stereo:
                body += self._rgb_right_mem_handles[i]
            body += self._depth_mem_handles[i]
        # One-shot: send blocking so the receiver definitely gets it before frames.
        self._socket.send(header + bytes(body))
        logger.info(
            "BridgePublisherIPC handshake sent (%d header + %d handles bytes)",
            len(header),
            len(body),
        )

    # ---------------------------------------------------------------------------
    # Per-frame publish
    # ---------------------------------------------------------------------------

    def publish_frame(
        self,
        rgb_gpu: torch.Tensor,
        depth_u16_gpu: torch.Tensor,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        t_cam_world_column_major: np.ndarray,
        timestamp_ns: int | None = None,
        rgb_right_gpu: torch.Tensor | None = None,
    ) -> None:
        """Copy ``rgb_gpu`` / ``depth_u16_gpu`` into the next slot and send the header.

        ``rgb_gpu``: CUDA uint8 tensor shape ``(H, W, 3)``.
        ``depth_u16_gpu``: CUDA uint16 tensor shape ``(H, W)``.
        ``rgb_right_gpu``: optional right-eye RGB (stereo publishers only). If
        the publisher is stereo and the right frame is momentarily missing,
        the left frame is duplicated into the right slot (warned once).
        """
        if self._socket is None:
            raise RuntimeError("BridgePublisherIPC socket is closed")
        if width != self._width or height != self._height:
            raise ValueError(
                f"BridgePublisherIPC was initialized for {self._width}x{self._height};"
                f" got a frame of {width}x{height}. Recreate the publisher."
            )
        if rgb_right_gpu is not None and not self._stereo:
            if not getattr(self, "_mono_right_warned", False):
                self._mono_right_warned = True
                logger.warning(
                    "BridgePublisherIPC was created mono (right eye missing on the"
                    " first frame); ignoring rgb_right_gpu on later frames."
                )
            rgb_right_gpu = None
        if self._stereo and rgb_right_gpu is None:
            if not getattr(self, "_stereo_left_dup_warned", False):
                self._stereo_left_dup_warned = True
                logger.warning(
                    "Stereo BridgePublisherIPC got no right frame; duplicating the left eye into the right slot."
                )
            rgb_right_gpu = rgb_gpu

        if not self._handshake_sent:
            self._send_handshake()
            self._handshake_sent = True

        slot = self._slot
        cudart = self._cudart

        with torch.cuda.device(self._device):
            # 1. Wait for Kit's RTX renderer to finish writing rgb_gpu /
            #    depth_u16_gpu. These are views into Kit's tensors, and Kit
            #    writes them on a graphics-interop stream — not the current
            #    torch stream. Without this device-wide sync, the D2D copy
            #    races with the renderer and pulls torn pixels.
            torch.cuda.synchronize(self._device)

            stream_ptr = torch.cuda.current_stream(self._device).cuda_stream

            # 2. D2D copy from Kit's tensors into the raw slot pointers.
            #    cudaMemcpyAsync accepts an arbitrary source pointer (including
            #    torch sub-allocations), so this is safe for ``rgb_gpu``.
            _check(
                cudart,
                cudart.cudaMemcpyAsync(
                    self._rgb_slot_ptrs[slot],
                    int(rgb_gpu.data_ptr()),
                    self._rgb_slot_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream_ptr,
                ),
                "cudaMemcpyAsync(rgb -> slot)",
            )
            if self._stereo:
                _check(
                    cudart,
                    cudart.cudaMemcpyAsync(
                        self._rgb_right_slot_ptrs[slot],
                        int(rgb_right_gpu.data_ptr()),
                        self._rgb_slot_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        stream_ptr,
                    ),
                    "cudaMemcpyAsync(rgb right -> slot)",
                )
            _check(
                cudart,
                cudart.cudaMemcpyAsync(
                    self._depth_slot_ptrs[slot],
                    int(depth_u16_gpu.data_ptr()),
                    self._depth_slot_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream_ptr,
                ),
                "cudaMemcpyAsync(depth -> slot)",
            )

            # 3. Sync our stream so the bytes are committed before we tell
            #    the consumer about them. (Event IPC isn't usable across the
            #    CUDA 12.8 / 13.0 process pair — see module docstring.)
            torch.cuda.current_stream(self._device).synchronize()

        # 4. Pack and send the small header.
        ts_ns = int(time.monotonic_ns()) if timestamp_ns is None else int(timestamp_ns)
        pose = np.ascontiguousarray(t_cam_world_column_major, dtype=np.float32).ravel()
        if pose.size != 16:
            raise ValueError("t_cam_world_column_major must contain 16 floats")

        header = struct.pack(
            _V2_FRAME_FORMAT,
            _MAGIC_V4 if self._stereo else _MAGIC_V2,
            _SCHEMA_VERSION_V4 if self._stereo else _SCHEMA_VERSION_V2,
            _MESSAGE_TYPE_FRAME,
            _V2_FRAME_SIZE,
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
            slot,
            _DEPTH_DTYPE_UINT16_MM,
        )

        # Diagnostic: log the first 3 RGB bytes of (a) the source view from
        # Kit and (b) the raw slot pointer we just wrote. Pair with the
        # receiver's matching log to verify IPC surfaces the producer's
        # bytes. Every 60 frames to avoid spam.
        if self._frame_id % 60 == 0:
            try:
                src_pix = rgb_gpu.flatten()[:3].cpu().tolist()
                slot_buf = (ctypes.c_uint8 * 3)()
                _check(
                    cudart,
                    cudart.cudaMemcpy(
                        ctypes.addressof(slot_buf),
                        self._rgb_slot_ptrs[slot],
                        3,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    ),
                    "cudaMemcpy(slot diag)",
                )
                print(
                    f"[bridge-camera] ipc-diag PUBLISH frame={self._frame_id}"
                    f" slot={slot} src_rgb=({src_pix[0]},{src_pix[1]},{src_pix[2]})"
                    f" slot_rgb=({slot_buf[0]},{slot_buf[1]},{slot_buf[2]})"
                    f" slot_ptr=0x{self._rgb_slot_ptrs[slot]:x}",
                    flush=True,
                )
            except Exception:  # pragma: no cover
                pass

        try:
            self._socket.send(header, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            # Receiver isn't keeping up; drop this frame's notification.
            # The data is still in the slot but the receiver will catch up
            # on the next frame. Note: producer will still wait for
            # consumed_events[slot] next cycle, so dropped frames just
            # mean wasted work, not a desync.
            logger.debug("BridgePublisherIPC dropped frame %d (receiver busy)", self._frame_id)
            return

        self._frame_id += 1
        self._slot = (self._slot + 1) % self._ring_size
