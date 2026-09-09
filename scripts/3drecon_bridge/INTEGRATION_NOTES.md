# Isaac Lab ↔ realtime-3d-reconstruction Bridge — Integration Notes

## Stable interactive defaults (2026-09-08)

Run `./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py`.
The runner enables cameras automatically, starts level, disables training
episode resets, and holds the robot until X+Y+A+B toggles engagement.
Look forward with the controllers in the robot's initial L-shaped arm pose
when engaging. Full SE(3) wrist clutching and a forward-view head rotation
reference prevent an initial pose snap. Left squeeze + A reclutches the hands.

The default world-parented gimbal follows the torso mount in translation
with a 0.25 s low-pass. While engaged, headset rotation relative to the engage
reference controls the view; headset translation never drives the camera.
Disengagement smoothly returns to the leveled torso heading. Head tracking
loss freezes the last camera orientation; fresh input is required to update
it. Controller loss captures a current hold once and reclutches on recovery.
`--no_xr_head_camera` also applies to the gimbal path.

Publication starts after mount initialization. The stereo host multipart
transport remains the default (the optional IPC ring has no consumer ack).
Left RGB/depth and right RGB must share a physics capture step; mismatched
pairs are dropped. Both rendered camera poses are checked against the
advertised rectified baseline. The published pose uses the left camera's
captured Fabric pose; missing exact poses drop the frame instead of falling
back to a potentially stale framework pose. The map is still rebased to the
first published optical camera frame with the existing X/Z axis remap.

The historical sections below describe earlier defaults and experiments;
this section supersedes their position-only clutch and pitched-camera defaults.

This document records the end-to-end design we converged on for hooking Isaac
Lab's G1 locomanipulation env into the `realtime-3d-reconstruction` Holoscan
app, including how two OpenXR/CloudXR clients coexist on one host. It is
intentionally code-adjacent: file paths and class names map directly to the
implementation.

## 1. Goal

Drive a G1 humanoid env in Isaac Lab from a VR headset (head + controllers),
feed the env's simulated camera output into nvblox in the
`realtime-3d-reconstruction` Holoscan app, and render the reconstructed mesh
back to the same headset. Three processes; one host; one HMD.

## 2. Processes and ownership

| Process | Host / container | Role | OpenXR? |
| --- | --- | --- | --- |
| `isaaclab_3d_recon` (C++ Holoscan app) | inside the realtime-3d-reconstruction Docker container | nvblox TSDF + mesh + Holoviz/XR render | **Foreground** OpenXR session (renders to HMD via CloudXR) |
| `run_g1_bridge_teleop.py` | host `env_isaaclab` venv | Isaac Lab sim, RGB-D publisher | **Standalone input-only** OpenXR client (no swap chain) |
| CloudXR runtime + HMD | host | OpenXR runtime + headset | n/a |

Two OpenXR clients connect to the same CloudXR runtime simultaneously. This is
the same pattern `gr00t/external_dependencies/isaac_teleop_app/groot_teleop`
already uses successfully against the same `teleop_3d_recon` Holoscan app.

## 3. The two-OpenXR-session story (the central design point)

### Why naively pairing Isaac Lab `--xr` with the container's `render_mode: xr` doesn't work

Isaac Lab's stock teleop path goes through
`isaaclab_teleop.IsaacTeleopDevice` → `TeleopSessionLifecycle`. Internally it
**piggy-backs on Kit's OpenXR session**:

```python
# isaaclab_teleop/session_lifecycle.py:534
oxr_handles = self._acquire_kit_oxr_handles(OpenXRSessionHandles)
```

If Kit isn't holding an OpenXR session (i.e. you didn't pass `--xr` to Kit),
`oxr_handles is None` and `_try_start_session` (lines 534–548) logs

> OpenXR handles not yet available (waiting for XR session); IsaacTeleop session creation deferred

and never makes progress, even though the very next comment on line 914 says

> omni.kit.xr.system.openxr not available; IsaacTeleop will create its own OpenXR session

The "create its own" fallback was planned but is not wired through the
lifecycle. So `IsaacTeleopDevice` is effectively glued to Kit-owned OpenXR.

If you *do* pass `--xr` to Kit, Kit then competes with the container's
`holoscan::XrSession` for the same CloudXR foreground session, which CloudXR
will not arbitrate — only one of them gets a usable swap chain.

### What we do instead

`gr00t/external_dependencies/isaac_teleop_app/groot_teleop/ros2_teleop_publisher_op.py`
shows the way out. It bypasses `IsaacTeleopDevice` and constructs the lower-
level session directly:

```python
session_config = TeleopSessionConfig(app_name="…", pipeline=pipeline, plugins=plugins)
self._session_ctx = TeleopSession(session_config)
self._session = self._session_ctx.__enter__()
```

When `TeleopSessionConfig` is built **without** `oxr_handles=…`, the underlying
`TeleopSession` calls `xrCreateInstance` / `xrCreateSession` itself. CloudXR
treats it as a *second* OpenXR client. Because it never creates a swap chain
(it only polls action/space data), it doesn't fight the rendering session for
the HMD display.

Our `scripts/3drecon_bridge/run_g1_bridge_teleop.py` does exactly the same in
`_StandaloneG1TeleopSession.try_connect()`. The realtime-3d-reconstruction
container keeps its existing `render_mode: xr` path — it owns the foreground
session and the HMD swap chain.

### Implications / hard rules

- **Do not pass `--xr` to the Isaac Lab runner.** Kit must stay out of
  OpenXR or it will hijack the runtime handles and the standalone path will
  conflict.
- **Start the container's `isaaclab_3d_recon` first**, so the foreground
  session is up and the HMD is rendering before the second client connects.
- **The retargeting pipeline must be the same on both sides of the bridge.**
  We reuse `_build_g1_locomanipulation_pipeline` from the upstream env config
  verbatim (`run_g1_bridge_teleop.py` imports it directly), so the action
  layout is identical to `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`.

## 4. End-to-end data flow

```
              CloudXR runtime + HMD
            ┌───────────────────────┐
            │      (host)           │
            └──┬───────────────┬────┘
   foreground  │               │ input-only client
   OpenXR      │               │ (xrCreateSession, no swap chain)
   (rendering) │               ▼
               │     ┌────────────────────────────────┐
               │     │ run_g1_bridge_teleop.py        │
               │     │ host venv (env_isaaclab)       │
               │     │                                │
               │     │  TeleopSession.step()          │
               │     │   ↳ head + controller poses    │
               │     │  retarget → 32-D action        │
               │     │                                │
               │     │  ManagerBasedRLEnv.step(action)│
               │     │   ↳ Kit + RTX render camera    │
               │     │                                │
               │     │  BridgeCamera                  │
               │     │   ↳ GPU buffers (torch)        │
               │     │   ↳ .cpu().numpy() → bytes     │
               │     │   ↳ ZMQ PUSH                   │
               │     └────────────────┬───────────────┘
               │                      │ tcp://127.0.0.1:5570
               │                      │ (host loopback, --network=host)
               │                      ▼
               │     ┌────────────────────────────────┐
               │     │ container: isaaclab_3d_recon   │
               │     │                                │
               │     │  IsaacLabBridgeReceiverOp      │
               │     │   ↳ ZMQ PULL                   │
               │     │   ↳ cudaMemcpyAsync H→D        │
               │     │                                │
               │     │  NvbloxOp                      │
               │     │   ↳ uint16 mm → float m kernel │
               │     │   ↳ integrateRgbd              │
               │     │                                │
               │     │  XrComposeOp → XrEndFrame ─────┘ (foreground OpenXR
               │     │   ↳ nvblox mesh to HMD            session)
               └─────│
                     └────────────────────────────────┘
```

Three I/O surfaces here, each with its own contract: OpenXR (twice), ZMQ
(once), CUDA (memcpy + kernel).

## 5. ZMQ wire format (Isaac Lab → Holoscan)

Endpoint: `tcp://127.0.0.1:5570`. PUSH on Isaac Lab side, PULL on the receiver
(`src/sensor/isaaclab_bridge_receiver.cpp`). `SNDHWM=1` and `RCVHWM=1` give
"drop newest if consumer is behind" semantics — the receiver always integrates
the freshest available frame, never a backlog.

Each frame is one ZMQ multipart message with three parts:

```
[ part 1: header  (140 bytes, packed) ]
[ part 2: rgb     (height * width * 3, uint8) ]
[ part 3: depth   (height * width * 2, uint16 little-endian, millimeters) ]
```

The header is exactly 140 bytes and mirrors `IsaacLabRgbdHeader` from
`src/sensor/isaaclab_bridge_receiver.cpp`. Python format string is
`<8sIIQqII5f16fQII`:

| Field | Type | Value / meaning |
| --- | --- | --- |
| `magic` | `char[8]` | `"ILRGBD1\0"` |
| `schema_version` | `uint32` | `1` |
| `header_size` | `uint32` | `140` (the sanity check value) |
| `frame_id` | `uint64` | monotonically increasing |
| `timestamp_ns` | `int64` | `time.monotonic_ns()` on publisher (informational) |
| `width`, `height` | `uint32` | image dimensions, pixels |
| `fx, fy, cx, cy` | `float32` ×4 | pinhole intrinsics in pixels |
| `depth_scale` | `float32` | `0.001` (meters per uint16 unit; informational) |
| `T_cam_world[16]` | `float32` ×16 | 4×4 column-major (see §6) |
| `reset_sequence` | `uint64` | bumped on publisher restart to invalidate cuVSLAM-SLAM keyframe cache downstream |
| `rgb_bytes`, `depth_bytes` | `uint32` ×2 | expected payload sizes (the receiver double-checks) |

The receiver rejects mismatched magic, schema, header size, or payload sizes
and skips the frame.

### Stereo variants (v3 host-staged, v4 CUDA-IPC)

When the scene declares a `stereo_role="right"` render-only camera (see
`locomanipulation_g1_bridge_env_cfg.py`), the publishing `"left"` camera ships
its RGB alongside so a consumer (e.g. the televiz `isaac_sim` source) gets a
time-aligned stereo pair + GT depth + GT pose:

* **v3 (host path, `use_cuda_ipc=False`)** — magic `"ILRGBD3\0"`, schema 3,
  header = the 140-byte v1 layout plus a trailing `baseline_m` float32
  (`header_size=144`, format `<8sIIQqII5f16fQIIf`), FOUR-part multipart:
  `header, rgb_left, rgb_right, depth`.
* **v4 (CUDA-IPC)** — magic `"ILRGBD4\0"`, schema 4. Handshake gains a
  trailing `baseline_m` float32 (40 bytes, `<8sIIIIIIIf`) and each ring slot
  carries THREE 64-byte mem handles (`rgb_left + rgb_right + depth`,
  192 bytes/slot). Per-frame header layout identical to v2 apart from
  magic/schema.

Mono scenes keep emitting v1/v2 byte-for-byte, so the existing Holoscan
receiver is unaffected. The right camera never publishes; it registers itself
(`bridge_camera._STEREO_REGISTRY`) and must be declared BEFORE the left camera
in the scene cfg so it renders first each step (the left camera verifies this
via a render-step counter and degrades to mono with a warning if violated).

### Stabilized camera mount (default; `--camera_stabilize_s`, 0 = rigid)

The Agile balance policy micro-sways the torso ~0.2 deg / ~0.6 mm EVERY step
(measured with `--pose_diag`; the mount-point lever arm makes it ~2.4 mm at
the camera). A torso-parented camera inherits that rigidly — ~3 px/frame of
image jitter at 1280 px. Two findings drove the fix:

1. A plain low-pass on the camera pose barely helps: the sway is a ~1 Hz
   oscillation, not per-frame noise, so tau=0.25 s attenuated rotation only
   0.21 -> 0.19 deg/step. Cranking tau would lag the camera when the robot
   walks.
2. What shakes the IMAGE is rotation; ~2 mm positional jitter is sub-pixel
   at scene distance.

So `_StabilizedCameraDriver` does what a real gimbal does: the cameras are
parented to the WORLD (no scene-graph coupling to the sway) and driven every
step to the torso mount point with the position low-passed (tau) and the
orientation LEVELED — roll/pitch dropped outright, only the torso yaw is
followed (smoothed), composed with the cfg mount rotation. Measured result:
0.21 -> **0.001 deg/step** (~200x); position ~1.9 mm/step residual
(sub-pixel). The XR head rotation, when engaged, passes through UNFILTERED
(it is a command, not sway). GT pose stays exact (leaf-prim readback), and a
warm-up drive before the first render keeps the pose-rebase anchor off the
world-parented spawn pose. `--keyboard`/`--demo` disable stabilization (they
own the camera and assume the torso-parented mount).

### XR-head-driven virtual camera

By default the runner drives the camera pair as a rotation-only virtual head
(`_XrHeadCameraController`; disable with `--no_xr_head_camera`): translation
stays rigidly bound to the torso through the plain scene-graph parenting
(the controller sets orientation-only via `positions=None` and does NOTHING
until the first valid head pose — re-setting the full pose per step from a
one-render-stale parent read double-differences the balance micro-motion
and visibly jitters the image), while rotation follows the raw XR headset
orientation. The raw head pose is not
exposed by the upstream pipeline (the anchor synchronizer's yaw comes from the
pelvis prim, not the headset), so `_build_pipeline_with_head()` wraps the
upstream pipeline with an isaacteleop `HeadSource` transformed by the same
`world_T_anchor` matrix (fed through its own external-input leaf
`world_T_anchor_head`), adding a `"head"` output to `session.step()`. The
OpenXR view-frame orientation (-Z forward, +Y up) is converted to the
world/ROS body convention via `_R_USDCAM_TO_BODY` before `set_world_poses`.
The published `T_cam_world` (GT pose) needs no extra plumbing: it is read back
from the moved leaf camera prim's Fabric matrix after each render, so it
always matches the XR-driven render pose.

### Teleop engage gate (default on; `--no_engage_gate` reverts)

The bare pipeline had three hazards: (1) no engagement — the robot followed
the controllers the moment the XR session connected (the official
`IsaacTeleopDevice` path gates via `DefaultTeleopStateManager` + buttons,
which our standalone session bypassed); (2) `Se3AbsRetargeter` is absolute
with no clutch, so the wrist target jumped by your-hand-vs-robot-hand
displacement at first tracked frame — and emitted its last pose
(initialized at the WORLD ORIGIN) whenever a controller was untracked,
yanking the arms toward the pelvis; (3) the locomotion retargeter has no
thumbstick deadzone and integrates hip height even while idle.

`_TeleopEngageGate` (runner) fixes all three. A parallel `ControllersSource`
("bridge_buttons") is exposed through the pipeline's OutputCombiner (the
`SessionLifecycle._button_controllers` idiom); the gate reads button edges +
grip validity from it and wraps every action:

* The engage chord **X+Y+A+B** (all four face buttons together;
  `--engage_chord`) toggles ENGAGED ↔ HELD; while held the robot gets a
  hold-pose action re-captured at its CURRENT configuration. A chord
  already held at connect never reads as a press (fail-safe edge init,
  like `DefaultTeleopStateManager`).
* On every engage / reset / tracking recovery, the gate captures
  `robot_wrist − target` and composes it onto all subsequent targets
  (`--engage_clutch pos` default: positions only; `full`: SE(3); `off`),
  so the first commanded pose IS the robot's current hand pose — no jump.
* Either controller untracked (grip invalid / group none) → auto-hold and
  re-clutch on recovery; engage presses are refused while untracked.
* The reset chord **left squeeze + right A** (`--reset_chord`): engaged →
  re-clutch + hip reset; held → re-capture the hold pose. Chord token
  syntax: x/y/a/b face buttons, [lr]squeeze, [lr]trigger, [lr]menu,
  [lr]stick. Overlapping chords resolve engage-first.
* Locomotion is recomputed from raw thumbsticks with `--stick_deadzone`
  (rescaled above it) using the stock retargeter's exact mapping/constants,
  and the gate owns hip-height integration (only while engaged).

## 6. Pose conventions (the easy thing to get wrong)

Isaac Lab and NvbloxOp use different conventions; the publisher does the
conversion.

- **Isaac Lab `CameraData.quat_w_world`** is the world-frame orientation of
  the sensor body, in the world-frame convention used by Isaac Sim
  (`+X` forward, `+Y` left, `+Z` up). Returned as `(x, y, z, w)`.
- **NvbloxOp `in_camera_pose`** is read as a 4×4 of 16 `float32` with
  `Eigen::Map<…, Eigen::ColMajor>` (`src/reconstruction/nvblox_op.cu:929-934`)
  and treated as `T_cam_world` — camera-from-world, with the camera in the
  OpenCV optical frame (`+Z` through the lens, `+X` right, `+Y` down).

The publisher's pipeline (`scripts/3drecon_bridge/bridge_publisher.py`):

1. `quat_xyzw_to_rotation_matrix(q)` → `R_world_body` (3×3).
2. `R_world_cam = R_world_body @ ROS_BODY_TO_OPTICAL`, where
   ```
   ROS_BODY_TO_OPTICAL = [[ 0, -1,  0],
                          [ 0,  0, -1],
                          [ 1,  0,  0]]
   ```
   maps body (X-fwd, Y-left, Z-up) → optical (Z-fwd, X-right, Y-down).
3. Assemble `T_world_cam`, take `np.linalg.inv` → `T_cam_world` (4×4 float32).
4. `np.asfortranarray(T_cam_world).ravel()` packs it column-major. The first
   12 floats are the rotation columns; translation is at indices 12, 13, 14
   (not 3, 7, 11).

The optical-frame correction is gated by
`BridgeCameraCfg.apply_optical_frame_correction` (default `True`). Flip it off
if you ever wire a consumer that expects the body convention.

## 7. Depth and color data conversions

### RGB

| Stage | Layout | Dtype | Memory |
| --- | --- | --- | --- |
| Isaac Lab `CameraData.output["rgb"]` | `(N, H, W, 3)` (or `(N, H, W, 4)` if renderer gave RGBA) | `uint8` | CUDA (Warp-backed torch view) |
| BridgeCamera `_extract_rgb_bytes` | strip alpha if present, make contiguous, `.cpu().numpy()` | `uint8` | host (D→H copy) |
| `tensor.tobytes()` | flat `H*W*3` bytes | `uint8` | host (H→H, zero-cost view of numpy buffer) |
| ZMQ wire | identical bytes | `uint8` | localhost loopback |
| `IsaacLabBridgeReceiverOp` GXF tensor | `(H, W, 3)` | `uint8` | CUDA (allocated via the receiver's `UnboundedAllocator`, populated with `cudaMemcpyAsync` H→D) |
| `NvbloxOp` ingestion | `(H, W, 3)` or `(H, W, 4)` | `uint8` | CUDA; D→D `cudaMemcpyAsync` into `nvblox::ColorImage`. If `color_channels == 4`, runs the `bgra_to_rgb` kernel instead. |

### Depth

| Stage | Dtype | Units | Memory |
| --- | --- | --- | --- |
| Isaac Lab `output["distance_to_image_plane"]` | `float32` | meters | CUDA |
| BridgeCamera mask + conversion | `valid = isfinite & (0 < d ≤ max_depth_m); d_mm = clamp(d*1000, 0, 65535).to(uint16)` | mm | still on CUDA |
| `.cpu().numpy().tobytes()` | `uint16` LE | mm | host |
| ZMQ wire | identical | mm | localhost loopback |
| `IsaacLabBridgeReceiverOp` GXF tensor | `uint16` | mm | CUDA (H→D `cudaMemcpyAsync`) |
| `NvbloxOp` kernel `depth_uint16_to_float` | `float32` | meters (×0.001 hard-coded) | CUDA (D→D in `depth_image_.dataPtr()`) |

We keep `uint16 mm` on the wire because (a) `NvbloxOp` is hard-coded to expect
that dtype (`nvblox_op.cu:940` casts directly to `uint16_t*`), (b) the
millimeter resolution is far below nvblox's `voxel_size_m: 0.01`, and
(c) it halves the per-depth-frame payload vs. float32.

### Copy counts per published frame at 848 × 480

| Hop | Direction | Bytes RGB | Bytes Depth |
| --- | --- | --- | --- |
| GPU → host (`.cpu()`) | D→H | 1,221,120 | 814,080 |
| numpy → bytes (`tobytes`) | H→H | view, ~0 | view, ~0 |
| ZMQ send (kernel buffer → loopback) | H→H | 1,221,120 | 814,080 |
| ZMQ recv → `std::vector<uint8_t>` | H→H | 1,221,120 | 814,080 |
| Receiver `cudaMemcpyAsync` | H→D | 1,221,120 | 814,080 |
| `depth_uint16_to_float` kernel | D→D | — | 1,628,160 (writes float32 dst) |

That's the price of the cross-process boundary. The CUDA-IPC variant we
sketched earlier collapses the host hops; see `INTEGRATION_NOTES_FOLLOWUPS`
at the bottom if/when we revisit it.

## 8. Files in this layout

### Isaac Lab side (`scripts/3drecon_bridge/`)

| File | Responsibility |
| --- | --- |
| `__init__.py` | Documentation only; the directory is on `sys.path` because each runner inserts it before any bridge import. |
| `bridge_publisher.py` | `BridgePublisher` (PyZMQ PUSH); 140-byte header packing; quaternion→rotation; `compute_t_cam_world` (with body→optical correction). |
| `bridge_camera_cfg.py` | `BridgeCameraCfg(CameraCfg)` — knobs: `endpoint`, `depth_data_type`, `rgb_data_type`, `publish_view_index`, `max_depth_m`, `apply_optical_frame_correction`, `publish_every_n_steps`, `rebase_to_initial_pose`, `invert_anchor_x_z`, `debug_window`. `class_type = "bridge_camera:BridgeCamera"`. |
| `bridge_camera.py` | `BridgeCamera(Camera)`. Overrides `_update_buffers_impl` to publish after the base render. **Also overrides `update(dt, force_recompute=True)`** so the renderer actually fires every step (no observation term reads this camera, otherwise it would never refresh). Handles pose rebasing (first frame → identity), the optional `invert_anchor_x_z` rotation, and the side-by-side cv2 preview window (`debug_window`). Lazy `BridgePublisher` creation on first publish. |
| `locomanipulation_g1_bridge_env_cfg.py` | Subclasses `LocomanipulationG1EnvCfg` + scene. Attaches the bridge camera to `/World/envs/env_.*/Robot/torso_link/bridge_camera` (G1 has **no** `head_link` body — see §14). Keeps `cfg.xr` from the upstream env so our `_build_external_inputs` can read `anchor_pos`; nulls `cfg.isaac_teleop` because we bypass `IsaacTeleopDevice`. |
| `run_g1_bridge.py` | Headless **non-XR** runner. Steps with a hold-pose action computed from current wrist body poses. Used for sanity-checking the camera → ZMQ → nvblox pipe without involving CloudXR. |
| `run_g1_bridge_teleop.py` | **The XR runner.** Builds the upstream `_build_g1_locomanipulation_pipeline()`, opens a standalone `isacteleop.TeleopSession`, retries until the HMD connects, computes `world_T_anchor` by literally instantiating the upstream `XrAnchorSynchronizer` (with a no-op `xr_core`) so the math is byte-identical to the `--xr` path, packages it via upstream's `XrAnchorManager._build_matrix`, pulls the 32-D action from `session.step()["action"]`, and passes it to `env.step()`. Also: sends a one-shot `re-center_xr_poses` ZMQ ping to the container's `XrComposeOp`, exposes `--left_quat_offset_rpy_deg` / `--right_quat_offset_rpy_deg` CLI flags for runtime tuning of the wrist-orientation Eulers, and prints the G1's full body-name dump on first reset for camera-placement debugging. Falls back to hold-pose when no XR client is connected. |
| `INTEGRATION_NOTES.md` | This document. |

### realtime-3d-reconstruction side

| File | Responsibility |
| --- | --- |
| `app/isaaclab_3d_recon.cpp` (new) | Minimal Holoscan `Application`: `IsaacLabBridgeReceiverOp` → `NvbloxOp` → either `XrComposeOp/XrBeginFrameOp/XrEndFrameOp` (XR) or `WindowedRenderOp` (windowed). Selects via `render_mode` in YAML. |
| `app/isaaclab_3d_recon.yaml` (new) | Defaults: `render_mode: xr`, bridge endpoint `tcp://0.0.0.0:5570`, nvblox parameters, XR compose config. |
| `app/CMakeLists.txt` (patched) | Adds `add_executable(isaaclab_3d_recon …)` linked against `holoscan::core`, `isaaclab_bridge_receiver`, `nvblox_op`, `render`. |
| `include/sensor/isaaclab_bridge_receiver.hpp`, `src/sensor/isaaclab_bridge_receiver.cpp` (pre-existing) | The ZMQ PULL operator + header struct + GXF tensor packing. We did not modify these — the wire format was already drafted; we built the publisher to match. |

## 9. Settings that matter (and the reasoning behind defaults)

### `BridgeCameraCfg`

| Key | Default | Why |
| --- | --- | --- |
| `endpoint` | `tcp://127.0.0.1:5570` | Container runs `--network=host`, so loopback is shared. |
| `data_types` | `["rgb", "distance_to_image_plane"]` | nvblox wants perpendicular depth, not Euclidean range. |
| `depth_data_type` | `"distance_to_image_plane"` | Same reason. |
| `update_period` | `0.0` (set in the scene cfg) | Mark outdated every render step. |
| `update_latest_camera_pose` | **`True`** (set in the scene cfg) | **Inherited from `CameraCfg`, defaults to `False`.** Must be `True` for any camera mounted on a moving articulation body — otherwise the parent's transform is snapshot at scene init and the camera sits there forever while the articulation walks past. Discovered the hard way. See §14. |
| `max_depth_m` | `65.0` | uint16 mm wire format caps at 65.535 m. Anything farther is set to 0 (nvblox treats 0 as no measurement). |
| `apply_optical_frame_correction` | `True` | NvbloxOp expects OpenCV optical convention. |
| `publish_every_n_steps` | `1` | Bridge publishes at the env's render rate. |
| `rebase_to_initial_pose` | `True` | First published `T_cam_world` is the identity; subsequent frames are camera-from-camera-at-t=0. Anchors the nvblox mesh's world origin at the robot's head pose when the session started, so the HMD view starts "at the robot's head" regardless of where the robot spawned in the sim world. |
| `invert_anchor_x_z` | `True` (currently) | After rebasing, right-multiplies a 180-degree-about-Y rotation so the anchor frame's axes become `+X = left, +Y = down, +Z = back`. Means the *first* published pose is `diag(-1, 1, -1, 1)`, **not** identity. Mutually exclusive with strict "first pose = identity"; pick whichever property you need. Flip to `False` to keep the optical convention (`+X = right, +Y = down, +Z = forward`) for the anchor frame. |
| `debug_window` | `True` | Pops up a single side-by-side cv2 preview window (`bridge-camera (sent to nvblox)`) with the RGB on the left and the colormapped (TURBO) uint16-mm depth on the right. Shows the exact bytes about to ship over ZMQ, so any publisher-side conversion bug is visible. Requires `opencv-python`. Self-disables with a warning if cv2 is missing or display is unavailable. Adds ~1-2 ms per frame on host CPU. |

### `BridgeCamera.update` override

`SensorBase.update` only marks buffers outdated. `_update_buffers_impl` is
called when `.data` is read. Our env's observation manager does not include
this camera, so without the override the camera would never render. We
unconditionally pass `force_recompute=True`.

### Runner correctness knobs

- `torch.no_grad()` wraps the entire sim loop. The upstream
  `AgileBasedLowerBodyAction` runs `agile_locomotion.pt` (a TorchScript
  policy) whose outputs carry `requires_grad=True`. Warp's
  `__cuda_array_interface__` rejects such tensors with
  `RuntimeError: Can't get __cuda_array_interface__ on Variable that requires grad`.
  `torch.no_grad()` keeps the policy's forward pass out of the autograd
  graph.
- `_build_hold_pose_action` uses `.detach().clone()` on the slices read from
  `robot.data.body_pos_w` / `body_quat_w` because those buffers are Warp
  views and may also be grad-tracked.
- `--enable_cameras` is required by Kit whenever a `Camera` (or
  `BridgeCamera`) sensor exists in the scene. The non-XR runner errors
  loudly without it; the XR runner needs it for the same reason.

### `isaaclab_3d_recon.yaml`

The currently checked-in default is:

```yaml
render_mode: xr
isaaclab:
  endpoint: "tcp://0.0.0.0:5570"
  timeout_ms: 2000
  drop_policy: "latest"
nvblox:
  voxel_size_m: 0.01
  max_integration_distance_m: 4.0
  fast_rgbd_integrator: true
  use_dynamic_tsdf: true
  …
```

`render_mode: xr` is the production setting for the two-OpenXR-session flow.
For initial bring-up (no HMD), switch to `render_mode: windowed`.

## 10. Operational order

1. Bring up the CloudXR runtime + pair the headset.
2. In the realtime-3d-reconstruction container:
   ```bash
   ./app/isaaclab_3d_recon ../app/isaaclab_3d_recon.yaml
   ```
   Confirm `[IsaacLabBridgeReceiverOp] listening on tcp://0.0.0.0:5570` and
   that the foreground XR session is alive (HMD shows the empty Holoviz/XR
   layer).
3. On the host:
   ```bash
   ./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --enable_cameras
   ```
   The runner will print `waiting for XR client to connect`, retry every
   2 s, then flip to `TeleopSession connected (standalone OpenXR client)`.
   `--xr` must NOT be passed.
4. Verify the loop:
   - Host: `[bridge-camera] _update_buffers_impl count=…` and bridge
     heartbeats.
   - Container: `[IsaacLabBridgeReceiverOp] received frame=… size=WxH
     reset_sequence=…`.
   - HMD: the nvblox mesh accumulating as the simulated G1 moves.

The non-XR runner (`run_g1_bridge.py`) and `render_mode: windowed` are the
fall-back combination for verifying the bridge without involving CloudXR.

## 11. Things we deliberately did not modify

- **Isaac Lab upstream code.** All bridge code lives under
  `scripts/3drecon_bridge/` and consumes upstream APIs only. The
  `_try_start_session` gap in `isaaclab_teleop/session_lifecycle.py` is left
  untouched. A clean upstream fix (about 15 lines: add
  `standalone_xr_session: bool` to `IsaacTeleopCfg`, branch in
  `_try_start_session` to call `TeleopSession(session_config)` without
  `oxr_handles` when set) would let `IsaacTeleopDevice` work standalone and
  remove the need for our bespoke runner. Worth contributing back if NVIDIA
  is open to it.
- **`IsaacLabBridgeReceiverOp` schema.** The v1 header was already drafted
  in `src/sensor/isaaclab_bridge_receiver.cpp`; we matched it from the
  Python side rather than evolve it. A v2 with CUDA IPC (zero host
  round-trip) was sketched but not adopted, because (i) the v1 host-staged
  copy is fast enough at this resolution and (ii) v1 is wire-compatible
  with a future dummy/test publisher.

## 12. Known limitations / follow-ups

- **Anchor pipeline status.** `_build_external_inputs` now literally
  instantiates the upstream `XrAnchorSynchronizer` (with a no-op `xr_core`
  to suppress the `set_world_transform_matrix` call we don't need) and reads
  the cached `(pos, quat_xyzw)` it produces, then passes that to
  `XrAnchorManager._build_matrix` for the final 4×4. So the `world_T_anchor`
  math — including `xr_cfg.anchor_pos` offset, `fixed_anchor_height`, the
  FOLLOW_PRIM_SMOOTHED yaw delta with Slerp, and the `_OXR_TO_USD_ROTATION`
  basis change — is byte-identical to what `teleop_se3_agent.py --xr`
  computes. The only behavioral diff: we don't drive Kit's renderer (we
  don't need to; the container's `XrComposeOp` owns the foreground
  CloudXR swap chain instead).
- **Single-view bridge.** The wire format carries one camera. Multi-view
  (head + wrist cams) would require a `camera_id` field in v2 of the
  header.
- **Pose pipeline assumes monotonic frame ids.** `reset_sequence` is bumped
  by `BridgePublisher.bump_reset_sequence()`, but the runner does not call
  it on env reset today. If you start resetting the env mid-session, wire
  that bump.
- **CloudXR multi-client behavior is empirical, not specified.** The
  arrangement works against CloudXR today (gr00t already runs this way),
  but it's not part of the OpenXR core spec. If you upgrade the CloudXR
  runtime, re-verify that the standalone session can still attach while a
  foreground session holds the swap chain.

## 13. Lessons learned the hard way

Gotchas we tripped over during bring-up, captured here so future-you (or
future-someone-else) doesn't.

### `body_quat_w` is xyzw on this branch — never swizzle it

This Isaac Lab branch changed the quaternion convention REPO-WIDE from wxyz
to xyzw (commit `9659a5ce`): `body_quat_w`, `matrix_from_quat`, and the Pink
IK action's quat slices are all `(x, y, z, w)`. The hold-pose builder
originally applied a wxyz→xyzw swizzle to `body_quat_w` — a cyclic
permutation of an already-xyzw quat is a large, orientation-dependent bogus
rotation, so both wrist targets were commanded heavily twisted (arms cranked
to one side, IK fighting the waist tasks, whole robot shaking) whenever the
hold action was applied. Related trap in the same function: the locomotion
slice's hip height is an ABSOLUTE command (clamped 0.4–1.0 downstream) —
zeroing it orders a full crouch. Hold actions must carry hip 0.72. Also note
the Pink IK action expects positions in the ENV-ORIGIN frame (it subtracts
`env_origins` from the base pose only), so `body_pos_w` needs the env origin
subtracted (a no-op at num_envs=1).

### G1 has no `head_link` body in the articulation

The Unitree G1 URDF/USD does **not** include a `head_link` as part of the
articulation chain. The articulation stops at the shoulders (z≈1.04 m world,
spawn pose) and the highest body in the central spine is `torso_link` at
z≈0.794 m. The head visible in the rendered scene is a static visual mesh
attached **under** `torso_link`, not its own articulated body. So:

- The original guess `prim_path="/World/envs/env_.*/Robot/head_link/bridge_camera"`
  was an invalid path; Kit didn't error, just silently anchored the camera
  to a fallback location (effectively static at world origin).
- The correct parent is `torso_link`, with a positive `offset.pos[2]`
  (~0.5 m) to land the camera at head visual height. The camera then follows
  the waist/torso articulation correctly.

**Always verify the parent body exists** before tuning offsets:

```python
robot = env.scene["robot"]
print(robot.body_names)
```

The runner's startup diagnostic does this dump automatically.

### Wrong `prim_path` is a silent failure

Related to the above. If `prim_path` references a parent prim that doesn't
exist, Kit doesn't raise an error — it spawns the camera at some fallback
location (typically under the env's USD scope at world origin) and you only
notice because the camera image is static while the robot moves around it.
There is no log line marking the failure. **Always run with `--viz kit`
during bring-up and verify in the Kit Stage panel that the camera prim is
actually a child of the body you expected.**

### `CameraCfg.update_latest_camera_pose` defaults to `False`

Inherited from upstream `CameraCfg`. With the default, the camera's world
transform is captured **once, at scene initialization**, and the renderer
reuses that cached value forever. The camera prim's position never tracks
its parent's motion. For a camera mounted on an articulation body that
should follow the body's motion, this flag must be set to `True`. The cost
is a per-step `FrameView` traversal — negligible at one camera, but the
reason the upstream default is `False` (most upstream examples use static
observer cameras anchored at world).

### Default `OffsetCfg` convention pitfall

`convention="opengl"` uses Kit-native camera local axes (`-Z` forward,
`+Y` up). It's the right choice when you author the offset in Kit's
viewport convention. But if you write the offset by reasoning about the
robot's body frame (X forward, Y left, Z up), use `convention="world"` —
that aligns the camera's forward axis with the parent prim's `+X` and up
with `+Z`, which is what every robotics-flavored frame in Isaac Lab uses.
We had a bug for a stretch where `convention="opengl"` was paired with a
rotation that was actually designed for the optical convention — the result
was a camera looking 180° backward, upside down.

### Two flags that are jointly inconsistent

`rebase_to_initial_pose=True` AND `invert_anchor_x_z=True` cannot both
satisfy "first pose is identity." Mathematically, the rebasing wants
`T_cam_anchor(0) = I`, while the invert flag rotates the anchor frame's
axes by 180° about Y — making the first pose `diag(-1, 1, -1, 1)` instead.
Pick whichever property you care about more; the cfg docstrings call this
out explicitly.

### `--enable_cameras` is mandatory whenever cameras exist

Even if you're not using the camera output, the moment a `Camera` (or
`BridgeCamera`) sensor lives in the scene, the `--enable_cameras` flag
must be passed or Kit refuses to spin up the RTX render. The error is
loud and obvious; just a reminder that the bridge env always needs it.

### Isaac Lab quaternion convention is `(w, x, y, z)`, not `(x, y, z, w)`

`robot.data.root_quat_w` and `robot.data.body_quat_w` return quaternions
in `(w, x, y, z)` order (scalar-first). scipy, ROS, OpenXR, and most other
libraries use `(x, y, z, w)` (scalar-last). Easy to swap and not notice
until rotations look subtly wrong. Our `_build_external_inputs` and
`_build_hold_pose_action` both do the unpack-and-reorder explicitly.

### Headset stage-origin convention varies per HMD

Some HMDs (e.g. Pico in standalone mode) set the OpenXR stage origin at
the user's *floor* during room-scale setup; others (some seated-mode
configurations) set it at the user's *head* at calibration time. The
`xr_cfg.anchor_pos.z = -0.95` calibration in the upstream G1 env is tuned
for standing mode at typical height. If the user's HMD reports stage origin
at head height instead of floor, the anchor offset over-corrects and the
wrist IK targets land below the ground. Fix: either rerun the HMD's room
setup in standing mode, or add a CLI flag exposing `anchor_pos.z` for
runtime tuning.

### Per-HMD controller-grip-to-wrist Euler offsets

The upstream `_build_g1_locomanipulation_pipeline` has hand-tuned
`target_offset_roll/pitch/yaw` Euler offsets per wrist (lines 78-108 of
`locomanipulation_g1_env_cfg.py`). Those numbers were calibrated for a
specific controller-grip convention. Different HMDs (Quest grip vs Pico
grip vs Vive wand) can disagree about which controller-local axis is
"forward" by 90° or 180°, so the upstream values may not match your HMD
out of the box. Without us touching upstream, the bridge runner exposes
`--left_quat_offset_rpy_deg` and `--right_quat_offset_rpy_deg` flags that
right-multiply a corrective rotation onto the final wrist IK target
quaternion — a fast way to sweep values until orientation aligns.

### Don't pass `--xr` to the runner

Bears repeating, because every other Isaac Lab teleop example uses it:
**this runner must never see `--xr`**. We want Kit's `omni.kit.xr.system.openxr`
to stay inert so that our standalone `TeleopSession` is the only OpenXR
client on the Isaac Lab side. The `--enable_cameras` flag is required;
`--viz kit` is optional (for the desktop viewport).

## 14. Quick file map for future-you

```
realtime-3d-reconstruction/
  app/isaaclab_3d_recon.cpp           # new: receiver → nvblox → render
  app/isaaclab_3d_recon.yaml          # new: render_mode: xr by default
  app/CMakeLists.txt                  # patched: add executable
  src/sensor/isaaclab_bridge_receiver.cpp   # pre-existing; reference for wire format
  include/sensor/isaaclab_bridge_receiver.hpp

IsaacLab/
  scripts/3drecon_bridge/
    bridge_publisher.py               # PUSH socket, header packing, pose math
    bridge_camera_cfg.py              # CameraCfg subclass
    bridge_camera.py                  # Camera subclass, force-recompute, publish
    locomanipulation_g1_bridge_env_cfg.py  # env subclass + bridge camera
    run_g1_bridge.py                  # non-XR runner (hold-pose, debug)
    run_g1_bridge_teleop.py           # XR runner (standalone TeleopSession)
    INTEGRATION_NOTES.md              # this file
```
