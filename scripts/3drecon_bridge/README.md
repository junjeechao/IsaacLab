# Isaac Lab stereo RGB-D teleop bridge

## Current status — 2026-09-09

Parked development checkpoint on `3drecon-bridge`. The user tested the current
bridge with gr00t televiz and reports that it works reasonably. This is a
working integration example, not a guarantee of jitter-free operation on every
headset/runtime. Eight CPU regression tests and a three-step GPU/OpenXR smoke
test passed during development; sustained on-device acceptance remains manual.

The tested consumer implementation is gr00t `junjeec/sim-source` at
`afb778b5d77`. The separate IsaacTeleop example
`examples/realtime_3d_reconstruction` did **not** have an `isaac_sim` source
adapter at inspection time (`junjeec/3drec`); it cannot consume this stream
without an adapter. Do not confuse that Python example with the older C++
Holoscan `isaaclab_3d_recon` receiver described in the historical notes.

This README describes current defaults. [INTEGRATION_NOTES.md](INTEGRATION_NOTES.md)
contains detailed history, earlier defaults, experiments, and legacy launch paths.

## Launch

Prerequisites: working Isaac Lab/Isaac Sim GPU installation, the `env_isaaclab`
environment with `isaacteleop`, NumPy, SciPy, torch and pyzmq, and a configured
CloudXR OpenXR runtime. Follow the official
[CloudXR setup guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/cloudxr_teleoperation.html)
for runtime/client setup. This bridge does not start the CloudXR server itself.

Start the downstream XR renderer and connect the headset, then start the bridge.
The bridge retries its input session if XR is initially unavailable.

```bash
# IsaacLab terminal, with env_isaaclab already activated:
cd ~/work/IsaacLab
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py
```

Cameras are enabled automatically; headless is the Isaac Lab default. Do **not**
pass `--xr`: the downstream renderer owns the foreground OpenXR rendering session.
The bridge opens a separate input-only session. This arrangement depends on the
runtime supporting those simultaneous sessions; it is not portable to every
OpenXR runtime merely because a server is running.

In a checkout containing televiz's `junjeec/sim-source` implementation:

```bash
cd ~/work/gr00t/groot/core/robotics/teleop/televiz
./televiz.sh run configs/sim_reproject_gt.yaml --mode xr
# Alternative: stereo quad display with ground-truth pose
./televiz.sh run configs/sim_quad_gt.yaml --mode xr
# Non-XR preview (no headset rendering)
./televiz.sh run configs/sim_reproject_gt.yaml --mode window
```

Choose one consumer command. These are not instructions to switch a busy gr00t
checkout: use an existing appropriate checkout or a separately prepared worktree.
The `sim_quad_cuvslam.yaml` and `sim_reproject_infer.yaml` presets exercise
perception instead of ground-truth pose/depth and require their respective dependencies.

Useful bridge commands:

```bash
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --help
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --viz kit
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --pose_diag
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --max_steps 3 --heartbeat_every 1
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --no_xr_head_camera
./isaaclab.sh -p scripts/3drecon_bridge/run_g1_bridge_teleop.py --keyboard
```

`--bridge_endpoint tcp://127.0.0.1:5570` overrides the publisher connection.
Set the consumer's `source.isaac_endpoint` to a matching bind endpoint. Only one
environment is supported. `run_g1_bridge.py` is a legacy non-teleop runner and
does not share all the stable interactive defaults above.

## Operator controls and camera behavior

- **X+Y+A+B:** toggle engage/hold. At engage, look forward and hold controllers
  in the robot's initial L-shaped arm pose, pointing forward.
- **Left squeeze + A:** re-clutch wrists while engaged; recapture hold while idle.
  This does not reset the simulation or recenter the downstream renderer.
- Wrist targets use the upstream G1 retargeting pipeline and full SE(3) clutching
  by default. Initial controller-to-robot pose offsets are absorbed. Full clutch
  also rotates subsequent translation deltas; `--engage_clutch pos` preserves
  absolute translation axes but permits an orientation snap.
- Thumbsticks control locomotion with a 0.15 deadzone. Hip-height integration
  occurs only while engaged. Tracking loss captures a hold once and reclutches
  on recovery; invalid/nonfinite actions are rejected.
- The camera pair is world-parented. Translation follows the torso mount
  through a 0.25 s first-order low-pass, independently of headset translation.
- While engaged, headset rotation is referenced to the camera orientation at
  engage. Head loss freezes rotation; recovery re-seats the rotation reference.
  While held, the camera returns smoothly to the leveled torso heading.
- Cameras start level (`--camera_pitch_down_deg 0`). `--no_xr_head_camera`
  disables head following, including in the default gimbal path.
- `--camera_stabilize_s 0` selects the older rigid torso mount; it inherits
  balance-policy sway and does not have all the default gimbal semantics.
  Keyboard/demo paths also disable the gimbal.
- Training termination/reset conditions are disabled for continuous teleop.
  Stop and restart the runner when an actual environment reset is needed.

## Design and data ownership

```text
CloudXR runtime + headset
  | tracking                             ^ headset images
  v                                      |
Isaac Lab input-only TeleopSession    downstream XR renderer
  | upstream G1 pipeline + engage gate    ^
  v                                      |
G1 physics + torso/head camera driver     |
  |                                      |
RTX stereo RGB + left depth + pose -> ZMQ receiver -> reconstruction/reprojection
```

The receiver does not command the robot or the bridge camera. The bridge uses
the upstream pelvis anchor and OpenXR-to-Isaac conversion for tracking. Its
custom engagement gate is independent of the downstream app's start/stop UI.
Legacy `--xr_recenter_*` arguments remain, but automatic external recentering
is commented out; the downstream renderer owns its own view alignment.

| File | Responsibility |
| --- | --- |
| `run_g1_bridge_teleop.py` | Launch, session retry, retargeting, engagement, camera driving |
| `locomanipulation_g1_bridge_env_cfg.py` | G1 scene and rectified stereo camera pair |
| `bridge_camera.py` | Capture validation, exact camera-pose snapshot, packaging inputs |
| `bridge_camera_cfg.py` | Transport and camera options |
| `bridge_publisher.py` | Default host-memory ZMQ multipart protocol |
| `bridge_publisher_ipc.py` | Optional experimental CUDA-IPC protocol |
| `test_teleop_stability.py` | CPU regressions without launching Kit |

## Default transport: host CPU memory over ZMQ/TCP

**Yes: the default path copies GPU images into CPU memory and sends their bytes
over TCP using ZeroMQ. It is neither CPU shared memory nor zero-copy CUDA IPC.**
The receiver then uploads images to its GPU as needed.

The bridge **connects** a PUSH socket; the consumer **binds** a PULL socket at
`tcp://127.0.0.1:5570`. A stereo capture is one four-part multipart message:

```text
[144-byte header] [left RGB bytes] [right RGB bytes] [left depth bytes]
```

- Magic `ILRGBD3\0`, schema 3; little-endian header format
  `<8sIIQqII5f16fQIIf`.
- Default images: 1280×720; rectified pinhole pair; 0.063 m baseline.
- RGB: contiguous H×W×3 uint8, RGB channel order.
- Depth: H×W uint16 little-endian millimeters; zero is invalid.
  It is optical-axis depth (`distance_to_image_plane`), not radial range.
- Header: frame ID, timestamp, dimensions, `fx/fy/cx/cy`, depth scale,
  16 float32 pose entries, reset sequence, payload sizes, and stereo baseline.
- Timestamp currently defaults to publisher `time.monotonic_ns()` when packing,
  after readback. It is a same-host join identifier, **not** a hardware exposure
  timestamp or an OpenXR predicted-display timestamp.
- About 7.37 MB of image payload per frame; approximately 221 MB/s at 30 FPS,
  before transport overhead. Actual throughput depends on rendering/readback.
- Nonblocking sends and a small send high-water mark limit queue growth.
  A full queue drops the new send, not necessarily the oldest queued frame.
  Low queue depth is not a guarantee of zero latency.
- PUSH/PULL is **not broadcast**: multiple PULL consumers split frames.
  Run one downstream app at a time, or add explicit fan-out for simultaneous use.

Host mono v1 (`ILRGBD1`, three parts) remains available for mono configurations.
The current stereo scene drops an incomplete pair instead of silently publishing mono.

### Synchronization and pose convention

Right camera is declared before left. Isaac RTX deduplicates rendering across
cameras for a simulation render epoch; RGB and depth come from the left camera's
annotators. The bridge checks matching physics capture steps and rendered stereo
extrinsics (parallel orientations and the advertised baseline), then packages
the immutable host payloads together. Missing exact poses or mismatched pairs
drop the capture. This is intended simulation-state synchronization, not physical
camera exposure synchronization.

The pose is captured from the **left camera leaf prim in Fabric after rendering**,
including the virtual gimbal/head rotation. It is not reconstructed from an old
torso pose. Published coordinates are rebased, not absolute Isaac world coordinates:

```text
T_wire(t) = T_camera_from_simworld(t)
          @ inverse(T_camera_from_simworld(first_capture))
          @ diag(-1, 1, -1, 1)
```

The camera frame is optical (+X right, +Y down, +Z forward). The wire matrix is
camera-from-map, flattened column-major (`order="F"`). A consumer wanting
map-from-camera must invert it. The first wire pose is the X/Z flip matrix,
not identity. Publication starts after camera mount initialization. Restarting
the bridge establishes a new map origin; restart/reinitialize downstream mapping
too. Do not assume the current reset-sequence field robustly identifies every
process restart. Nonzero startup pitch tilts the rebased map; do not assert
gravity alignment in the consumer in that case.

### Optional CUDA IPC

`BridgeCameraCfg.use_cuda_ipc=True` selects v2 mono/v4 stereo GPU ring buffers;
there is currently no CLI switch. ZMQ then carries handles/metadata rather than
all pixels. **This path is experimental:** there is no consumer acknowledgement
before ring-slot reuse, so delayed consumers can read overwritten pixels with an
older pose header. Immediate consumer copies reduce but do not eliminate the race.
Use the default `False` for the tested coherent host transport. Keep both
processes on compatible CUDA devices/runtimes if investigating IPC.

## Verification and remaining work

```bash
cd ~/work/IsaacLab
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest \
  scripts/3drecon_bridge/test_teleop_stability.py -q
git diff --check
```

Disabling pytest plugin autoload avoids unrelated ROS plugin dependencies.
The tests cover invalid controller input, hold/reclutch recovery, full wrist
clutching, stereo capture-step checks, engage-time camera alignment, head-only
rotation, stereo baseline and head tracking recovery. They do not replace a
sustained headset/receiver test.

Remaining work:

- Add an `isaac_sim` receiver and GT passthrough stages to the separate
  IsaacTeleop reconstruction example; test it independently.
- Measure end-to-end pose age, FPS, jitter, and prolonged disconnect/reconnect
  behavior. Camera commands currently update once per environment step.
- Add wire-level live capture tests and robust restart/reset epochs.
- Add consume acknowledgement/back-pressure before endorsing CUDA IPC.
- Clarify/remove obsolete recenter controls via a compatibility/deprecation path.
- Split the large runner into session, gate, and camera modules; consolidate
  legacy rigid-mount and keyboard/demo behavior with the tested defaults.
- Review quantitative image quality (AA, depth edges, disocclusion) separately
  from pose correctness. Optional `--no_aa` and depth-edge masking are diagnostics.
