# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Millimeter-wave radar fall detection prototype (毫米波雷达摔倒检测系统). PyQt5 desktop app simulating FMCW radar point clouds for human fall detection. Pure rule-based — no ML/deep learning. Python 3.8+, single package under `radar_fall_detection_demo/`.

## Commands

```bash
# Install dependencies (numpy, PyQt5, matplotlib)
pip install -r radar_fall_detection_demo/requirements.txt

# Launch GUI
python radar_fall_detection_demo/main.py

# Run evaluation (3 test cases: fall sequence, standing, walking)
python radar_fall_detection_demo/evaluate.py
```

No linter, type checker, or test framework is configured. The app requires a display (PyQt5 GUI).

## Architecture

Data flows one-way: `simulator → processor → detector → UI`

| Module | Core Classes | Responsibility |
|--------|-------------|----------------|
| `simulator.py` | `GeometricModel`, `RadarPhysicsModel`, `RadarPointCloudSimulator` | Physics-based radar point cloud generation using ellipsoid/cylinder body models, RCS reflection, miss-detection simulation, noise |
| `processor.py` | `PointCloudPreprocessor` | Filtering (passthrough, statistical/MAD, velocity, density) + feature extraction |
| `detector.py` | `FallDetectionModule` | 5-feature weighted fusion detection + pet filtering + post-fall confirmation |
| `skeleton.py` | `SkeletonEstimator` | Multi-frame fusion skeleton estimation (16 joints), bone length constraints, 1€ filter smoothing |
| `filters.py` | `OneEuroFilter` | Adaptive low-pass filter for temporal smoothing |
| `evaluator.py` | `Evaluator` | Quantitative evaluation (precision/recall/F1/FAR/detection delay) |
| `test_generator.py` | `TestSequenceGenerator` | Test sequence generation for evaluation |
| `main.py` | `MainWindow` | PyQt5 main window, event loop, system control |
| `ui_components.py` | `PointCloudVisualizer`, `SkeletonVisualizer`, `VisualLegend`, `ProgramGuide`, `RealTimePlot` | All visualization widgets |
| `config.py` | `HumanState` enum, config dicts | Centralized configuration management |
| `utils.py` | logging, export, screenshot | Utility functions |

### Key architectural rules

- **Point cloud and skeleton are fully decoupled.** The simulator (`GeometricModel`) never outputs skeleton data — skeleton is derived purely from point clouds via `SkeletonEstimator`. This is a hard boundary.
- Point clouds are 5D arrays: `[x, y, z, velocity_z, intensity]`, shape `(N, 5)`.
- `HumanState` enum has 4 states: `STANDING`, `WALKING`, `FALLING`, `FALLEN`.
- Centroid, skeleton, and trajectory all derive from the same `fused_centroid` (multi-frame buffer in `SkeletonEstimator`).
- Skeleton smoothing uses `OneEuroFilter` from `filters.py` with params `min_cutoff=0.5`, `beta=0.3`.

### Fall detection algorithm

5-feature weighted fusion — composite score ≥ 50% triggers fall, requires N consecutive frames (default N=2):

| Feature | Weight | Threshold |
|---------|--------|-----------|
| Human height | 30% | < 0.4m |
| Vertical velocity | 25% | < -0.6 m/s |
| Frame-to-frame height change rate | 20% | > 5 cm/frame |
| Velocity std deviation | 15% | < 0.3 |
| Point cloud density | 10% | > 20 pts/m² |

Pet filtering: multi-feature check (height < 0.5m, width < 0.4m, velocity std > 0.3, height std < 0.1).

Post-fall confirmation: after fall detected, must remain quasi-static for 5 frames before confirming alarm.

## Code Conventions

- **Python 3.8+ compatible** — use `typing.Tuple`, not `tuple[...]` for type hints.
- **Chinese docstrings and comments throughout** — maintain this convention.
- **Matplotlib embedded in PyQt5** via `FigureCanvasQTAgg` — never use `plt.show()`.
- **3D plot updates** use delete/recreate pattern, not `_offsets3d` assignment (causes flicker).
- Config values live in `config.py` as module-level dicts with getter/setter functions.

## Gotchas

- `ui_components.py` is the largest file (~69KB) — contains all visualization widgets.
- `alarms/` directory must exist at runtime for screenshot saving.
- `run.bat` sets `chcp 65001` and `PYTHONIOENCODING=utf-8` for Chinese output on Windows.
- `simulator.py`'s `generate_surface_points()` returns 3 values: `(points, rcs, velocities)`. The velocity pipeline flows through `simulate_missed_detection` → `apply_rcs_filter` → `add_measurement_noise`.
- RCS uses `cos²(incident_angle)` model per body part. Intensity uses radar equation `RCS/d⁴`.
- `self.position[2]` (Z offset) IS applied to point cloud coordinates — previous versions had a bug where only X/Y were offset.
