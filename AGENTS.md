# AGENTS.md

## Project Overview

PyQt5 desktop app simulating mmWave radar point clouds for fall detection. Pure rule-based (no ML). Single-package Python repo.

## Key Files

| File | Role |
|------|------|
| `main.py` | Entry point, MainWindow, event loop |
| `simulator.py` | Radar point cloud generation (GeometricModel, RadarPhysicsModel) |
| `processor.py` | Point cloud filtering and feature extraction |
| `detector.py` | Fall detection (state filter + 5-feature fusion + pet filter) |
| `ui_components.py` | All visualization widgets |
| `skeleton.py` | Skeleton estimation (SkeletonEstimator, multi-frame fusion, bone constraints) |
| `filters.py` | OneEuroFilter adaptive low-pass filter |
| `evaluator.py` | Evaluation framework (precision/recall/F1/delay) |
| `test_generator.py` | Test sequence generator for evaluation |
| `evaluate.py` | Evaluation script (run `python evaluate.py`) |
| `config.py` | All config dicts, HumanState enum (4 states), getters/setters |
| `utils.py` | Logging, export, screenshot utilities |

## Running

```bash
pip install -r requirements.txt
python main.py        # Launch GUI
python evaluate.py    # Run evaluation (3 test cases)
```

No linter or typecheck commands exist. The app requires a display (PyQt5 GUI).

## Architecture Rules

- Point cloud generation and skeleton estimation are **fully decoupled**. The simulator never outputs skeleton data — skeleton is derived purely from point clouds via `SkeletonEstimator` in `skeleton.py`.
- Data flow is one-way: `simulator → processor → detector → UI`.
- Point clouds are 5D arrays: `[x, y, z, velocity_z, intensity]` with shape `(N, 5)`.
- `HumanState` has 4 states: STANDING, WALKING, FALLING, FALLEN.
- Centroid, skeleton, and trajectory all derive from the same `fused_centroid` (multi-frame buffer in SkeletonEstimator). Centroid is additionally smoothed with a 5-frame sliding average in `MainWindow`.
- Keyboard shortcuts: Space=start/stop, S=stop, F=trigger fall, R=reset. State combo uses Chinese labels (站立/行走/摔倒中/倒地).

## Code Conventions

- Python 3.8+ compatible (type hints use `typing.Tuple`, not `tuple[...]`)
- No external dependencies beyond numpy, PyQt5, matplotlib
- Chinese docstrings and comments throughout — maintain this convention
- Matplotlib embedded in PyQt5 via `FigureCanvasQTAgg` — do not use `plt.show()`
- 3D plots use persistent artists with in-place `_offsets3d`/`set_data_3d` updates to eliminate flicker. See `_init_artists()` in visualizer classes.

## Gotchas

- `ui_components.py` is the largest file — contains PointCloudVisualizer, SkeletonVisualizer, VisualLegend, ProgramGuide, RealTimePlot, and CollapsibleSection.
- Skeleton smoothing uses OneEuroFilter from `filters.py`. Parameters: `min_cutoff=0.3`, `beta=0.1`. Multi-frame fusion buffer: 5 frames.
- Alarm screenshots save to `alarms/` dir — it must exist at runtime.
- `run.bat` sets `chcp 65001` and `PYTHONIOENCODING=utf-8` to handle Chinese output on Windows.
- `simulator.py` uses `generate_surface_points()` which returns 3 values: `(points, rcs, velocities)`. The velocity pipeline must flow through `simulate_missed_detection` → `apply_rcs_filter` → `add_measurement_noise`.
- RCS uses cos²(incident_angle) model per body part. Intensity uses radar equation `RCS/d⁴`.
- `self.position[2]` (Z offset) IS applied to point cloud coordinates — previous versions had a bug where only X/Y were offset.
