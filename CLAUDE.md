# CLAUDE.md

## What This Is

FTCN (Fully Temporal Convolution Network) — video deepfake detection. Two-stage pipeline:
1. **FTCN**: I3D-based ResNet with spatial conv kernels reduced to 1×1 (temporal-only convolutions)
2. **Temporal Transformer**: attends to long-range temporal coherence across the clip

Inference only (training code not yet released). Pretrained checkpoint: `checkpoints/ftcn_tt.pth`.

## Environment

Uses devenv + uv. Enter the shell before running anything:

```bash
devenv shell
```

All Python deps are managed via `uv` (see `pyproject.toml`). PyTorch is pinned to CPU wheel index by default — GPU inference works if CUDA is available at runtime.

## Running Inference

```bash
python test_on_raw_video.py <input_video> <output_dir>
```

Example:
```bash
python test_on_raw_video.py examples/shining.mp4 output
```

Output is an annotated `.avi` in `<output_dir>`. Detection results are cached as `<video>_768.pth` alongside the input file to skip re-detection on repeated runs.

## Config System

Layered YAML config (load order matters — later layers override earlier):
1. `root_setting.yaml` — base defaults (loaded via `cfg.init_with_yaml()`)
2. `setting/<name>.yaml` — experiment-specific overrides (e.g. `ftcn_tt.yaml`)
3. Manual overrides via `cfg.update_args()`
4. `cfg.freeze()` locks the config (access after freeze raises `AttributeError`)

`config.py` implements `AttrDict` — a freezable nested dict accessible as attributes. Config is a module-level singleton `config` imported as `cfg`.

Active model config for inference: `setting/ftcn_tt.yaml` → `classifier_type: i3d_temporal_var_fix_dropout_tt_cfg`, `clip_size: 32`, `imsize: 224`.

## Architecture

### Model Loading

`PluginLoader` (`utils/plugin_loader.py`) dynamically imports by name:
- `PluginLoader.get_classifier("i3d_temporal_var_fix_dropout_tt_cfg")` → imports `model.classifier.i3d_temporal_var_fix_dropout_tt_cfg` and returns the `Classifier` class

### Classifier

`model/classifier/i3d_temporal_var_fix_dropout_tt_cfg.py`:
- `I3D8x8`: wraps SlowFast's `ResNet` (from `slowfast/`), calls `temporal_only_conv()` at init to replace all spatial conv kernels with 1×1
- `TransformerHead` replaces the ResNet head; pools the 3D feature map then feeds patches to `TimeTransformer`
- `Classifier(ClassifierBase)`: thin wrapper that exposes `load()` / `save()` via `_classifier_base.py`

### Inference Pipeline (`test_on_raw_video.py`)

1. `detect_all()` → RetinaFace detection + 68-point landmark prediction per frame (batched in chunks of 50)
2. `multiple_tracking()` → IoU-based multi-face tracking across frames → "super clips"
3. Short clips padded via palindrome reflection; clipped into `clip_size`-length windows
4. `FasterCropAlignXRay` → crops + aligns faces per clip using landmarks
5. Classifier forward pass; per-frame predictions averaged across overlapping clips
6. `SupplyWriter` renders bounding boxes + fake probability score onto output video

### Test Tools

- `test_tools/ct/detection/` — RetinaFace face detector (`FaceDetector`)
- `test_tools/ct/face_alignment/` — 68-point landmark predictor (`LandmarkPredictor`)
- `test_tools/ct/tracking/` — IoU tracker used by `multiple_tracking()`
- `test_tools/faster_crop_align_xray.py` — face crop + alignment using affine transform to canonical landmarks

### SlowFast

`slowfast/` is a local copy of Facebook's SlowFast codebase, used only for `slowfast.models.video_model_builder.ResNet` and `slowfast.config.defaults.get_cfg`. Do not modify.

## Gotchas

- **Cache invalidation**: delete `<video>_768.pth` to force re-detection (stale after face/landmark code changes)
- **Checkpoint**: must exist at `checkpoints/ftcn_tt.pth` before inference; no auto-download
- **GPU vs CPU**: inference is functional on both; CPU is ~10–20× slower but produces identical results
