https://doi.org/10.5281/zenodo.23908546
# EdgeAcousticsDL — TinySoundMTL-Net

Reproducible research implementation of supervised joint acoustic anomaly detection and machine-type classification from *Lightweight Multitask Deep Learning for Joint Acoustic Anomaly Detection and Machine Type Classification on Embedded Linux Devices*.

## Implemented

- 16-kHz, first-channel WAV ingestion; 1-s/0.5-s log-Mel windows (128 × 32); deterministic train-only per-band normalization.
- Recording-level, machine-ID × class × condition stratification and leak checks; held-out machine ID (LOMO) and machine-type fault-family surrogate (LOFO).
- Depthwise-separable CNN → three dilated causal residual TCN blocks → additive temporal attention → separate binary/4-class heads.
- Supervised multitask training, patience-based stopping, five seeds, saved checkpoints, validation-only anomaly threshold.
- Recording-level classification metrics, visualizations at 600 dpi, checkpoint evaluation, repeated-run statistics.
- Classical comparators; CNN/CNN-LSTM/CNN-GRU/CNN-attention and mobile neural comparators; backbone/attention/TCN/task ablations.
- Magnitude-based weight pruning; portable ONNX and optional ONNX Runtime QDQ INT8 post-training quantisation.
- Streaming 5-window aggregation, alert logs, WAV replay, optional microphone, CPU timing and memory benchmarking.
- Synthetic end-to-end smoke test and automated tests.

## Scientific and engineering limitations

This repository is an independent PyTorch implementation, whereas the manuscript describes TensorFlow/Keras. PyTorch was selected to enable executable smoke testing in the present environment. Shapes and core operations follow the manuscript, but exact numerical equivalence, parameter count and model performance are not guaranteed. The intermediate channel sizes were unspecified and are fixed in `configs/default.json`; the default backbone has 2,010,199 parameters (approximately 2.01 M), including both heads. The parameter count is verified programmatically. The reported 97.26% and 98.14% accuracies are manuscript claims, not precomputed repository results.

Export difference: ONNX Runtime static INT8 QDQ quantisation is offered instead of TensorFlow Lite full-integer conversion. Quantised weights/operators are validated by the ONNX exporter/runtime; the file can retain float input/output tensors. Consequently it must not be described as the manuscript's full INT8 TFLite implementation. A compatible ONNX Runtime build is required on Raspberry Pi. Because unstructured pruning does not remove dense operator dimensions, it does not automatically reduce runtime or model size. EfficientNet-B0 is used as a declared surrogate when `efficientnet_lite` is requested, not as actual EfficientNet-Lite. Such results must be labelled as a surrogate and cannot be reported as EfficientNet-Lite results.

LOFO removes anomalous sounds from one machine type during training/validation; it is not individual fault-mechanism LOFO, because public MIMII lacks fault-type labels. Synthetic audio is for smoke testing only and must never be reported as research performance. Runtime CPU metrics measured on a workstation cannot be called Raspberry Pi measurements. Power and current cannot be measured without an external meter or trusted device telemetry and are deliberately not fabricated.

## Installation

Python 3.10–3.12 recommended. Use a virtual environment and a compatible CPU or CUDA installation of PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-extra.txt # optional extras for comparators/ONNX/microphone
python -m pytest -q
```

## Dataset

Official MIMII: <https://zenodo.org/records/3384388>. Download the four 6 dB archives (fan, pump, slider, valve), extract beneath `data/mimii/` and retain the `id_00`, `id_02`, `id_04`, `id_06`, `normal`, and `abnormal` directories. Dataset is not included. The loader uses the first waveform channel (Python index zero); it excludes archive paths explicitly identified as 0 dB or -6 dB. Verify the scanned manifest distribution against the paper's Table 2A: 18,019 recordings, including 14,719 normal and 3,300 anomalous. If count differs, inspect the directory structure before running full experiments.

```bash
python -m edgeacoustics prepare --data data/mimii --output data/split.csv --seed 42
python -m edgeacoustics train --manifest data/split.csv --output results/primary --config configs/default.json --cache data/features
python -m edgeacoustics evaluate --checkpoint results/primary/best.pt --manifest data/split.csv --output results/evaluation
```

Splits are performed per recording, before windowing; training statistics are reused for val, test, replay and deployment. The 10-second input WAV must produce 19 windows; shorter or longer recordings are rejected by the dataset loader.

## Experiments

```bash
python -m edgeacoustics experiment --kind primary --manifest data/split.csv --output results/primary5 --cache data/features
python -m edgeacoustics experiment --kind classical --manifest data/split.csv --output results/classical
python -m edgeacoustics experiment --kind deep_baselines --manifest data/split.csv --output results/deep
python -m edgeacoustics experiment --kind ablation --manifest data/split.csv --output results/ablation
python -m edgeacoustics experiment --kind lomo --manifest data/split.csv --output results/lomo
python -m edgeacoustics experiment --kind lofo --manifest data/split.csv --output results/lofo
python -m edgeacoustics stats --summary results/ablation/summary.json --output results/ablation/stats.json
```

Training for the full matrix is computationally expensive. Each run records its own checkpoint, normalization, best epoch, per-recording probabilities, threshold, test metrics and plots. Independent runs use `42`, `123`, `2024`, `2025`, `3407` by default. `configs/smoke.json` is for testing the software, not reproduction of manuscript accuracy. Classical comparisons use one averaged 256-dimensional log-Mel summary per recording whereas neural methods use 19 overlapping spectrogram windows; this deliberate representation difference must be stated when comparing algorithms.

## Export and runtime

```bash
python -m edgeacoustics compress --checkpoint results/primary/best.pt --manifest data/split.csv --output results/export --sparsity 0.30
python -m edgeacoustics benchmark --checkpoint results/primary/best.pt --output results/benchmark.json
python -m edgeacoustics replay --checkpoint results/primary/best.pt --audio path/to/recording.wav --output results/replay.jsonl
python -m edgeacoustics live --checkpoint results/primary/best.pt --output results/live.jsonl
```

`compress` emits `pruned.pt`, `model.onnx` and (with ONNX Runtime extras) `model_int8.onnx`. Quantization calibration uses train-only windows. The `benchmark` command measures inference and preprocessing + inference separately for a synthetic one-second window and documents its hardware and software environment. `replay` produces timestamped JSON Lines with anomaly scores, machine probabilities, warmup and alert flags. During a five-window warmup, alerts remain disabled. The logger does not implement network or PLC actuation.

## Fast smoke run (no public dataset needed)

```bash
python scripts/make_demo.py --output demo_mimii --per-group 6
python -m edgeacoustics prepare --data demo_mimii --output demo_split.csv
python -m edgeacoustics train --manifest demo_split.csv --output demo_run --config configs/smoke.json
python -m edgeacoustics replay --checkpoint demo_run/best.pt --audio demo_mimii/fan/id_00/normal/0000.wav --output demo_replay.jsonl
```

Synthetic data are designed only to test the path from data to trained model to inference; never report resulting scores as MIMII benchmarks.

## Training reproducibility and manuscript alignment

- **Classes:** fan, pump, slider (slide rail), valve. Binary: `normal=0`, `abnormal=1`.
- **Strata:** machine type × machine ID × anomaly state. Per-stratum deterministic `floor(0.70n)` train, `floor(0.15n)` val, rest test; small strata receive at least one sample per subset. Counts may differ slightly from manuscript table due to rounding policy, but no recording crosses partitions.
- **Features:** Hamming FFT 1024, hop 512, centered zero-padded waveform, 128 HTK triangular Mel filters, `log(mel+1e-6)`; `[mel, time]=[128,32]`, 19 windows/recording.
- Adam `1e-3`, batch 32, max 100 epochs, val-loss early stopping patience 10, LR decay patience 5, L2 `1e-4`, dropout `.3`, equal task loss weights.
- Recording-level mean of 19 window probabilities; anomaly threshold maximizes validation F1 and is fixed on test data. LOMO and LOFO use re-split/re-train; train-only normalizer recomputed per run.
- **Model shape:** preserves time sequence through frequency-axis pooling, TCN and additive attention; model does not include causal *audio acquisition* in reported latency.

## Security and responsible usage

Do not treat this prototype as a certified plant safety controller. Review alerts and failure conditions in a human-supervised monitoring loop. Model files loaded via PyTorch pickle must come from trusted sources. The runtime does not transmit audio outside the host.

### Using the exported INT8/ONNX file directly on embedded Linux

Copy `results/export/model_int8.onnx` (or `model.onnx`), `results/export/config.json` and `results/primary/normalization.npz` to the Linux device. Install a compatible ONNX Runtime Python wheel, then use:

```bash
python -m edgeacoustics replay-onnx --model results/export/model_int8.onnx --normalization results/primary/normalization.npz --audio path/to/recording.wav --output results/onnx_replay.jsonl --threshold 0.5
```

Use the saved validation threshold from `results/primary/threshold.json`, not the example `0.5`. The ONNX runner times its own preprocessing, operator execution and postprocessing; it never claims microphone-capture time is included. Static QDQ export and ONNX Runtime execution are optional and must be validated on the target device; no such hardware test was performed during preparation of this repository.
