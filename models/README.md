# Models

This project currently uses the Ultra Fast Lane Detection (UFLD) model trained on the TuSimple lane detection dataset.

## Current Model

Model:

```text
tusimple_18.onnx
```

Framework:

```text
ONNX Runtime (CPU)
```

Expected input shape:

```text
[1, 3, 288, 800]
```

Expected output shape:

```text
[1, 101, 56, 4]
```

## Download

Download the model and place it in:

```text
models/tusimple_18.onnx
```

Download URL:

```text
https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/tusimple_18.onnx
```

## Why the model is not stored in Git

The model file is intentionally excluded from version control because large binary files should not be committed to Git repositories.

The repository contains only code and documentation. Model files should be downloaded locally.
