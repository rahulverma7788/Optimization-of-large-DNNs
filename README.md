# Optimization-of-large-DNNs

# Model Conversion and Inference with OpenVINO

This project demonstrates how to convert a Hugging Face Transformers model for use on CPU hardware using the OpenVINO toolkit, and run inference with the converted model. The model used is `microsoft/Florence-2-base-ft`, which supports both image and text inputs.

## Tasks Overview

### 1. Model Conversion to OpenVINO
The first task involves exporting a PyTorch model to ONNX format, followed by conversion to OpenVINO IR format using the Model Optimizer (`mo`). The model uses flash-attention, which is disabled for CPU compatibility. Additionally, conditional code that is not traceable by PyTorch tracing is handled to ensure a smooth export.

### 2. Inference on CPU with OpenVINO
The second task is to run inference using the converted OpenVINO model on CPU. This involves:
- Preprocessing an image and a text prompt for model input.
- Running inference using OpenVINO's runtime.
- Applying softmax and top-k sampling to the output to generate the final text prediction.

## Prerequisites

- Google Colab or a local environment with the following packages:
  - PyTorch 2.4.1+cu121
  - Transformers 4.44.2
  - ONNX 1.16.0
  - OpenVINO 2024.4.0
  - Pillow 10.4.0
  - NumPy 1.26.4
  - Requests 2.32.3
