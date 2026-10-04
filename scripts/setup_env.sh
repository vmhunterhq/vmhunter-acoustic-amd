#!/usr/bin/env bash
# Create the training venv at venv (torch with CUDA 12.8 wheels).
set -euo pipefail
VENV=venv
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install -q numpy pandas soundfile scikit-learn tqdm onnx onnxruntime
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
