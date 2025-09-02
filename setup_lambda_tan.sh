#!/usr/bin/env bash
# Setup script for TAM on Lambda/Ubuntu (works on x86_64 and aarch64)
# - Installs Miniforge (per-arch)
# - Creates/activates conda env TAM (Python 3.9)
# - Installs requirements.txt (if present)
# - Installs PyMuPDF correctly (uninstalls legacy 'fitz')
# - Verifies imports (torch, cv2, transformers, fitz, numpy, matplotlib)
# - Optionally logs into Hugging Face using HF_TOKEN env var (safer than inline tokens)
# - Installs PyTorch nightly cu128 if CUDA GPU detected & platform supported
#
# Usage:
#   export HF_TOKEN="hf_xxx"   # optional
#   bash setup_lambda_tam.sh

set -euo pipefail

log() { printf "\033[1;36m[%s]\033[0m %s\n" "$(date +'%H:%M:%S')" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[ERR]\033[0m  %s\n" "$*" >&2; }

ARCH="$(uname -m)"
OS="$(uname -s)"
WORKDIR="$(pwd)"

MINIFORGE_BASE="${HOME}/miniforge3"
ENV_NAME="TAM"
PY_VER="3.9"
REQUIREMENTS_FILE="${WORKDIR}/requirements.txt"

# ---------- helpers ----------
need_cmd() { command -v "$1" >/dev/null 2>&1; }
have_gpu() { need_cmd nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }
cuda_reported() { nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}'; } # e.g., 12.8

install_miniforge() {
  local url file
  case "${OS}-${ARCH}" in
    Linux-x86_64)  file="Miniforge3-Linux-x86_64.sh" ;;
    Linux-aarch64) file="Miniforge3-Linux-aarch64.sh" ;;
    Darwin-arm64)  file="Miniforge3-MacOSX-arm64.sh" ;;  # just in case
    Darwin-x86_64) file="Miniforge3-MacOSX-x86_64.sh" ;;
    *) err "Unsupported platform: ${OS}-${ARCH}"; exit 1 ;;
  esac
  url="https://github.com/conda-forge/miniforge/releases/latest/download/${file}"

  log "Downloading Miniforge: ${file}"
  curl -L -o "/tmp/${file}" "${url}"
  chmod +x "/tmp/${file}"

  log "Installing Miniforge to ${MINIFORGE_BASE}"
  bash "/tmp/${file}" -b -p "${MINIFORGE_BASE}"

  log "Initializing Conda for bash/zsh"
  "${MINIFORGE_BASE}/bin/conda" init bash >/dev/null 2>&1 || true
  if [ -n "${ZSH_VERSION-}" ]; then
    "${MINIFORGE_BASE}/bin/conda" init zsh >/dev/null 2>&1 || true
  fi
}

ensure_conda() {
  if ! need_cmd conda; then
    if [ -x "${MINIFORGE_BASE}/bin/conda" ]; then
      # shellcheck source=/dev/null
      source "${MINIFORGE_BASE}/etc/profile.d/conda.sh"
    else
      install_miniforge
      # shellcheck source=/dev/null
      source "${MINIFORGE_BASE}/etc/profile.d/conda.sh"
    fi
  else
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
}

create_env() {
  log "Creating conda env '${ENV_NAME}' (Python ${PY_VER})"
  # Prefer conda-forge and strict channel priority for reproducibility
  conda config --add channels conda-forge >/dev/null 2>&1 || true
  conda config --set channel_priority strict >/dev/null 2>&1 || true
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
}

activate_env() {
  log "Activating env '${ENV_NAME}'"
  conda activate "${ENV_NAME}"
}

install_requirements() {
  log "Upgrading pip"
  python -m pip install --upgrade pip

  if [ -f "${REQUIREMENTS_FILE}" ]; then
    log "Installing from requirements.txt"
    # Robust install: retry once if transient failure
    python -m pip install -r "${REQUIREMENTS_FILE}" || {
      warn "requirements.txt install failed once — retrying..."
      python -m pip install -r "${REQUIREMENTS_FILE}"
    }
  else
    warn "No requirements.txt found at ${REQUIREMENTS_FILE} — skipping"
  fi

  # Clean up conflicting legacy 'fitz' and ensure proper PyMuPDF
  log "Ensuring correct PyMuPDF (import as 'fitz')"
  python -m pip uninstall -y fitz >/dev/null 2>&1 || true
  python -m pip install --upgrade PyMuPDF
}

hf_login_if_token() {
  if [ -n "${HF_TOKEN-}" ]; then
    log "Logging into Hugging Face via HF_TOKEN env (safer than inline tokens)"
    # Newer huggingface_hub supports non-interactive login:
    python - <<'PY'
import os, sys
from huggingface_hub import login
tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit(0)
try:
    login(token=tok, add_to_git_credential=True)
    print("Hugging Face login: OK")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    sys.exit(1)
PY
  else
    warn "HF_TOKEN not set; skipping Hugging Face login"
  fi
}

maybe_install_torch_nightly_cu128() {
  if have_gpu; then
    local cuda_ver
    cuda_ver="$(cuda_reported || echo "unknown")"
    log "NVIDIA GPU detected (CUDA reported by driver: ${cuda_ver})"

    if [ "${ARCH}" = "x86_64" ]; then
      log "Installing PyTorch nightly (cu128) for x86_64"
      # Remove any preexisting torch family to avoid ABI mismatches
      python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

      # Nightly cu128 wheels
      python -m pip install --pre --upgrade \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
        torch torchvision

      python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
    else
      warn "Non-x86_64 GPU system detected (${ARCH}); installing CPU-only torch for stability"
      python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
  else
    warn "No NVIDIA GPU detected; installing CPU-only torch"
    python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
}

verify_imports() {
  log "Verifying critical imports"
  python - <<'PY'
mods = ["torch","cv2","transformers","fitz","numpy","matplotlib"]
failed = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        failed.append((m, str(e)))
if failed:
    print("Import failures:")
    for m, e in failed:
        print(f" - {m}: {e}")
    raise SystemExit(1)
print("✅ All critical packages imported successfully")
PY
}

print_summary() {
  echo
  log "🎉 TAM environment setup completed!"
  echo "Next steps:"
  echo "  1) conda activate ${ENV_NAME}"
  echo "  2) python demo.py   # if your repo provides a demo"
  echo
  echo "Environment info:"
  echo -n "  - Python: " && python --version
  echo -n "  - PyTorch: " && python -c 'import torch; print(getattr(torch, "__version__", "not installed"))' || true
  echo -n "  - OpenCV: " && python -c 'import cv2; print(cv2.__version__)' || true
  echo
  echo "Utilities:"
  echo "  - Deactivate: conda deactivate"
  echo "  - Remove env: conda env remove -n ${ENV_NAME}"
}

main() {
  log "🚀 Starting TAM setup for ${OS}-${ARCH} in ${WORKDIR}"

  ensure_conda

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    warn "Conda env '${ENV_NAME}' already exists — reusing"
  else
    create_env
  fi

  activate_env
  install_requirements
  hf_login_if_token
  maybe_install_torch_nightly_cu128
  verify_imports
  print_summary
}

main "$@"
