#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_status() {
  local name="${1:?name required}"
  local value="${!name:-}"
  if [[ -n "${value}" ]]; then
    printf '%s=%s\n' "${name}" '<set>'
  else
    printf '%s=%s\n' "${name}" '<unset>'
  fi
}

load_if_present() {
  local file="${1:?file required}"
  if [[ -f "${file}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${file}"
    set +a
  fi
}

load_if_present "${SSH_ENV_FILE:-${REPO_ROOT}/.ssh_server.env}"
load_if_present "${PROJECT_ENV_FILE:-${REPO_ROOT}/.project.env}"
load_if_present "${HF_ENV_FILE:-${REPO_ROOT}/.hf.env}"
load_if_present "${WANDB_ENV_FILE:-${REPO_ROOT}/.wandb.env}"

for name in \
  SSH_REMOTE_USER \
  SSH_REMOTE_HOST \
  SSH_REMOTE_PORT \
  SSH_REMOTE_CODE_DIR \
  SSH_REMOTE_RESULTS_DIR \
  LOCAL_CONFIG_PATH \
  LOCAL_TRAIN_ENTRY \
  REMOTE_CONDA_ENV \
  REMOTE_GPU_IDS \
  REMOTE_TRAIN_CONFIG \
  HF_ENDPOINT \
  HF_TOKEN \
  WANDB_BASE_URL \
  WANDB_API_KEY
do
  print_status "${name}"
done
