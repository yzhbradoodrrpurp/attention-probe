#!/usr/bin/env bash

# Shared SSH/rsync env loader. Source this file from rsync scripts; do not execute it
# directly.

require_remote_env() {
  local repo_root="${1:?repo root required}"
  SSH_ENV_FILE="${SSH_ENV_FILE:-${repo_root}/.ssh_server.env}"

  if [[ -f "${SSH_ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${SSH_ENV_FILE}"
    set +a
  elif [[ -z "${SSH_REMOTE_USER:-}" || -z "${SSH_REMOTE_HOST:-}" || -z "${SSH_REMOTE_PORT:-}" || -z "${SSH_REMOTE_CODE_DIR:-}" ]]; then
    echo "Missing SSH env file: ${SSH_ENV_FILE}" >&2
    echo "Create it locally from .ssh_server.env.example or export SSH_REMOTE_USER, SSH_REMOTE_HOST, SSH_REMOTE_PORT, and SSH_REMOTE_CODE_DIR." >&2
    exit 1
  fi

  : "${SSH_REMOTE_USER:?Missing SSH_REMOTE_USER in ${SSH_ENV_FILE}}"
  : "${SSH_REMOTE_HOST:?Missing SSH_REMOTE_HOST in ${SSH_ENV_FILE}}"
  : "${SSH_REMOTE_PORT:?Missing SSH_REMOTE_PORT in ${SSH_ENV_FILE}}"
  : "${SSH_REMOTE_CODE_DIR:?Missing SSH_REMOTE_CODE_DIR in ${SSH_ENV_FILE}}"

  if [[ -z "${SSH_REMOTE_PASSWORD:-}" && -n "${SSH_PASSWORD:-}" ]]; then
    SSH_REMOTE_PASSWORD="${SSH_PASSWORD}"
  fi

  case "${SSH_REMOTE_PORT}" in
    ''|*[!0-9]*)
      echo "SSH_REMOTE_PORT must be numeric." >&2
      exit 1
      ;;
  esac
  case "${SSH_REMOTE_USER}" in
    ''|*[!A-Za-z0-9._-]*)
      echo "SSH_REMOTE_USER contains unsupported characters." >&2
      exit 1
      ;;
  esac
  case "${SSH_REMOTE_HOST}" in
    ''|*[!A-Za-z0-9._:-]*)
      echo "SSH_REMOTE_HOST contains unsupported characters." >&2
      exit 1
      ;;
  esac

  require_absolute_remote_path "SSH_REMOTE_CODE_DIR" "${SSH_REMOTE_CODE_DIR}"
  if [[ -n "${SSH_REMOTE_RESULTS_DIR:-}" ]]; then
    require_absolute_remote_path "SSH_REMOTE_RESULTS_DIR" "${SSH_REMOTE_RESULTS_DIR}"
  fi

  REMOTE_TARGET="${SSH_REMOTE_USER}@${SSH_REMOTE_HOST}"
  REMOTE_SSH_ARGS=(-p "${SSH_REMOTE_PORT}")
  if [[ -n "${SSH_REMOTE_SSH_OPTIONS:-}" ]]; then
    local extra_ssh_args=()
    read -r -a extra_ssh_args <<< "${SSH_REMOTE_SSH_OPTIONS}"
    REMOTE_SSH_ARGS+=("${extra_ssh_args[@]}")
  fi

  RSYNC_SSH_CMD="ssh"
  local arg=""
  for arg in "${REMOTE_SSH_ARGS[@]}"; do
    RSYNC_SSH_CMD+=" $(printf '%q' "${arg}")"
  done
}

require_absolute_remote_path() {
  local name="${1:?name required}"
  local value="${2:?value required}"
  case "${value}" in
    /*) ;;
    *)
      echo "${name} must be an absolute path." >&2
      exit 1
      ;;
  esac
  case "${value}" in
    *[!A-Za-z0-9_./-]*)
      echo "${name} contains unsupported characters." >&2
      exit 1
      ;;
  esac
}

run_with_password_prompt() {
  if [[ -n "${SSH_REMOTE_PASSWORD:-}" ]] && command -v expect >/dev/null 2>&1; then
    local command_string=""
    printf -v command_string "%q " "$@"
    command_string="${command_string% }"
    SSH_REMOTE_PASSWORD="${SSH_REMOTE_PASSWORD}" EXPECT_CMD="${command_string}" expect -c '
set timeout -1
log_user 1
spawn /bin/sh -c $env(EXPECT_CMD)
expect {
  -re "(?i)password:" {
    send -- "$env(SSH_REMOTE_PASSWORD)\r"
    exp_continue
  }
  -re "(yes/no|fingerprint)" {
    send -- "yes\r"
    exp_continue
  }
  eof {
    catch wait result
    exit [lindex $result 3]
  }
}
'
  else
    "$@"
  fi
}

remote_mkdir() {
  local remote_dir="${1:?remote directory required}"
  local remote_dir_escaped=""
  remote_dir_escaped="$(printf '%q' "${remote_dir}")"
  run_with_password_prompt \
    ssh "${REMOTE_SSH_ARGS[@]}" "${REMOTE_TARGET}" \
    "mkdir -p -- ${remote_dir_escaped}"
}
