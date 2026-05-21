# attention

Research scaffold for attention experiments.

This repository starts as a clean template with:

- agent-facing rules and project memory files
- local-only environment templates
- W&B/Hugging Face/project/SSH environment loading conventions
- rsync upload/download helpers for remote GPU servers
- a minimal Python package and test skeleton

## Quick Start

```bash
cd ~/Desktop/attention
cp .project.env.example .project.env
cp .ssh_server.env.example .ssh_server.env
cp .wandb.env.example .wandb.env
cp .hf.env.example .hf.env
scripts/env_check.sh
```

Fill only local ignored env files with real values. Do not commit credentials.

## Remote Sync

Upload explicit files:

```bash
rsync/upload.sh README.md src/attention
```

Upload the initial repository without overwriting existing remote files:

```bash
rsync/bootstrap.sh
```

Download remote results:

```bash
REMOTE_RESULT_DIR=/absolute/remote/results rsync/download.sh
```

## Project Memory

- `AGENTS.md`: high-level project intent and agent context.
- `AGENT_RULES.md`: operating rules for local/remote work.
- `PROJECT.md`: repository structure, commands, env conventions.
- `HISTORY.md`: concise change log.
