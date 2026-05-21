# attention Project

This document tracks repository structure, runtime conventions, and stable entry
points. Keep experiment ideas in `AGENTS.md` and operational rules in
`AGENT_RULES.md`.

## Structure

```text
src/attention/            Python package
tests/                    lightweight tests
results/                  downloaded or processed results, ignored
scripts/                  env checks
rsync/                    rsync upload/download helpers
rsync/lib/                shared SSH/rsync shell helpers
AGENTS.md                 project intent and high-level context
AGENT_RULES.md            agent operating rules
PROJECT.md                structure and command conventions
HISTORY.md                concise change log
```

## Env Conventions

Tracked templates:

- `.project.env.example`
- `.ssh_server.env.example`
- `.wandb.env.example`
- `.hf.env.example`

Ignored local files:

- `.project.env`
- `.ssh_server.env`
- `.wandb.env`
- `.hf.env`

## Common Commands

Check local env presence without printing secrets:

```bash
scripts/env_check.sh
```

Run tests:

```bash
python -m pytest
```

Upload selected files:

```bash
rsync/upload.sh path/to/file
```

Download results:

```bash
REMOTE_RESULT_DIR=/absolute/remote/results rsync/download.sh
```

## W&B

Use `.wandb.env` for real W&B configuration. Keep real values out of git.

Expected variables:

- `WANDB_BASE_URL`
- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_MODE`

## Remote Project

Remote paths and runtime defaults come from `.project.env` and `.ssh_server.env`.
Do not hardcode a specific server in project scripts.
