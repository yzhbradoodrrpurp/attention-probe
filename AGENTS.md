# Attention Project

This repository is a fresh scaffold for attention-related research and engineering.

Use this file for high-level ideas, experimental motivation, and assumptions that
future agents should understand before editing code.

## Operating Model

- Local machine: code editing, review, documentation, and lightweight tests.
- Remote GPU server: dependency installation, training, evaluation, and long runs.
- Sync mechanism: use `rsync/upload.sh` for explicit scoped uploads.
- Secrets: never commit or print real tokens, passwords, host credentials, or W&B keys.

## Project Memory Files

- Read `AGENT_RULES.md` before doing remote, SSH, rsync, git, or env work.
- Read `PROJECT.md` before changing repository structure or runtime commands.
- Append a short entry to `HISTORY.md` after meaningful changes.

## Current Scope

This is a template. Add method details here once the concrete attention experiment
is defined.
