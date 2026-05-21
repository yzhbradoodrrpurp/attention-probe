# Agent Rules

This repository uses a local-development and remote-GPU workflow.

## Environment Boundaries

- Edit code locally first.
- Use remote servers for GPU-heavy training, evaluation, and environment checks.
- Do not hardcode local paths, remote paths, SSH targets, conda env names, GPU ids,
  data roots, or result directories in scripts. Put them in ignored env files.
- Do not modify remote code manually when a local edit plus rsync is practical.

## Secrets

- Real values belong only in ignored local env files or the current shell.
- Commit only `*.env.example` templates.
- Never print or commit passwords, API keys, tokens, or private host details.
- `.ssh_server.env` is local-only and should not be uploaded by default.

## Env Files

Scripts may load these files when present:

- `.project.env`
- `.ssh_server.env`
- `.wandb.env`
- `.hf.env`

Example files are tracked. Real env files are ignored.

## SSH And Rsync

- SSH settings are read from `.ssh_server.env` or exported shell variables.
- Use `rsync/upload.sh path/to/file ...` for normal work.
- Use `rsync/bootstrap.sh` only for initial remote bootstrap.
- Use `rsync/download.sh` for remote-to-local result downloads.
- Do not use `--delete` unless the user explicitly asks for that exact scope.
- Prefer moving unwanted remote files to a temp-delete directory over deleting.

## Git

- Keep changes scoped.
- Review `git status --short` before summarizing work.
- Do not commit secrets or generated run artifacts.
- Use `HISTORY.md` for concise project memory.

## History

After meaningful changes, append a short `HISTORY.md` entry with:

- timestamp
- files changed
- effect of the change
