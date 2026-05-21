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
rsync/upload.sh README.md src scripts
```

Upload the initial repository without overwriting existing remote files:

```bash
rsync/bootstrap.sh
```

Download remote results:

```bash
REMOTE_RESULT_DIR=/absolute/remote/results rsync/download.sh
```

## Attention Run

Edit `src/config.yaml` for the model and dataset metadata:

```yaml
model: Qwen/Qwen3-VL-4B-Instruct
dataset: >-
  /home/deeplearn/dataDisk/zwz_dataset/openai_export/test_10k_11k_original_with_crop_open_qa.openai_messages.jsonl
subset: open_qa

generation:
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  presence_penalty: 1.5
  repetition_penalty: 1.0
  seed: 42

image:
  min_pixels: 65536
  max_pixels: 16777216
```

Run the image-attention comparison on the remote server:

```bash
scripts/run_attention.sh \
  --sample-index 0
```

By default, outputs are written to `results/run_{sample_index}`.

The default dataset is an OpenAI messages JSONL file. The loader treats the
first image in a sample as the original image and the second image as the crop.
Use `--image`, `--question`, and `--crop-image` or `--crop-box` to override the
dataset loader and run a manual sample.

Generation and image-size defaults match the `mm-eval` configuration from
`inclusionAI/Zooming-without-Zooming`. `presence_penalty` is recorded for
comparability but is not passed to Hugging Face `generate()`.

## Project Memory

- `AGENTS.md`: high-level project intent and agent context.
- `AGENT_RULES.md`: operating rules for local/remote work.
- `PROJECT.md`: repository structure, commands, env conventions.
- `HISTORY.md`: concise change log.
