# History

## 2026-05-21 CST

- Initialized repository scaffold for `attention`.
- Added agent rules, project memory docs, env templates, rsync helpers, package
  skeleton, and a basic smoke test.
- Initialized local git repository without committing secrets or generated files.
- Moved rsync helper scripts into `rsync/`, shortened command names, updated
  documentation references, and kept shared SSH/rsync helper code under
  `rsync/lib/`.
- Removed local `data/` and `outputs/` workspace conventions and deleted their
  related environment variables from project and SSH templates.
- Added an image-attention comparison entry point under `src/attention.py` for
  original-image prompts with and without an additional relevant crop.
- Added `scripts/run_attention.sh` to run the attention comparison module.
- Flattened the source layout by moving the entry point directly under `src/`
  and removing the `src/attention/` package directory.
- Added `config.yaml` for manually setting model and dataset metadata used by
  the attention run.
- Set the default dataset to the remote OpenAI messages JSONL and added a JSONL
  sample adapter for original-image plus crop attention runs.
- Changed the default attention output directory to `results/run_{sample_index}`.
- Moved the attention run config to `src/config.yaml`.
- Replaced rsync `--info=progress2` with `--progress` for macOS rsync
  compatibility.
- Cleared unused `.project.env` variables and removed them from env checks.
- Matched attention generation and image-size defaults to
  `inclusionAI/Zooming-without-Zooming` mm-eval settings.
