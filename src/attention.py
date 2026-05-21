#!/usr/bin/env python
"""Compare image attention with and without an additional related crop."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class ImageSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Config:
    model: str
    dataset: str | None
    subset: str | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    seed: int
    min_pixels: int
    max_pixels: int


@dataclass(frozen=True)
class Sample:
    image: Path
    question: str
    crop_image: Path | None
    crop_box: str | None
    source: str
    index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two MLLM prompts, with and without an additional crop, and compare "
            "generation-time attention to the original image tokens."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML config with model, dataset, and subset fields.",
    )
    parser.add_argument("--model", help="Model name or path. Overrides config.yaml.")
    parser.add_argument("--dataset", help="Dataset name. Overrides config.yaml.")
    parser.add_argument("--subset", help="Dataset subset name. Overrides config.yaml.")
    parser.add_argument("--sample-index", type=int, default=0, help="JSONL sample index to run.")
    parser.add_argument("--image", type=Path, help="Path to the original image.")
    parser.add_argument("--question", help="Text question for both conditions.")
    parser.add_argument(
        "--crop-box",
        help="Crop box as x1,y1,x2,y2 in original-image pixel coordinates.",
    )
    parser.add_argument(
        "--crop-image",
        type=Path,
        help="Optional precomputed crop image. Overrides --crop-box if both are provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for JSON summaries, attention arrays, and heatmaps.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument(
        "--layers",
        default="-1",
        help="Comma-separated layer indexes to average, e.g. -1 or -4,-3,-2,-1.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'auto' for normal remote GPU runs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".hf.env"),
        help="Optional env file to load before model download/cache setup.",
    )
    return parser.parse_args()


def parse_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def parse_int(value: Any, default: int) -> int:
    scalar = parse_scalar(value)
    if scalar is None:
        return default
    return int(float(scalar))


def parse_float(value: Any, default: float) -> float:
    scalar = parse_scalar(value)
    if scalar is None:
        return default
    return float(scalar)


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    lines = path.read_text().splitlines()
    line_index = 0
    while line_index < len(lines):
        raw_line = lines[line_index]
        line = raw_line.split("#", 1)[0].rstrip()
        line_index += 1
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value in {">", ">-", "|", "|-"}:
                block_lines = []
                while line_index < len(lines) and (
                    lines[line_index].startswith(" ") or not lines[line_index].strip()
                ):
                    block_line = lines[line_index].strip()
                    if block_line:
                        block_lines.append(block_line)
                    line_index += 1
                if value.startswith(">"):
                    data[key] = " ".join(block_lines)
                else:
                    data[key] = "\n".join(block_lines)
                current_section = None
                continue
            scalar = parse_scalar(value)
            if scalar is None:
                data[key] = {}
                current_section = key
            else:
                data[key] = scalar
                current_section = None
            continue
        if current_section and line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            section = data.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key.strip()] = parse_scalar(value)
    return data


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config(
            model="Qwen/Qwen3-VL-4B-Instruct",
            dataset=None,
            subset=None,
            max_new_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            seed=42,
            min_pixels=65536,
            max_pixels=16777216,
        )

    try:
        import yaml
    except ImportError:
        raw = load_simple_yaml(path)
    else:
        raw = yaml.safe_load(path.read_text()) or {}

    generation = raw.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}
    image = raw.get("image", {})
    if not isinstance(image, dict):
        image = {}

    dataset_value = raw.get("dataset")
    subset = raw.get("subset")
    if isinstance(dataset_value, dict):
        dataset = dataset_value.get("name")
        subset = dataset_value.get("subset", subset)
    else:
        dataset = dataset_value

    return Config(
        model=parse_scalar(raw.get("model")) or "Qwen/Qwen3-VL-4B-Instruct",
        dataset=parse_scalar(dataset),
        subset=parse_scalar(subset),
        max_new_tokens=parse_int(generation.get("max_new_tokens"), 8192),
        temperature=parse_float(generation.get("temperature"), 0.7),
        top_p=parse_float(generation.get("top_p"), 0.8),
        top_k=parse_int(generation.get("top_k"), 20),
        presence_penalty=parse_float(generation.get("presence_penalty"), 1.5),
        repetition_penalty=parse_float(generation.get("repetition_penalty"), 1.0),
        seed=parse_int(generation.get("seed"), 42),
        min_pixels=parse_int(image.get("min_pixels"), 65536),
        max_pixels=parse_int(image.get("max_pixels"), 16777216),
    )


def resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    return Config(
        model=args.model or config.model,
        dataset=parse_scalar(args.dataset) if args.dataset is not None else config.dataset,
        subset=parse_scalar(args.subset) if args.subset is not None else config.subset,
        max_new_tokens=args.max_new_tokens or config.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else config.temperature,
        top_p=args.top_p if args.top_p is not None else config.top_p,
        top_k=args.top_k if args.top_k is not None else config.top_k,
        presence_penalty=(
            args.presence_penalty if args.presence_penalty is not None else config.presence_penalty
        ),
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else config.repetition_penalty
        ),
        seed=args.seed if args.seed is not None else config.seed,
        min_pixels=args.min_pixels if args.min_pixels is not None else config.min_pixels,
        max_pixels=args.max_pixels if args.max_pixels is not None else config.max_pixels,
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_crop_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop-box must contain exactly four integers: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--crop-box must satisfy x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def read_jsonl_record(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError("--sample-index must be non-negative.")
    with path.open() as handle:
        for line_number, line in enumerate(handle):
            if line_number == index:
                return json.loads(line)
    raise IndexError(f"Sample index {index} is out of range for {path}.")


def iter_openai_contents(record: Any) -> list[dict[str, Any]]:
    if isinstance(record, list):
        messages = record
    elif isinstance(record, dict):
        messages = record.get("messages", [])
    else:
        messages = []

    contents: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            contents.append({"role": role, "type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("role", role)
                    contents.append(merged)
    return contents


def extract_image_value(item: dict[str, Any]) -> str | None:
    if item.get("type") in {"image", "input_image"}:
        return parse_scalar(item.get("image") or item.get("path") or item.get("url"))
    if item.get("type") == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            return parse_scalar(image_url.get("url"))
        return parse_scalar(image_url)
    return None


def normalize_image_path(value: str, dataset_path: Path) -> Path | None:
    if value.startswith("data:"):
        return None
    if value.startswith("file://"):
        value = value.removeprefix("file://")
    path = Path(value)
    if path.is_absolute():
        return path

    dataset_root = dataset_path.parent.parent
    candidates = [
        dataset_path.parent / path,
        dataset_root / path,
        dataset_root / "images" / "extracted" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_top_level_path(
    record: dict[str, Any],
    keys: tuple[str, ...],
    dataset_path: Path,
) -> Path | None:
    for key in keys:
        value = parse_scalar(record.get(key))
        if value:
            path = normalize_image_path(value, dataset_path)
            if path is not None:
                return path
    return None


def parse_record_crop_box(record: dict[str, Any]) -> str | None:
    for key in ("crop_pixel_bbox", "bbox", "bbox_2d", "crop_box"):
        value = record.get(key)
        if isinstance(value, list | tuple) and len(value) == 4:
            return ",".join(str(int(float(part))) for part in value)
        scalar = parse_scalar(value)
        if scalar:
            return scalar
    return None


def load_sample_from_dataset(config: Config, index: int) -> Sample:
    if not config.dataset:
        raise ValueError(
            "No dataset configured. Set dataset in config.yaml or pass --image and --question."
        )

    dataset_path = Path(config.dataset)
    record = read_jsonl_record(dataset_path, index)
    if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object at sample index {index}.")

    contents = iter_openai_contents(record)
    image_paths = []
    for item in contents:
        value = extract_image_value(item)
        if value:
            path = normalize_image_path(value, dataset_path)
            if path is not None:
                image_paths.append(path)

    original_image = extract_top_level_path(
        record,
        ("original_image", "original_image_path", "image", "image_path"),
        dataset_path,
    )
    crop_image = extract_top_level_path(
        record,
        ("crop_image", "crop_image_path", "crop", "crop_path"),
        dataset_path,
    )
    if original_image is None and image_paths:
        original_image = image_paths[0]
    if crop_image is None and len(image_paths) > 1:
        crop_image = image_paths[1]
    if crop_image is not None and not crop_image.exists():
        crop_image = None

    question = parse_scalar(record.get("question") or record.get("prompt"))
    if question is None:
        text_parts = [
            parse_scalar(item.get("text"))
            for item in contents
            if item.get("role") in {None, "user"} and parse_scalar(item.get("text"))
        ]
        question = "\n".join(part for part in text_parts if part)

    if original_image is None:
        raise ValueError(f"Could not find an original image path in sample {index}.")
    if not question:
        raise ValueError(f"Could not find a text question in sample {index}.")

    crop_box = parse_record_crop_box(record)
    return Sample(
        image=original_image,
        question=question,
        crop_image=crop_image,
        crop_box=crop_box,
        source=str(dataset_path),
        index=index,
    )


def resolve_sample(args: argparse.Namespace, config: Config) -> Sample:
    if args.image is not None or args.question is not None:
        if args.image is None or args.question is None:
            raise ValueError("Manual mode requires both --image and --question.")
        return Sample(
            image=args.image,
            question=args.question,
            crop_image=args.crop_image,
            crop_box=args.crop_box,
            source="manual",
            index=None,
        )
    return load_sample_from_dataset(config, args.sample_index)


def resolve_output_dir(args: argparse.Namespace, sample: Sample) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if sample.index is not None:
        return Path("results") / f"run_{sample.index}"
    return Path("results") / "run_manual"


def ensure_crop(args: argparse.Namespace) -> Path:
    from PIL import Image

    if args.crop_image is not None:
        return args.crop_image
    if not args.crop_box:
        raise ValueError("Provide either --crop-image or --crop-box for the crop condition.")

    image = Image.open(args.image).convert("RGB")
    box = parse_crop_box(args.crop_box)
    crop = image.crop(box)
    crop_path = args.output_dir / "crop.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)
    return crop_path


def build_messages(
    image_paths: list[Path],
    question: str,
    config: Config,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append(
            {
                "type": "image",
                "image": str(image_path),
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
            }
        )
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def contiguous_spans(mask: Any) -> list[ImageSpan]:
    import torch

    indexes = torch.where(mask)[0].tolist()
    if not indexes:
        return []
    spans: list[ImageSpan] = []
    start = prev = indexes[0]
    for index in indexes[1:]:
        if index == prev + 1:
            prev = index
            continue
        spans.append(ImageSpan(start=start, end=prev + 1))
        start = prev = index
    spans.append(ImageSpan(start=start, end=prev + 1))
    return spans


def find_image_spans(inputs: dict[str, Any], processor: Any, model: Any) -> list[ImageSpan]:
    input_ids = inputs["input_ids"][0].detach().cpu()
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    spans = contiguous_spans(input_ids == image_token_id)
    if spans:
        return spans

    token_types = inputs.get("mm_token_type_ids")
    if token_types is not None:
        return contiguous_spans(token_types[0].detach().cpu() == 1)
    raise RuntimeError("Could not locate image token spans in model inputs.")


def resolve_layer_indexes(selected: list[int], num_layers: int) -> list[int]:
    resolved: list[int] = []
    for layer in selected:
        index = layer if layer >= 0 else num_layers + layer
        if index < 0 or index >= num_layers:
            raise ValueError(f"Layer {layer} is out of range for {num_layers} layers.")
        resolved.append(index)
    return resolved


def layer_attention_to_span(
    attentions: tuple[tuple[Any, ...], ...],
    span: ImageSpan,
    selected_layers: list[int],
) -> tuple[Any, list[float]]:
    import numpy as np

    if not attentions:
        raise RuntimeError("Generation output did not include attentions.")

    num_layers = len(attentions[0])
    layers = resolve_layer_indexes(selected_layers, num_layers)
    attention_sum: np.ndarray | None = None
    mass_by_token: list[float] = []
    used = 0

    for step_attentions in attentions:
        step_vectors = []
        step_mass = []
        for layer in layers:
            tensor = step_attentions[layer]
            if tensor.shape[-1] < span.end:
                continue
            vector = (
                tensor[0, :, -1, span.start : span.end]
                .detach()
                .float()
                .mean(dim=0)
                .cpu()
                .numpy()
            )
            step_vectors.append(vector)
            step_mass.append(float(vector.sum()))
        if not step_vectors:
            continue
        step_vector = np.mean(np.stack(step_vectors, axis=0), axis=0)
        attention_sum = step_vector if attention_sum is None else attention_sum + step_vector
        mass_by_token.append(float(np.mean(step_mass)))
        used += 1

    if attention_sum is None or used == 0:
        raise RuntimeError("No generated-token attentions covered the requested image span.")
    return attention_sum / used, mass_by_token


def image_grid_shape(
    inputs: dict[str, Any],
    image_index: int,
    span_len: int,
    model: Any,
) -> tuple[int, int] | None:
    grid = inputs.get("image_grid_thw")
    if grid is None or len(grid) <= image_index:
        return None
    _, height, width = [int(value) for value in grid[image_index].detach().cpu().tolist()]
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 1))
    height = max(1, height // merge)
    width = max(1, width // merge)
    if height * width == span_len:
        return height, width
    return None


def normalize_map(values: np.ndarray) -> np.ndarray:
    import numpy as np

    values = values.astype(np.float32)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def save_heatmap_overlay(
    image_path: Path,
    attention: Any,
    grid_shape: tuple[int, int] | None,
    output_path: Path,
) -> None:
    import numpy as np
    from PIL import Image

    if grid_shape is None:
        return
    original = Image.open(image_path).convert("RGBA")
    heat = normalize_map(attention.reshape(grid_shape))
    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L").resize(
        original.size,
        Image.Resampling.BICUBIC,
    )
    red = Image.new("RGBA", original.size, (255, 32, 0, 0))
    red.putalpha(heat_img.point(lambda value: int(value * 0.55)))
    Image.alpha_composite(original, red).save(output_path)


def save_diff_heatmap(
    image_path: Path,
    before: Any,
    after: Any,
    grid_shape: tuple[int, int] | None,
    output_path: Path,
) -> None:
    import numpy as np
    from PIL import Image

    if grid_shape is None:
        return
    before_norm = before / max(float(before.sum()), 1e-12)
    after_norm = after / max(float(after.sum()), 1e-12)
    diff = after_norm - before_norm
    scaled = normalize_map(np.abs(diff.reshape(grid_shape)))
    original = Image.open(image_path).convert("RGBA")
    alpha = Image.fromarray(np.uint8(scaled * 255), mode="L").resize(
        original.size, Image.Resampling.BICUBIC
    )
    overlay_array = np.zeros((original.height, original.width, 4), dtype=np.uint8)
    overlay_array[..., 1] = 120
    overlay_array[..., 2] = 255
    overlay_array[..., 3] = np.uint8(np.asarray(alpha) * 0.55)
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    Image.alpha_composite(original, overlay).save(output_path)


def run_condition(
    *,
    name: str,
    image_paths: list[Path],
    question: str,
    original_image: Path,
    processor: Any,
    model: Any,
    config: Config,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np
    import torch

    messages = build_messages(image_paths, question, config)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    spans = find_image_spans(inputs, processor, model)
    if not spans:
        raise RuntimeError(f"No image spans found for condition {name}.")

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = generation.sequences[:, prompt_len:]
    answer = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    original_attention, original_mass = layer_attention_to_span(
        generation.attentions,
        spans[0],
        parse_layers(args.layers),
    )

    condition_dir = args.output_dir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    np.save(condition_dir / "original_attention.npy", original_attention)
    grid_shape = image_grid_shape(inputs, 0, spans[0].length, model)
    save_heatmap_overlay(
        original_image,
        original_attention,
        grid_shape,
        condition_dir / "original_attention.png",
    )

    crop_mass: list[float] | None = None
    if len(spans) > 1:
        _, crop_mass = layer_attention_to_span(
            generation.attentions,
            spans[1],
            parse_layers(args.layers),
        )

    summary = {
        "condition": name,
        "model": config.model,
        "dataset": config.dataset,
        "subset": config.subset,
        "question": question,
        "answer": answer,
        "input_token_count": prompt_len,
        "generated_token_count": int(generated_ids.shape[1]),
        "image_spans": [span.__dict__ for span in spans],
        "original_attention_mass_mean": float(np.mean(original_mass)),
        "original_attention_mass_by_generated_token": original_mass,
        "crop_attention_mass_mean": None if crop_mass is None else float(np.mean(crop_mass)),
        "crop_attention_mass_by_generated_token": crop_mass,
        "original_grid_shape": grid_shape,
        "layers": args.layers,
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "presence_penalty": config.presence_penalty,
            "repetition_penalty": config.repetition_penalty,
            "seed": config.seed,
        },
        "image": {
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
        },
    }
    (condition_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return {
        "name": name,
        "summary": summary,
        "original_attention": original_attention,
        "grid_shape": grid_shape,
    }


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    sample = resolve_sample(args, config)
    args.output_dir = resolve_output_dir(args, sample)
    args.image = sample.image
    args.question = sample.question
    args.crop_image = sample.crop_image
    args.crop_box = sample.crop_box
    load_env_file(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = ensure_crop(args)

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    processor = AutoProcessor.from_pretrained(
        config.model,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        trust_remote_code=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    original_only = run_condition(
        name="original_only",
        image_paths=[sample.image],
        question=sample.question,
        original_image=sample.image,
        processor=processor,
        model=model,
        config=config,
        args=args,
    )
    original_plus_crop = run_condition(
        name="original_plus_crop",
        image_paths=[sample.image, crop_path],
        question=sample.question,
        original_image=sample.image,
        processor=processor,
        model=model,
        config=config,
        args=args,
    )

    grid_shape = original_only["grid_shape"]
    if grid_shape == original_plus_crop["grid_shape"]:
        save_diff_heatmap(
            sample.image,
            original_only["original_attention"],
            original_plus_crop["original_attention"],
            grid_shape,
            args.output_dir / "original_attention_diff_abs.png",
        )

    comparison = {
        "model": config.model,
        "dataset": config.dataset,
        "subset": config.subset,
        "sample_source": sample.source,
        "sample_index": sample.index,
        "image": str(sample.image),
        "crop_image": str(crop_path),
        "question": sample.question,
        "original_only": original_only["summary"],
        "original_plus_crop": original_plus_crop["summary"],
        "delta_original_attention_mass_mean": (
            original_plus_crop["summary"]["original_attention_mass_mean"]
            - original_only["summary"]["original_attention_mass_mean"]
        ),
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "presence_penalty": config.presence_penalty,
            "repetition_penalty": config.repetition_penalty,
            "seed": config.seed,
        },
        "image_config": {
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False)
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
