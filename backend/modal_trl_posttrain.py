from __future__ import annotations

"""
Unified Modal + TRL post-training example.

This script is designed to be run with `modal run` and covers:
- `SFTTrainer`
- `DPOTrainer`
- `KTOTrainer`
- `ORPOTrainer`
- `CPOTrainer`
- `BCOTrainer`

Representative commands:

```bash
modal run backend/modal_trl_posttrain.py --config backend/modal_trl_posttrain.example.yaml
```

vLLM handoff example:

```bash
vllm serve Qwen/Qwen3.5-9B-Base --enable-lora --lora-modules run=/checkpoints/experiments/qwen3.5-9b-dpo/final_adapter
```

Notes:
- This example is intentionally PEFT/LoRA-first.
- `gpu_type` is part of the public config. Runtime launch normalizes public aliases
  (`A10`, `A10G`, `L40S`, `H100`) and dispatches to GPU-specific Modal classes.
- For non-SFT trainers, this example expects a prior SFT adapter path in
  `seed_artifact`, following TRL's recommended workflow.
- Create a Modal secret named `huggingface-secret` before running this example.
- If `enable_wandb=True`, also create a Modal secret named `wandb-secret`.
- Pass a YAML file to `--config`.
- In YAML, `target_modules` can be either a proper YAML list or a comma-separated string.
"""

import json
import os
import random
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import modal


TrainerType = Literal["sft", "dpo", "kto", "orpo", "cpo", "bco"]

APP_NAME = "trl-posttraining"
DEFAULT_GPU_TYPE = "A10"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_COMPARISON_MAX_EXAMPLES = 15
DEFAULT_CLASSIFICATION_EVAL_MAX_EXAMPLES = 256
DEFAULT_GENERATION_EVAL_MAX_NEW_TOKENS = 256
DEFAULT_GENERATION_EVAL_OPENAI_MAX_WORKERS = 4
DEFAULT_EVAL_JUDGE_MODEL = "gpt-5.4"
OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
THINK_TAG_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

NON_TEXT_COLUMN_PATTERNS = (
    re.compile(r"(^|_)(image|images|img|photo|picture|pixel|pixels|frame|frames)(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)(scan|dicom|xray|x_ray|mammogram|thumbnail)(_|$)", re.IGNORECASE),
)

MODEL_CACHE_DIR = Path("/model_cache")
DATASET_CACHE_DIR = Path("/dataset_cache")
CHECKPOINTS_DIR = Path("/checkpoints")
EXPERIMENTS_DIR = CHECKPOINTS_DIR / "experiments"
PROVENANCE_DATASET_COLUMN = "__pt_dataset_id"
PROVENANCE_SPLIT_COLUMN = "__pt_source_split"
STRUCTURED_TRAINING_METRIC_PREFIX = "PT_METRIC_EVENT::"
STRUCTURED_LIFECYCLE_EVENT_PREFIX = "PT_LIFECYCLE_EVENT::"
STRUCTURED_PROGRESS_PREFIX = "PT_PROGRESS::"

HF_SECRET = modal.Secret.from_name("huggingface-secret")
MODEL_CACHE_VOLUME = modal.Volume.from_name("trl-model-cache", create_if_missing=True)
DATASET_CACHE_VOLUME = modal.Volume.from_name("trl-dataset-cache", create_if_missing=True)
CHECKPOINTS_VOLUME = modal.Volume.from_name("trl-checkpoints", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "wandb",
        "sentencepiece",
        "pyyaml",
    )
    .env(
        {
            "HF_HOME": str(MODEL_CACHE_DIR / "huggingface"),
            "TRANSFORMERS_CACHE": str(MODEL_CACHE_DIR / "transformers"),
            "HF_DATASETS_CACHE": str(DATASET_CACHE_DIR / "huggingface_datasets"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)

PUBLIC_GPU_TYPE_BY_ALIAS = {
    "A10": "A10",
    "A10G": "A10",
    "L40S": "L40S",
    "H100": "H100",
}
MODAL_GPU_TYPE_BY_PUBLIC_GPU_TYPE = {
    "A10": "A10G",
    "L40S": "L40S",
    "H100": "H100",
}


def _normalize_gpu_type(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("gpu_type must be one of A10, A10G, L40S, or H100.")

    normalized = value.strip().upper()
    public_gpu_type = PUBLIC_GPU_TYPE_BY_ALIAS.get(normalized)
    if public_gpu_type is None:
        raise ValueError(
            f"Unsupported gpu_type {value!r}. Expected one of A10, A10G, L40S, or H100."
        )

    modal_gpu_type = MODAL_GPU_TYPE_BY_PUBLIC_GPU_TYPE[public_gpu_type]
    return public_gpu_type, modal_gpu_type


@dataclass
class TrainConfig:
    trainer_type: TrainerType
    dataset_name: str
    output_name: str
    base_model: str = DEFAULT_BASE_MODEL
    base_model_revision: str | None = None
    dataset_config: str | None = None
    dataset_source_type: str = "huggingface"
    prepared_dataset_manifest: dict[str, Any] | None = None
    task_spec: dict[str, Any] | None = None
    evaluation_plan: dict[str, Any] | None = None
    training_estimate: dict[str, Any] | None = None
    train_split: str = "train"
    eval_split: str | None = None
    seed_artifact: str | None = None
    gpu_type: str = DEFAULT_GPU_TYPE
    max_length: int = 4096
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    max_steps: int = -1
    use_peft: bool = True
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    learning_rate: float | None = None
    beta: float = 0.1
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    merge_after_train: bool = False
    enable_wandb: bool = False

    def __post_init__(self) -> None:
        self.trainer_type = self.trainer_type.lower().strip()  # type: ignore[assignment]
        if self.trainer_type not in {"sft", "dpo", "kto", "orpo", "cpo", "bco"}:
            raise ValueError(f"Unsupported trainer_type: {self.trainer_type}")
        self.dataset_source_type = self.dataset_source_type.lower().strip()
        if self.dataset_source_type not in {"huggingface", "prepared_manifest"}:
            raise ValueError("dataset_source_type must be either 'huggingface' or 'prepared_manifest'.")
        if self.dataset_source_type == "huggingface" and not self.dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string for huggingface datasets.")
        if self.dataset_source_type == "prepared_manifest" and not isinstance(
            self.prepared_dataset_manifest, dict
        ):
            raise ValueError(
                "prepared_dataset_manifest must be a mapping when dataset_source_type='prepared_manifest'."
            )
        if not self.output_name.strip():
            raise ValueError("output_name must be a non-empty string.")
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty string.")
        self.gpu_type, _ = _normalize_gpu_type(self.gpu_type)
        if self.trainer_type != "sft" and not (self.seed_artifact or "").strip():
            raise ValueError("seed_artifact is required for dpo, kto, orpo, cpo, and bco runs.")
        if not self.use_peft:
            raise ValueError("This example is LoRA/PEFT-first. Set use_peft=True.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.per_device_train_batch_size <= 0 or self.per_device_eval_batch_size <= 0:
            raise ValueError("Per-device batch sizes must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("save_steps and logging_steps must be positive.")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive.")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive.")
        if not self.target_modules:
            raise ValueError("target_modules must contain at least one module name.")
        if Path(self.output_name).is_absolute() or ".." in Path(self.output_name).parts:
            raise ValueError("output_name must be a simple experiment name, not a path.")

    @property
    def resolved_learning_rate(self) -> float:
        if self.learning_rate is not None:
            return self.learning_rate
        if self.trainer_type == "sft":
            return 1e-4
        if self.trainer_type == "bco":
            return 5e-7
        return 1e-6

    @property
    def report_to(self) -> str | list[str]:
        return ["wandb"] if self.enable_wandb else "none"

    @property
    def run_dir(self) -> Path:
        return EXPERIMENTS_DIR / self.output_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def final_adapter_dir(self) -> Path:
        return self.run_dir / "final_adapter"

    @property
    def merged_dir(self) -> Path:
        return self.run_dir / "merged"

    @property
    def run_config_path(self) -> Path:
        return self.run_dir / "run_config.yaml"

    @property
    def modal_gpu_type(self) -> str:
        _, modal_gpu_type = _normalize_gpu_type(self.gpu_type)
        return modal_gpu_type

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["learning_rate"] = self.resolved_learning_rate
        data["run_dir"] = str(self.run_dir)
        data["checkpoints_dir"] = str(self.checkpoints_dir)
        data["final_adapter_dir"] = str(self.final_adapter_dir)
        data["merged_dir"] = str(self.merged_dir)
        data["run_config_path"] = str(self.run_config_path)
        return data


def _build_runtime_secrets(config: TrainConfig) -> list[modal.Secret]:
    secrets = [HF_SECRET]
    if config.enable_wandb:
        secrets.append(modal.Secret.from_name("wandb-secret"))
    if (
        config.trainer_type == "sft"
        and (config.task_spec or {}).get("task_family") == "generation"
        and _maybe_none(os.environ.get("OPENAI_API_KEY"))
    ):
        secret_payload = {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        }
        judge_model = _maybe_none(os.environ.get("POSTTRAINING_EVAL_JUDGE_MODEL"))
        if judge_model:
            secret_payload["POSTTRAINING_EVAL_JUDGE_MODEL"] = judge_model
        secrets.append(modal.Secret.from_dict(secret_payload))
    return secrets


def _csv_to_target_modules(value: str) -> list[str]:
    modules = [part.strip() for part in value.split(",") if part.strip()]
    return modules or list(DEFAULT_TARGET_MODULES)


def _maybe_none(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a string or null value, got {type(value).__name__}.")
    normalized = value.strip()
    return normalized or None


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = _maybe_none(os.environ.get(name))
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _is_qwen_family_base_model(model_id: Any) -> bool:
    return isinstance(model_id, str) and model_id.startswith("Qwen/")


def _is_conversational_value(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) and "role" in item for item in value)


def _normalize_text_value(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Expected `{field_name}` to be populated.")
    if isinstance(value, str):
        return value
    raise ValueError(
        f"Expected `{field_name}` to be a string or conversational messages list; got {type(value).__name__}."
    )


def _normalize_label_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "chosen", "good", "positive"}:
            return True
        if normalized in {"false", "0", "no", "rejected", "bad", "negative"}:
            return False
    raise ValueError(
        "Label values must be bool-like. Expected one of: true/false, 1/0, yes/no, chosen/rejected, good/bad."
    )


def _resolve_seed_artifact_path(config: TrainConfig) -> Path:
    if not config.seed_artifact:
        raise ValueError("seed_artifact is required for this trainer.")
    raw_path = Path(config.seed_artifact)
    resolved = raw_path if raw_path.is_absolute() else CHECKPOINTS_DIR / raw_path
    return resolved


def _write_run_config(config: TrainConfig) -> None:
    import yaml

    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with config.run_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_serializable_dict(), handle, sort_keys=False)


def _load_yaml_mapping(config_path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required locally to load YAML configs for this script. Install it with `pip install pyyaml`."
        ) from exc

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must parse to a top-level mapping/object.")
    return raw


def _config_from_mapping(raw: dict[str, Any]) -> TrainConfig:
    config_data = dict(raw)

    for field_name in ("base_model_revision", "dataset_config", "eval_split", "seed_artifact"):
        if field_name in config_data:
            config_data[field_name] = _maybe_none(config_data[field_name])

    if "target_modules" in config_data:
        target_modules = config_data["target_modules"]
        if isinstance(target_modules, str):
            config_data["target_modules"] = _csv_to_target_modules(target_modules)
        elif target_modules is None:
            config_data["target_modules"] = list(DEFAULT_TARGET_MODULES)
        elif not isinstance(target_modules, list):
            raise ValueError("target_modules must be either a YAML list or a comma-separated string.")

    if "prepared_dataset_manifest" in config_data and config_data["prepared_dataset_manifest"] is not None:
        if not isinstance(config_data["prepared_dataset_manifest"], dict):
            raise ValueError("prepared_dataset_manifest must be a mapping/object when provided.")

    return TrainConfig(**config_data)


def _load_single_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
) -> Any:
    from datasets import load_dataset

    dataset_kwargs: dict[str, Any] = {
        "path": dataset_name,
        "split": split,
        "cache_dir": str(DATASET_CACHE_DIR),
    }
    if dataset_config:
        dataset_kwargs["name"] = dataset_config
    return load_dataset(**dataset_kwargs)


def _normalized_probabilities(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Prepared dataset manifest weights must sum to a positive number.")
    return [weight / total for weight in weights]


def _label_mapping_candidates(value: Any) -> list[str]:
    candidates = [str(value)]
    if isinstance(value, float) and value.is_integer():
        candidates.append(str(int(value)))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped != value:
            candidates.append(stripped)
        try:
            as_float = float(stripped)
        except ValueError:
            as_float = None
        if as_float is not None and as_float.is_integer():
            candidates.append(str(int(as_float)))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _resolve_label_mapping_value(
    raw_label: Any,
    label_mapping: dict[str, Any] | None,
    field_name: str,
    dataset_id: str | None = None,
) -> str:
    if label_mapping:
        for candidate in _label_mapping_candidates(raw_label):
            if candidate in label_mapping:
                return str(label_mapping[candidate])
        dataset_prefix = (
            f"Prepared dataset `{dataset_id}` " if isinstance(dataset_id, str) and dataset_id.strip() else ""
        )
        raise ValueError(
            f"{dataset_prefix}could not map `{field_name}` value `{raw_label}` through explicit value_mapping. "
            "Add a string key for this raw label or remove value_mapping for passthrough labels."
        )
    return _stringify_manifest_value(raw_label, field_name)


def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


def _extract_template_variables(template: str) -> list[str]:
    return list(dict.fromkeys(match.group(1).strip() for match in re.finditer(r"{([^{}]+)}", template) if match.group(1).strip()))


def _looks_like_image_payload_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, dict):
        normalized_keys = {str(key).strip().lower() for key in value.keys()}
        if "bytes" in normalized_keys or "blob" in normalized_keys or "pixel_values" in normalized_keys:
            return True
        if (
            ("path" in normalized_keys or "src" in normalized_keys or "url" in normalized_keys)
            and ("bytes" in normalized_keys or "height" in normalized_keys or "width" in normalized_keys)
        ):
            return True
    if isinstance(value, str):
        return value.strip().lower().startswith("data:image/")
    return False


def _sample_dataset_column_values(dataset: Any, column_name: str, *, limit: int = 8) -> list[Any]:
    samples: list[Any] = []
    for index in range(min(len(dataset), limit)):
        row = dataset[index]
        if isinstance(row, dict) and column_name in row:
            samples.append(row[column_name])
    return samples


def _find_non_text_required_columns(dataset: Any, column_names: list[str]) -> list[str]:
    unsupported_columns: list[str] = []
    for column_name in column_names:
        if any(pattern.search(column_name) for pattern in NON_TEXT_COLUMN_PATTERNS):
            unsupported_columns.append(column_name)
            continue
        sample_values = _sample_dataset_column_values(dataset, column_name)
        if any(_looks_like_image_payload_value(value) for value in sample_values):
            unsupported_columns.append(column_name)
    return unsupported_columns


def _stringify_manifest_value(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if _looks_like_image_payload_value(value):
        raise ValueError(
            f"Expected `{field_name}` to resolve to plain text, not an image/blob payload."
        )
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _is_missing_manifest_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _build_label_space_text(label_mapping: dict[str, Any] | None) -> str:
    if not label_mapping:
        return ""
    label_values = [str(value) for value in label_mapping.values()]
    return ", ".join(dict.fromkeys(label_values))


def _require_manifest_mapping(entry: dict[str, Any], key: str) -> dict[str, Any]:
    value = entry.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Prepared dataset manifest entry `{key}` must be a mapping/object.")
    return value


def _require_manifest_string(entry: dict[str, Any], key: str, dataset_id: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prepared dataset `{dataset_id}` must define a non-empty `{key}`.")
    return value.strip()


def _require_source_column(field_mapping: dict[str, Any], key: str, dataset_id: str) -> str:
    value = field_mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Prepared dataset `{dataset_id}` is missing field_mapping.{key} for its transform preset."
        )
    return value.strip()


def _assert_columns_exist(dataset: Any, columns: list[str], dataset_id: str) -> None:
    available = set(dataset.column_names)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"Prepared dataset `{dataset_id}` is missing required columns {missing}. "
            f"Available columns: {sorted(available)}"
        )


def _require_normalization_field(fields: dict[str, Any], key: str, dataset_id: str) -> dict[str, Any]:
    value = fields.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Prepared dataset `{dataset_id}` normalization.fields.{key} must be an object.")
    return value


def _normalization_field_columns(field_spec: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    source_column = field_spec.get("source_column")
    if isinstance(source_column, str) and source_column.strip():
        columns.append(source_column.strip())

    template = field_spec.get("template")
    if isinstance(template, str) and template.strip():
        columns.extend(_extract_template_variables(template))

    return list(dict.fromkeys(column for column in columns if column))


def _render_normalization_field(
    example: dict[str, Any],
    field_spec: dict[str, Any],
    field_name: str,
    dataset_id: str,
) -> str:
    raw_value = _resolve_normalization_raw_value(example, field_spec, field_name)
    value_mapping = field_spec.get("value_mapping")

    if value_mapping is not None and not isinstance(value_mapping, dict):
        raise ValueError(f"normalization.fields.{field_name}.value_mapping must be an object when provided.")

    if isinstance(value_mapping, dict):
        return _resolve_label_mapping_value(
            raw_value,
            value_mapping,
            f"normalization.fields.{field_name}",
            dataset_id,
        )
    return _stringify_manifest_value(raw_value, field_name)


def _resolve_normalization_raw_value(
    example: dict[str, Any],
    field_spec: dict[str, Any],
    field_name: str,
) -> Any:
    source_column = field_spec.get("source_column")
    template = field_spec.get("template")

    has_source = isinstance(source_column, str) and source_column.strip()
    has_template = isinstance(template, str) and template.strip()
    if has_source == has_template:
        raise ValueError(
            f"normalization.fields.{field_name} must define exactly one of source_column or template."
        )

    if has_source:
        raw_value = example[source_column.strip()]
    else:
        template_vars = _extract_template_variables(template)
        return _render_prompt_template(
            template,
            {variable: example.get(variable) for variable in template_vars},
        )

    return raw_value


def _drop_invalid_required_sft_rows(
    dataset_id: str,
    dataset: Any,
    required_fields: tuple[str, ...],
    *,
    split_name: str,
    selected_target_column: str | None,
) -> tuple[Any, dict[str, Any]]:
    total_rows = len(dataset)
    kept_indices: list[int] = []
    dropped_rows_by_field = {field_name: 0 for field_name in required_fields}

    field_values = {field_name: dataset[field_name] for field_name in required_fields}

    for index in range(total_rows):
        missing_fields = [
            field_name
            for field_name in required_fields
            if _is_missing_manifest_value(field_values[field_name][index])
        ]
        if missing_fields:
            for field_name in missing_fields:
                dropped_rows_by_field[field_name] += 1
            continue
        kept_indices.append(index)

    kept_rows = len(kept_indices)
    dropped_rows = total_rows - kept_rows
    diagnostic = {
        "dataset": dataset_id,
        "split": split_name,
        "selected_target_column": selected_target_column,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows_invalid_examples": dropped_rows,
        "dropped_fraction": (dropped_rows / total_rows) if total_rows else 0.0,
        "dropped_rows_by_field": dropped_rows_by_field,
    }
    for field_name, dropped_count in dropped_rows_by_field.items():
        diagnostic[f"dropped_rows_missing_{field_name}"] = dropped_count
    if "completion" in dropped_rows_by_field:
        diagnostic["dropped_rows_missing_target"] = dropped_rows_by_field["completion"]

    if dropped_rows <= 0:
        return dataset, diagnostic
    return dataset.select(kept_indices), diagnostic


def _summarize_preprocessing_diagnostics(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    total_rows = sum(int(entry.get("total_rows", 0)) for entry in entries)
    kept_rows = sum(int(entry.get("kept_rows", 0)) for entry in entries)
    dropped_invalid_rows = sum(int(entry.get("dropped_rows_invalid_examples", 0)) for entry in entries)
    dropped_missing_target_rows = sum(int(entry.get("dropped_rows_missing_target", 0)) for entry in entries)
    dropped_rows_by_field: dict[str, int] = {}

    for entry in entries:
        entry_field_counts = entry.get("dropped_rows_by_field")
        if not isinstance(entry_field_counts, dict):
            continue
        for field_name, count in entry_field_counts.items():
            if not isinstance(field_name, str):
                continue
            dropped_rows_by_field[field_name] = dropped_rows_by_field.get(field_name, 0) + int(count)

    summary = {
        "invalid_sft_example_filtering": {
            "policy": "drop_rows_with_blank_required_sft_fields",
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "dropped_rows_invalid_examples": dropped_invalid_rows,
            "dropped_fraction": (dropped_invalid_rows / total_rows) if total_rows else 0.0,
            "dropped_rows_by_field": dropped_rows_by_field,
            "datasets": entries,
        }
    }

    if dropped_missing_target_rows > 0:
        summary["missing_target_label_filtering"] = {
            "policy": "drop_missing_selected_target_labels",
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "dropped_rows_missing_target": dropped_missing_target_rows,
            "dropped_fraction": (dropped_missing_target_rows / total_rows) if total_rows else 0.0,
            "datasets": entries,
        }

    return summary


def _drop_blank_normalized_sft_rows(
    dataset: Any,
    required_fields: tuple[str, ...],
    *,
    dataset_label: str,
) -> Any:
    total_rows = len(dataset)
    kept_indices: list[int] = []
    field_values = {field_name: dataset[field_name] for field_name in required_fields}

    for index in range(total_rows):
        if all(
            not _is_missing_manifest_value(field_values[field_name][index])
            for field_name in required_fields
        ):
            kept_indices.append(index)

    if len(kept_indices) == total_rows:
        return dataset
    if not kept_indices:
        required_field_list = ", ".join(required_fields)
        raise ValueError(
            f"{dataset_label} has no usable rows after dropping examples with blank required "
            f"field(s): {required_field_list}."
        )
    return dataset.select(kept_indices)


def _transform_dataset_with_normalization(dataset_id: str, dataset: Any, entry: dict[str, Any]) -> Any:
    normalization = entry.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"Prepared dataset `{dataset_id}` normalization must be an object.")

    shape = _require_manifest_string(normalization, "shape", dataset_id)
    fields = _require_manifest_mapping(normalization, "fields")
    source_columns = normalization.get("source_columns")
    if not isinstance(source_columns, list) or not source_columns:
        raise ValueError(f"Prepared dataset `{dataset_id}` normalization.source_columns must be a non-empty array.")

    normalized_source_columns: list[str] = []
    for column in source_columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"Prepared dataset `{dataset_id}` normalization.source_columns must contain only non-empty strings."
            )
        normalized_source_columns.append(column.strip())

    required_columns = set(normalized_source_columns)

    if shape == "text":
        text_field = _require_normalization_field(fields, "text", dataset_id)
        required_columns.update(_normalization_field_columns(text_field))
        _assert_columns_exist(dataset, sorted(required_columns), dataset_id)
        unsupported_columns = _find_non_text_required_columns(dataset, sorted(required_columns))
        if unsupported_columns:
            raise ValueError(
                f"Prepared dataset `{dataset_id}` normalization references unsupported image/blob columns "
                f"{unsupported_columns}. This trainer is text-only; remove those columns from "
                "`normalization.source_columns` and prompt templates."
            )
        return dataset.map(
            lambda example: {"text": _render_normalization_field(example, text_field, "text", dataset_id)},
            remove_columns=list(dataset.column_names),
            desc=f"Preparing {dataset_id} with normalization.text",
        )

    if shape == "prompt_completion":
        prompt_field = _require_normalization_field(fields, "prompt", dataset_id)
        completion_field = _require_normalization_field(fields, "completion", dataset_id)
        required_columns.update(_normalization_field_columns(prompt_field))
        required_columns.update(_normalization_field_columns(completion_field))
        _assert_columns_exist(dataset, sorted(required_columns), dataset_id)
        unsupported_columns = _find_non_text_required_columns(dataset, sorted(required_columns))
        if unsupported_columns:
            raise ValueError(
                f"Prepared dataset `{dataset_id}` normalization references unsupported image/blob columns "
                f"{unsupported_columns}. This trainer is text-only; remove those columns from "
                "`normalization.source_columns` and prompt templates."
            )
        return dataset.map(
            lambda example: {
                "prompt": _render_normalization_field(example, prompt_field, "prompt", dataset_id),
                "completion": _render_normalization_field(
                    example,
                    completion_field,
                    "completion",
                    dataset_id,
                ),
            },
            remove_columns=list(dataset.column_names),
            desc=f"Preparing {dataset_id} with normalization.prompt_completion",
        )

    raise ValueError(
        f"Prepared dataset `{dataset_id}` normalization.shape `{shape}` is not supported."
    )

def _prepare_prepared_dataset_entry(
    dataset: Any,
    entry: dict[str, Any],
    *,
    split_name: str,
) -> tuple[Any, dict[str, Any] | None]:
    dataset_id = _require_manifest_string(entry, "dataset", entry.get("dataset", "dataset"))
    normalization = entry.get("normalization")
    if normalization is None:
        raise ValueError(
            f"Prepared dataset `{dataset_id}` must define `normalization`. "
            "Legacy transform_preset manifests are no longer supported."
        )
    if not isinstance(normalization, dict):
        raise ValueError(f"Prepared dataset `{dataset_id}` normalization must be an object.")

    diagnostics = None
    prepared_dataset = _transform_dataset_with_normalization(dataset_id, dataset, entry)
    shape = _require_manifest_string(normalization, "shape", dataset_id)
    if shape == "text":
        prepared_dataset, diagnostics = _drop_invalid_required_sft_rows(
            dataset_id,
            prepared_dataset,
            ("text",),
            split_name=split_name,
            selected_target_column=None,
        )
    elif shape == "prompt_completion":
        prepared_dataset, diagnostics = _drop_invalid_required_sft_rows(
            dataset_id,
            prepared_dataset,
            ("prompt", "completion"),
            split_name=split_name,
            selected_target_column=_maybe_none(entry.get("selected_target_column")),
        )

    return prepared_dataset, diagnostics


def _dataset_column_signature(dataset: Any) -> tuple[str, ...]:
    return tuple(sorted(dataset.column_names))


def _mix_datasets(datasets_to_mix: list[Any], probabilities: list[float]) -> Any:
    from datasets import interleave_datasets

    if len(datasets_to_mix) == 1:
        return datasets_to_mix[0]
    return interleave_datasets(
        datasets_to_mix,
        probabilities=_normalized_probabilities(probabilities),
        seed=42,
        stopping_strategy="all_exhausted",
    )


def _concat_datasets(datasets_to_concat: list[Any]) -> Any:
    from datasets import concatenate_datasets

    if len(datasets_to_concat) == 1:
        return datasets_to_concat[0]
    return concatenate_datasets(datasets_to_concat)


def _resolve_source_splits_for_entry(entry: dict[str, Any], config: TrainConfig) -> list[str]:
    raw_source_splits = entry.get("source_splits")
    if isinstance(raw_source_splits, list):
        source_splits: list[str] = []
        for raw_split in raw_source_splits:
            normalized = _maybe_none(raw_split)
            if normalized and normalized not in source_splits:
                source_splits.append(normalized)
        if source_splits:
            return source_splits

    fallback_split = _maybe_none(entry.get("train_split")) or config.train_split
    if not fallback_split:
        raise ValueError("Prepared dataset manifest entry is missing both source_splits and train_split.")
    return [fallback_split]


def _annotate_dataset_provenance(dataset: Any, dataset_id: str, split_name: str) -> Any:
    return dataset.map(
        lambda example: {
            **example,
            PROVENANCE_DATASET_COLUMN: dataset_id,
            PROVENANCE_SPLIT_COLUMN: split_name,
        },
        desc=f"Annotating provenance for {dataset_id}:{split_name}",
    )


def _dataset_distribution_by_provenance(dataset: Any) -> dict[str, int]:
    if dataset is None or PROVENANCE_DATASET_COLUMN not in set(dataset.column_names):
        return {}

    counts: dict[str, int] = {}
    for dataset_id in dataset[PROVENANCE_DATASET_COLUMN]:
        normalized = str(dataset_id).strip()
        counts[normalized] = counts.get(normalized, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _create_random_holdout(
    dataset: Any,
    *,
    fraction: float,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    total_rows = len(dataset)
    if total_rows <= 1:
        return dataset, dataset.select([]), {
            "created_holdout": False,
            "strategy": "deterministic_random_holdout",
            "fraction": fraction,
            "seed": seed,
            "train_examples": total_rows,
            "eval_examples": 0,
        }

    rng = random.Random(seed)
    indices = list(range(total_rows))
    rng.shuffle(indices)
    eval_count = max(1, round(total_rows * fraction))
    eval_count = min(eval_count, total_rows - 1)
    eval_indices = sorted(indices[:eval_count])
    train_indices = sorted(indices[eval_count:])

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)
    return train_dataset, eval_dataset, {
        "created_holdout": True,
        "strategy": "deterministic_random_holdout",
        "fraction": fraction,
        "seed": seed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    }


def _reweight_train_dataset_by_provenance(train_dataset: Any, dataset_weights: dict[str, float]) -> Any:
    if (
        train_dataset is None
        or len(train_dataset) <= 0
        or PROVENANCE_DATASET_COLUMN not in set(train_dataset.column_names)
    ):
        return train_dataset

    indices_by_dataset: dict[str, list[int]] = {}
    for index, dataset_id in enumerate(train_dataset[PROVENANCE_DATASET_COLUMN]):
        normalized = str(dataset_id).strip()
        indices_by_dataset.setdefault(normalized, []).append(index)

    datasets_to_mix: list[Any] = []
    probabilities: list[float] = []
    for dataset_id, weight in dataset_weights.items():
        indices = indices_by_dataset.get(dataset_id, [])
        if not indices:
            continue
        datasets_to_mix.append(train_dataset.select(indices))
        probabilities.append(float(weight))

    if not datasets_to_mix:
        return train_dataset
    return _mix_datasets(datasets_to_mix, probabilities)


def _load_sft_prepared_manifest_datasets(
    config: TrainConfig,
) -> tuple[Any, Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    manifest = config.prepared_dataset_manifest or {}
    selected_datasets = manifest.get("selected_datasets")
    if not isinstance(selected_datasets, list) or not selected_datasets:
        raise ValueError(
            "prepared_dataset_manifest must include a non-empty `selected_datasets` list."
        )

    prepared_pools: list[Any] = []
    dataset_weights: dict[str, float] = {}
    expected_signature: tuple[str, ...] | None = None
    preprocessing_entries: list[dict[str, Any]] = []
    source_splits_by_dataset: dict[str, list[str]] = {}

    for raw_entry in selected_datasets:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each prepared_dataset_manifest selected_datasets entry must be an object.")

        dataset_id = _require_manifest_string(raw_entry, "dataset", raw_entry.get("dataset", "dataset"))
        dataset_config = _maybe_none(raw_entry.get("dataset_config"))
        source_splits = _resolve_source_splits_for_entry(raw_entry, config)
        source_splits_by_dataset[dataset_id] = source_splits

        raw_weight = raw_entry.get("weight", 1)
        try:
            train_weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prepared dataset `{dataset_id}` has invalid weight `{raw_weight}`."
            ) from exc
        if train_weight <= 0:
            raise ValueError(f"Prepared dataset `{dataset_id}` weight must be positive.")

        per_split_datasets: list[Any] = []
        for split_name in source_splits:
            source_dataset = _load_single_dataset(
                dataset_name=dataset_id,
                dataset_config=dataset_config,
                split=split_name,
            )
            prepared_split, split_diagnostics = _prepare_prepared_dataset_entry(
                source_dataset,
                raw_entry,
                split_name=split_name,
            )
            if split_diagnostics is not None:
                preprocessing_entries.append(split_diagnostics)
            if len(prepared_split) <= 0:
                continue

            prepared_split = _annotate_dataset_provenance(prepared_split, dataset_id, split_name)
            current_signature = _dataset_column_signature(prepared_split)
            if expected_signature is None:
                expected_signature = current_signature
            elif current_signature != expected_signature:
                raise ValueError(
                    "All selected datasets in prepared_dataset_manifest must normalize to the same "
                    f"column shape. Expected {expected_signature}, got {current_signature} for `{dataset_id}`."
                )
            per_split_datasets.append(prepared_split)

        if not per_split_datasets:
            raise ValueError(
                f"Prepared dataset `{dataset_id}` has no usable rows across source_splits {source_splits}."
            )

        prepared_pools.append(_concat_datasets(per_split_datasets))
        dataset_weights[dataset_id] = train_weight

    combined_pool = _concat_datasets(prepared_pools)
    evaluation_plan = config.evaluation_plan or {}
    holdout_fraction = float(evaluation_plan.get("holdout_fraction", 0.1))
    deterministic_seed = int(evaluation_plan.get("deterministic_seed", 42))
    use_stratified_holdout = (
        (config.task_spec or {}).get("task_family") == "classification"
        and "completion" in set(combined_pool.column_names)
    )

    if use_stratified_holdout:
        raw_train_dataset, raw_eval_dataset, holdout_metadata = _create_stratified_holdout(
            combined_pool,
            fraction=holdout_fraction,
            seed=deterministic_seed,
        )
    else:
        raw_train_dataset, raw_eval_dataset, holdout_metadata = _create_random_holdout(
            combined_pool,
            fraction=holdout_fraction,
            seed=deterministic_seed,
        )

    weighted_train_dataset = _reweight_train_dataset_by_provenance(raw_train_dataset, dataset_weights)
    if raw_eval_dataset is not None and len(raw_eval_dataset) <= 0:
        raw_eval_dataset = None

    if holdout_metadata is not None:
        holdout_metadata["source_splits_by_dataset"] = source_splits_by_dataset
        holdout_metadata["pre_weight_train_examples"] = len(raw_train_dataset)
        holdout_metadata["weighted_train_examples"] = len(weighted_train_dataset)
        holdout_metadata["train_dataset_distribution"] = _dataset_distribution_by_provenance(raw_train_dataset)
        holdout_metadata["eval_dataset_distribution"] = (
            _dataset_distribution_by_provenance(raw_eval_dataset) if raw_eval_dataset is not None else {}
        )

    return (
        weighted_train_dataset,
        raw_eval_dataset,
        _summarize_preprocessing_diagnostics(preprocessing_entries),
        holdout_metadata,
    )


def _load_prepared_manifest_datasets(config: TrainConfig) -> tuple[Any, Any, dict[str, Any] | None]:
    manifest = config.prepared_dataset_manifest or {}
    selected_datasets = manifest.get("selected_datasets")
    if not isinstance(selected_datasets, list) or not selected_datasets:
        raise ValueError(
            "prepared_dataset_manifest must include a non-empty `selected_datasets` list."
        )

    train_datasets: list[Any] = []
    train_weights: list[float] = []
    eval_datasets: list[Any] = []
    expected_signature: tuple[str, ...] | None = None
    eval_signature: tuple[str, ...] | None = None
    preprocessing_entries: list[dict[str, Any]] = []

    for raw_entry in selected_datasets:
        if not isinstance(raw_entry, dict):
            raise ValueError("Each prepared_dataset_manifest selected_datasets entry must be an object.")
        dataset_id = _require_manifest_string(raw_entry, "dataset", raw_entry.get("dataset", "dataset"))
        dataset_config = _maybe_none(raw_entry.get("dataset_config"))
        train_split = _maybe_none(raw_entry.get("train_split")) or config.train_split
        if not train_split:
            raise ValueError(f"Prepared dataset `{dataset_id}` must define train_split.")

        raw_weight = raw_entry.get("weight", 1)
        try:
            train_weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Prepared dataset `{dataset_id}` has invalid weight `{raw_weight}`."
            ) from exc
        if train_weight <= 0:
            raise ValueError(f"Prepared dataset `{dataset_id}` weight must be positive.")

        train_dataset = _load_single_dataset(
            dataset_name=dataset_id,
            dataset_config=dataset_config,
            split=train_split,
        )
        prepared_train, train_diagnostics = _prepare_prepared_dataset_entry(
            train_dataset,
            raw_entry,
            split_name=train_split,
        )
        if train_diagnostics is not None:
            preprocessing_entries.append(train_diagnostics)
        if len(prepared_train) <= 0:
            selected_target_column = (
                train_diagnostics.get("selected_target_column") if isinstance(train_diagnostics, dict) else None
            )
            raise ValueError(
                f"Prepared dataset `{dataset_id}` split `{train_split}` has no usable rows after dropping "
                f"examples with missing target labels{f' for `{selected_target_column}`' if selected_target_column else ''}."
            )
        signature = _dataset_column_signature(prepared_train)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                "All selected datasets in prepared_dataset_manifest must normalize to the same "
                f"column shape. Expected {expected_signature}, got {signature} for `{dataset_id}`."
            )

        train_datasets.append(prepared_train)
        train_weights.append(train_weight)

        eval_split = _maybe_none(raw_entry.get("eval_split")) or config.eval_split
        if eval_split:
            eval_dataset = _load_single_dataset(
                dataset_name=dataset_id,
                dataset_config=dataset_config,
                split=eval_split,
            )
            prepared_eval, eval_diagnostics = _prepare_prepared_dataset_entry(
                eval_dataset,
                raw_entry,
                split_name=eval_split,
            )
            if eval_diagnostics is not None:
                preprocessing_entries.append(eval_diagnostics)
            if len(prepared_eval) <= 0:
                continue
            current_eval_signature = _dataset_column_signature(prepared_eval)
            if eval_signature is None:
                eval_signature = current_eval_signature
            elif current_eval_signature != eval_signature:
                raise ValueError(
                    "All prepared eval datasets must normalize to the same column shape. "
                    f"Expected {eval_signature}, got {current_eval_signature} for `{dataset_id}`."
                )
            eval_datasets.append(prepared_eval)

    mixed_train = _mix_datasets(train_datasets, train_weights)
    mixed_eval = _concat_datasets(eval_datasets) if eval_datasets else None
    return mixed_train, mixed_eval, _summarize_preprocessing_diagnostics(preprocessing_entries)


def _load_datasets(config: TrainConfig) -> tuple[Any, Any, dict[str, Any] | None]:
    if config.dataset_source_type == "prepared_manifest":
        return _load_prepared_manifest_datasets(config)

    train_dataset = _load_single_dataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.train_split,
    )

    eval_dataset = None
    if config.eval_split:
        eval_dataset = _load_single_dataset(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            split=config.eval_split,
        )

    return train_dataset, eval_dataset, None


def _label_distribution(dataset: Any) -> dict[str, int]:
    if dataset is None or "completion" not in set(dataset.column_names):
        return {}
    counts: dict[str, int] = {}
    for value in dataset["completion"]:
        label = str(value).strip()
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _create_stratified_holdout(
    dataset: Any,
    *,
    fraction: float,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    if "completion" not in set(dataset.column_names):
        raise ValueError("Cannot create a classification holdout without a `completion` column.")

    label_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(dataset["completion"]):
        label_to_indices.setdefault(str(label).strip(), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    eval_indices: list[int] = []

    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_indices.extend(shuffled)
            continue
        eval_count = max(1, round(len(shuffled) * fraction))
        eval_count = min(eval_count, len(shuffled) - 1)
        eval_indices.extend(shuffled[:eval_count])
        train_indices.extend(shuffled[eval_count:])

    if not eval_indices and len(dataset) > 1:
        all_indices = list(range(len(dataset)))
        rng.shuffle(all_indices)
        eval_indices.append(all_indices[0])
        train_index_set = set(all_indices[1:])
        train_indices = [index for index in train_indices if index in train_index_set]
        if not train_indices:
            train_indices = all_indices[1:]

    train_indices = sorted(dict.fromkeys(train_indices))
    eval_indices = sorted(dict.fromkeys(eval_indices))
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)

    return train_dataset, eval_dataset, {
        "created_holdout": True,
        "strategy": "stratified_completion_holdout",
        "fraction": fraction,
        "seed": seed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "train_label_distribution": _label_distribution(train_dataset),
        "eval_label_distribution": _label_distribution(eval_dataset),
    }


def _normalized_label(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^[\"'`\s]+|[\"'`\s]+$", "", text)
    return text.casefold()


def _sample_eval_dataset(dataset: Any, *, max_examples: int, seed: int) -> Any:
    if len(dataset) <= max_examples:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return dataset.select(sorted(indices[:max_examples]))


def _truncate_preview(value: Any, limit: int = 200) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _comparison_sample_policy(total_eval_examples: int, *, max_examples: int, seed: int) -> dict[str, Any]:
    return {
        "max_cases": max_examples,
        "sampled_cases": min(total_eval_examples, max_examples),
        "total_eval_examples": total_eval_examples,
        "seed": seed,
    }


def _predict_completion_label(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 16) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    completion_ids = generated[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _predict_classification_outputs(model: Any, tokenizer: Any, eval_dataset: Any) -> dict[str, Any]:
    gold_labels = [str(value).strip() for value in eval_dataset["completion"]]
    normalized_to_label = {}
    for label in gold_labels:
        normalized = _normalized_label(label)
        if normalized and normalized not in normalized_to_label:
            normalized_to_label[normalized] = label

    predictions: list[str | None] = []
    raw_predictions: list[str] = []
    for prompt in eval_dataset["prompt"]:
        raw_prediction = _predict_completion_label(model, tokenizer, str(prompt))
        raw_predictions.append(raw_prediction)
        canonical = normalized_to_label.get(_normalized_label(raw_prediction))
        predictions.append(canonical)

    return {
        "gold_labels": gold_labels,
        "labels": sorted(set(gold_labels)),
        "predictions": predictions,
        "raw_predictions": raw_predictions,
    }


def _build_classification_eval_result(eval_dataset: Any, prediction_bundle: dict[str, Any]) -> dict[str, Any]:
    gold_labels = prediction_bundle["gold_labels"]
    labels = prediction_bundle["labels"]
    predictions = prediction_bundle["predictions"]
    raw_predictions = prediction_bundle["raw_predictions"]

    total = len(gold_labels)
    invalid_predictions = sum(1 for prediction in predictions if prediction is None)
    valid_predictions = [prediction if prediction is not None else "__invalid__" for prediction in predictions]
    accuracy = (
        sum(1 for expected, actual in zip(gold_labels, predictions) if expected == actual) / total
        if total
        else 0.0
    )
    macro_f1 = _macro_f1(labels, gold_labels, valid_predictions)
    coverage = len({prediction for prediction in predictions if prediction in labels}) / len(labels) if labels else 0.0

    confusion_matrix: dict[str, dict[str, int]] = {}
    for expected, actual in zip(gold_labels, valid_predictions):
        row = confusion_matrix.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1

    return {
        "task_family": "classification",
        "strategy": "offline_prompt_completion_eval",
        "sampled_examples": total,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "invalid_label_rate": (invalid_predictions / total) if total else 1.0,
            "label_coverage": coverage,
        },
        "label_distributions": {
            "gold": _label_distribution(eval_dataset),
            "predicted": dict(
                sorted(
                    (
                        (
                            label,
                            sum(1 for prediction in predictions if prediction == label),
                        )
                        for label in labels
                    ),
                    key=lambda item: item[0],
                )
            ),
        },
        "confusion_matrix": confusion_matrix,
        "sample_predictions": [
            {
                "prompt_preview": str(prompt)[:160],
                "gold": gold,
                "prediction": prediction,
                "raw_prediction": raw_prediction[:80],
            }
            for prompt, gold, prediction, raw_prediction in zip(
                eval_dataset["prompt"][:10],
                gold_labels[:10],
                predictions[:10],
                raw_predictions[:10],
            )
        ],
    }


def _macro_f1(labels: list[str], gold: list[str], predicted: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(1 for expected, actual in zip(gold, predicted) if expected == label and actual == label)
        false_positive = sum(1 for expected, actual in zip(gold, predicted) if expected != label and actual == label)
        false_negative = sum(1 for expected, actual in zip(gold, predicted) if expected == label and actual != label)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_classification_dataset(
    model: Any,
    tokenizer: Any,
    eval_dataset: Any,
    *,
    max_examples: int,
    seed: int,
) -> dict[str, Any]:
    sampled_eval = _sample_eval_dataset(eval_dataset, max_examples=max_examples, seed=seed)
    prediction_bundle = _predict_classification_outputs(model, tokenizer, sampled_eval)
    return _build_classification_eval_result(sampled_eval, prediction_bundle)


def _classification_winner_from_metrics(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> str:
    baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0))
    candidate_accuracy = float(candidate_metrics.get("accuracy", 0.0))
    if candidate_accuracy > baseline_accuracy:
        return "candidate"
    if baseline_accuracy > candidate_accuracy:
        return "baseline"

    baseline_macro_f1 = float(baseline_metrics.get("macro_f1", 0.0))
    candidate_macro_f1 = float(candidate_metrics.get("macro_f1", 0.0))
    if candidate_macro_f1 > baseline_macro_f1:
        return "candidate"
    if baseline_macro_f1 > candidate_macro_f1:
        return "baseline"
    return "tie"


def _metric_deltas(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, candidate_value in candidate_metrics.items():
        baseline_value = baseline_metrics.get(key)
        if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[key] = float(candidate_value) - float(baseline_value)
    return deltas


def _clear_inference_model(model: Any | None) -> None:
    if model is None:
        return

    try:
        del model
    finally:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _selected_dataset_ids(config: TrainConfig) -> list[str]:
    if config.dataset_source_type != "prepared_manifest":
        return [config.dataset_name]
    return [
        entry.get("dataset")
        for entry in (config.prepared_dataset_manifest or {}).get("selected_datasets", [])
        if isinstance(entry, dict) and entry.get("dataset")
    ]


def _should_build_merged_artifact(config: TrainConfig) -> bool:
    return bool(config.merge_after_train or _is_qwen_family_base_model(config.base_model))


def _emit_structured_lifecycle_event(event_type: str, **data: Any) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event_type,
        **data,
    }
    print(f"{STRUCTURED_LIFECYCLE_EVENT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


def _emit_progress_update(message: str) -> None:
    print(f"{STRUCTURED_PROGRESS_PREFIX}{message}", flush=True)


def _build_training_result(
    config: TrainConfig,
    *,
    trainer_meta: dict[str, Any],
    normalized_train_dataset: Any,
    normalized_eval_dataset: Any | None,
    training_output: Any,
    preprocessing_diagnostics: dict[str, Any] | None,
    holdout_metadata: dict[str, Any] | None,
    resumed_from_checkpoint: str | None,
    include_merged_dir: bool,
) -> dict[str, Any]:
    return {
        "trainer_type": config.trainer_type,
        "trainer_class": trainer_meta["trainer"],
        "base_model": config.base_model,
        "dataset_name": config.dataset_name,
        "dataset_source_type": config.dataset_source_type,
        "selected_datasets": _selected_dataset_ids(config),
        "output_name": config.output_name,
        "gpu_type": config.gpu_type,
        "modal_gpu_type": config.modal_gpu_type,
        "learning_rate": config.resolved_learning_rate,
        "task_spec": config.task_spec,
        "training_estimate": config.training_estimate,
        "preprocessing_diagnostics": preprocessing_diagnostics,
        "train_examples": len(normalized_train_dataset),
        "eval_examples": len(normalized_eval_dataset) if normalized_eval_dataset is not None else 0,
        "checkpoint_dir": str(config.checkpoints_dir),
        "final_adapter_dir": str(config.final_adapter_dir),
        "merged_dir": str(config.merged_dir) if include_merged_dir else None,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "global_step": int(getattr(training_output, "global_step", 0)),
        "training_loss": float(getattr(training_output, "training_loss", 0.0)),
        "evaluation": None,
        "holdout": holdout_metadata,
        "notes": [
            "Use final_adapter_dir for vLLM LoRA serving with --enable-lora.",
            "Use merged_dir for direct vLLM base-model serving when merge_after_train=True or for Qwen-family deployments.",
        ],
    }


def _load_adapter_inference_model(config: TrainConfig) -> Any:
    from peft import PeftModel

    adapter_dir = config.final_adapter_dir
    if not adapter_dir.exists():
        raise FileNotFoundError(f"final_adapter_dir does not exist: {adapter_dir}")

    model = _load_base_model(config)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.config.use_cache = False
    return model


def _predict_generation_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    return _predict_completion_label(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
    )


def _strip_thinking_blocks(text: str) -> str:
    value = str(text or "")
    if "<think>" not in value.lower():
        return value.strip()

    stripped = THINK_TAG_BLOCK_RE.sub("", value)
    stripped = re.sub(r"</?think>", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


_GENERATION_EVAL_TARGET_SECTION_MARKERS = (
    "\n\n### Clinical Note\n",
    "\n\n### Answer\n",
    "\n\n### Response\n",
    "\n\n### Completion\n",
)


def _extract_generation_eval_source_input(source_text: str) -> str:
    text = str(source_text).strip()
    for marker in _GENERATION_EVAL_TARGET_SECTION_MARKERS:
        marker_index = text.find(marker)
        if marker_index > 0:
            return text[:marker_index].rstrip()
    return text


def _build_grounded_generation_eval_prompt(prompt: str, source_text: str) -> str:
    grounded_source_text = _extract_generation_eval_source_input(source_text)
    return (
        "Answer the task below using only the provided source text.\n"
        "Do not use outside knowledge, placeholders, or unsupported details.\n"
        "Return only the final answer.\n\n"
        f"Task:\n{str(prompt).strip()}\n\n"
        f"Source text:\n{grounded_source_text}"
    )


def _extract_openai_output_text(response: dict[str, Any]) -> str | None:
    top_level = response.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    for output_item in response.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if (
                isinstance(content_item, dict)
                and content_item.get("type") == "output_text"
                and isinstance(content_item.get("text"), str)
                and content_item["text"].strip()
            ):
                return content_item["text"].strip()
    return None


def _call_openai_json_response(
    *,
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    api_key = _maybe_none(os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for generation comparison evaluation.")

    payload = {
        "model": model,
        "store": False,
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }

    request = urllib.request.Request(
        OPENAI_RESPONSES_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {_truncate_preview(body, 400)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    parsed = json.loads(body)
    output_text = _extract_openai_output_text(parsed)
    if not output_text:
        raise RuntimeError("OpenAI comparison call returned no output_text.")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI comparison call returned invalid JSON: {exc}") from exc


def _synthesize_generation_case(source_text: str, *, judge_model: str) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "prompt": {"type": "string"},
            "reference_answer": {"type": "string"},
            "rubric": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
            },
            "source_summary": {"type": "string"},
        },
        "required": ["prompt", "reference_answer", "rubric", "source_summary"],
    }

    result = _call_openai_json_response(
        model=judge_model,
        schema_name="generation_eval_case",
        schema=schema,
        system_prompt=(
            "You convert a raw post-training eval text sample into a deterministic evaluation case. "
            "Return only strict JSON."
        ),
        user_prompt=(
            "Given the raw eval text below, create:\n"
            "1. one realistic user prompt that the sample implies the assistant should answer\n"
            "2. one ideal reference answer in the same task/style\n"
            "3. a short rubric with 3-5 criteria\n"
            "4. a one-sentence source summary\n\n"
            f"Raw eval text:\n{_truncate_preview(source_text, 2400)}"
        ),
    )

    prompt = str(result.get("prompt", "")).strip()
    reference_answer = str(result.get("reference_answer", "")).strip()
    rubric = [str(item).strip() for item in result.get("rubric", []) if str(item).strip()]
    source_summary = str(result.get("source_summary", "")).strip()
    if not prompt or not reference_answer or len(rubric) < 3 or not source_summary:
        raise RuntimeError("OpenAI generation case synthesis returned incomplete fields.")

    return {
        "prompt": prompt,
        "reference_answer": reference_answer,
        "rubric": rubric,
        "source_summary": source_summary,
    }


def _judge_generation_outputs(
    *,
    prompt: str,
    source_text: str,
    reference_answer: str,
    rubric: list[str],
    baseline_output: str,
    candidate_output: str,
    judge_model: str,
    flip_order: bool,
) -> dict[str, Any]:
    output_a = candidate_output if flip_order else baseline_output
    output_b = baseline_output if flip_order else candidate_output
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["output_a", "output_b", "tie"],
            },
            "score_output_a": {"type": "number"},
            "score_output_b": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["winner", "score_output_a", "score_output_b", "reason"],
    }

    result = _call_openai_json_response(
        model=judge_model,
        schema_name="generation_eval_judgment",
        schema=schema,
        system_prompt=(
            "You compare two model outputs against a reference answer and rubric. "
            "Score each from 0 to 10, pick the better output, and return only strict JSON."
        ),
        user_prompt=(
            f"Prompt:\n{prompt}\n\n"
            f"Source eval text:\n{_truncate_preview(source_text, 1800)}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"Rubric:\n- " + "\n- ".join(rubric) + "\n\n"
            f"Output A:\n{output_a}\n\n"
            f"Output B:\n{output_b}"
        ),
    )

    winner = str(result.get("winner", "")).strip()
    score_output_a = float(result.get("score_output_a", 0.0))
    score_output_b = float(result.get("score_output_b", 0.0))
    reason = str(result.get("reason", "")).strip()
    if winner not in {"output_a", "output_b", "tie"}:
        raise RuntimeError(f"Unexpected generation judge winner: {winner}")

    mapped_winner = winner
    if winner == "output_a":
        mapped_winner = "candidate" if flip_order else "baseline"
    elif winner == "output_b":
        mapped_winner = "baseline" if flip_order else "candidate"

    baseline_score = score_output_b if flip_order else score_output_a
    candidate_score = score_output_a if flip_order else score_output_b
    return {
        "winner": mapped_winner,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "reason": reason,
    }


def _resolve_generation_eval_openai_max_workers(total_items: int) -> int:
    if total_items <= 1:
        return 1
    configured = _read_positive_int_env(
        "POSTTRAINING_EVAL_OPENAI_MAX_WORKERS",
        DEFAULT_GENERATION_EVAL_OPENAI_MAX_WORKERS,
    )
    return max(1, min(total_items, configured))


def _run_generation_eval_openai_tasks_with_progress(
    items: list[Any],
    *,
    progress_prefix: str,
    worker: Any,
) -> list[Any]:
    total_items = len(items)
    _emit_progress_update(f"{progress_prefix} 0/{total_items}")
    if total_items <= 0:
        return []

    max_workers = _resolve_generation_eval_openai_max_workers(total_items)
    if max_workers == 1:
        results = []
        for index, item in enumerate(items, start=1):
            results.append(worker(item))
            _emit_progress_update(f"{progress_prefix} {index}/{total_items}")
        return results

    results: list[Any] = [None] * total_items
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            completed += 1
            _emit_progress_update(f"{progress_prefix} {completed}/{total_items}")
    return results


def _evaluate_classification_model_comparison(
    config: TrainConfig,
    tokenizer: Any,
    eval_dataset: Any,
    *,
    holdout_metadata: dict[str, Any] | None,
    candidate_model: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_plan = config.evaluation_plan or {}
    seed = int(evaluation_plan.get("deterministic_seed", 42))
    comparison_max_examples = int(evaluation_plan.get("comparison_max_examples", DEFAULT_COMPARISON_MAX_EXAMPLES))
    sample = _sample_eval_dataset(eval_dataset, max_examples=comparison_max_examples, seed=seed)

    baseline_model = _load_base_model(config)
    baseline_model.eval()
    baseline_predictions = _predict_classification_outputs(baseline_model, tokenizer, sample)
    baseline_eval = _build_classification_eval_result(sample, baseline_predictions)
    _clear_inference_model(baseline_model)

    owned_candidate_model = candidate_model is None
    candidate_model = candidate_model or _load_adapter_inference_model(config)
    candidate_model.eval()
    candidate_predictions = _predict_classification_outputs(candidate_model, tokenizer, sample)
    candidate_eval = _build_classification_eval_result(sample, candidate_predictions)
    evaluation_result = _evaluate_classification_dataset(
        candidate_model,
        tokenizer,
        eval_dataset,
        max_examples=int(evaluation_plan.get("max_examples", DEFAULT_CLASSIFICATION_EVAL_MAX_EXAMPLES)),
        seed=seed,
    )
    if owned_candidate_model:
        _clear_inference_model(candidate_model)

    if holdout_metadata is not None:
        evaluation_result["holdout"] = holdout_metadata

    disagreements = {
        "both_correct": 0,
        "baseline_only_correct": 0,
        "candidate_only_correct": 0,
        "both_wrong": 0,
    }
    cases = []
    for index, prompt in enumerate(sample["prompt"]):
        gold = baseline_predictions["gold_labels"][index]
        baseline_prediction = baseline_predictions["predictions"][index]
        candidate_prediction = candidate_predictions["predictions"][index]
        baseline_correct = baseline_prediction == gold
        candidate_correct = candidate_prediction == gold
        if baseline_correct and candidate_correct:
            disagreements["both_correct"] += 1
        elif baseline_correct:
            disagreements["baseline_only_correct"] += 1
        elif candidate_correct:
            disagreements["candidate_only_correct"] += 1
        else:
            disagreements["both_wrong"] += 1

        cases.append(
            {
                "case_id": index + 1,
                "prompt_preview": _truncate_preview(prompt, 200),
                "gold": gold,
                "baseline_prediction": baseline_prediction,
                "baseline_raw_prediction": baseline_predictions["raw_predictions"][index][:120],
                "candidate_prediction": candidate_prediction,
                "candidate_raw_prediction": candidate_predictions["raw_predictions"][index][:120],
                "baseline_correct": baseline_correct,
                "candidate_correct": candidate_correct,
            }
        )

    comparison = {
        "task_family": "classification",
        "strategy": "sampled_base_vs_tuned_prompt_completion_eval",
        "gold_available": True,
        "sample_policy": _comparison_sample_policy(
            len(eval_dataset),
            max_examples=comparison_max_examples,
            seed=seed,
        ),
        "baseline": {
            "kind": "base_model",
            "model_id": config.base_model,
            "adapter_path": None,
        },
        "candidate": {
            "kind": "adapter",
            "model_id": config.base_model,
            "adapter_path": str(config.final_adapter_dir),
        },
        "summary": {
            "winner": _classification_winner_from_metrics(
                baseline_eval["metrics"],
                candidate_eval["metrics"],
            ),
            "baseline_metrics": baseline_eval["metrics"],
            "candidate_metrics": candidate_eval["metrics"],
            "delta_metrics": _metric_deltas(
                baseline_eval["metrics"],
                candidate_eval["metrics"],
            ),
            "disagreement_counts": disagreements,
        },
        "cases": cases,
    }
    if holdout_metadata is not None:
        comparison["holdout"] = holdout_metadata

    return evaluation_result, comparison


def _evaluate_generation_model_comparison(
    config: TrainConfig,
    tokenizer: Any,
    eval_dataset: Any,
    *,
    holdout_metadata: dict[str, Any] | None,
    candidate_model: Any | None = None,
) -> dict[str, Any]:
    evaluation_plan = config.evaluation_plan or {}
    seed = int(evaluation_plan.get("deterministic_seed", 42))
    comparison_max_examples = int(evaluation_plan.get("comparison_max_examples", DEFAULT_COMPARISON_MAX_EXAMPLES))
    max_new_tokens = int(
        evaluation_plan.get("comparison_max_new_tokens", DEFAULT_GENERATION_EVAL_MAX_NEW_TOKENS)
    )
    judge_model = _maybe_none(os.environ.get("POSTTRAINING_EVAL_JUDGE_MODEL")) or DEFAULT_EVAL_JUDGE_MODEL
    sample = _sample_eval_dataset(eval_dataset, max_examples=comparison_max_examples, seed=seed)
    total_cases = len(sample)

    case_specs = _run_generation_eval_openai_tasks_with_progress(
        [str(text) for text in sample["text"]],
        progress_prefix="synthesizing_generation_cases",
        worker=lambda source_text: _synthesize_generation_case(source_text, judge_model=judge_model),
    )
    model_inputs = [
        _build_grounded_generation_eval_prompt(case_spec["prompt"], str(source_text))
        for source_text, case_spec in zip(sample["text"], case_specs)
    ]

    baseline_model = _load_base_model(config)
    baseline_model.eval()
    baseline_outputs = []
    _emit_progress_update(f"running_baseline_generation_cases 0/{total_cases}")
    for index, model_input in enumerate(model_inputs, start=1):
        baseline_outputs.append(
            _strip_thinking_blocks(
                _predict_generation_response(
                    baseline_model,
                    tokenizer,
                    model_input,
                    max_new_tokens=max_new_tokens,
                )
            )
        )
        _emit_progress_update(f"running_baseline_generation_cases {index}/{total_cases}")
    _clear_inference_model(baseline_model)

    owned_candidate_model = candidate_model is None
    candidate_model = candidate_model or _load_adapter_inference_model(config)
    candidate_model.eval()
    candidate_outputs = []
    _emit_progress_update(f"running_candidate_generation_cases 0/{total_cases}")
    for index, model_input in enumerate(model_inputs, start=1):
        candidate_outputs.append(
            _strip_thinking_blocks(
                _predict_generation_response(
                    candidate_model,
                    tokenizer,
                    model_input,
                    max_new_tokens=max_new_tokens,
                )
            )
        )
        _emit_progress_update(f"running_candidate_generation_cases {index}/{total_cases}")
    if owned_candidate_model:
        _clear_inference_model(candidate_model)

    baseline_wins = 0
    candidate_wins = 0
    ties = 0
    baseline_score_total = 0.0
    candidate_score_total = 0.0
    cases = []

    judgment_tasks = [
        {
            "index": index,
            "source_text": str(source_text),
            "case_spec": case_spec,
            "baseline_output": baseline_output,
            "candidate_output": candidate_output,
        }
        for index, (source_text, case_spec, baseline_output, candidate_output) in enumerate(
            zip(sample["text"], case_specs, baseline_outputs, candidate_outputs)
        )
    ]
    judgments = _run_generation_eval_openai_tasks_with_progress(
        judgment_tasks,
        progress_prefix="judging_generation_cases",
        worker=lambda task: _judge_generation_outputs(
            prompt=task["case_spec"]["prompt"],
            source_text=task["source_text"],
            reference_answer=task["case_spec"]["reference_answer"],
            rubric=task["case_spec"]["rubric"],
            baseline_output=task["baseline_output"],
            candidate_output=task["candidate_output"],
            judge_model=judge_model,
            flip_order=bool(task["index"] % 2),
        ),
    )
    for index, (source_text, case_spec, model_input, baseline_output, candidate_output, judgment) in enumerate(
        zip(sample["text"], case_specs, model_inputs, baseline_outputs, candidate_outputs, judgments)
    ):
        winner = judgment["winner"]
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "candidate":
            candidate_wins += 1
        else:
            ties += 1

        baseline_score_total += float(judgment["baseline_score"])
        candidate_score_total += float(judgment["candidate_score"])
        cases.append(
            {
                "case_id": index + 1,
                "source_text_preview": _truncate_preview(source_text, 220),
                "source_summary": case_spec["source_summary"],
                "prompt": case_spec["prompt"],
                "model_input_preview": _truncate_preview(model_input, 500),
                "reference_answer": case_spec["reference_answer"],
                "rubric": case_spec["rubric"],
                "baseline_output": baseline_output,
                "candidate_output": candidate_output,
                "judgment": judgment,
            }
        )

    total_cases = len(cases)
    comparison = {
        "task_family": "generation",
        "strategy": "sampled_base_vs_tuned_generation_eval",
        "gold_available": False,
        "judge_model": judge_model,
        "sample_policy": _comparison_sample_policy(
            len(eval_dataset),
            max_examples=comparison_max_examples,
            seed=seed,
        ),
        "baseline": {
            "kind": "base_model",
            "model_id": config.base_model,
            "adapter_path": None,
        },
        "candidate": {
            "kind": "adapter",
            "model_id": config.base_model,
            "adapter_path": str(config.final_adapter_dir),
        },
        "summary": {
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "ties": ties,
            "baseline_win_rate": (baseline_wins / total_cases) if total_cases else 0.0,
            "candidate_win_rate": (candidate_wins / total_cases) if total_cases else 0.0,
            "tie_rate": (ties / total_cases) if total_cases else 0.0,
            "baseline_average_score": (baseline_score_total / total_cases) if total_cases else 0.0,
            "candidate_average_score": (candidate_score_total / total_cases) if total_cases else 0.0,
        },
        "cases": cases,
    }
    if holdout_metadata is not None:
        comparison["holdout"] = holdout_metadata
    return comparison


def _evaluate_generation_prompt_completion_model_comparison(
    config: TrainConfig,
    tokenizer: Any,
    eval_dataset: Any,
    *,
    holdout_metadata: dict[str, Any] | None,
    candidate_model: Any | None = None,
) -> dict[str, Any]:
    evaluation_plan = config.evaluation_plan or {}
    seed = int(evaluation_plan.get("deterministic_seed", 42))
    comparison_max_examples = int(evaluation_plan.get("comparison_max_examples", DEFAULT_COMPARISON_MAX_EXAMPLES))
    max_new_tokens = int(
        evaluation_plan.get("comparison_max_new_tokens", DEFAULT_GENERATION_EVAL_MAX_NEW_TOKENS)
    )
    judge_model = _maybe_none(os.environ.get("POSTTRAINING_EVAL_JUDGE_MODEL")) or DEFAULT_EVAL_JUDGE_MODEL
    sample = _sample_eval_dataset(eval_dataset, max_examples=comparison_max_examples, seed=seed)
    total_cases = len(sample)

    baseline_model = _load_base_model(config)
    baseline_model.eval()
    baseline_outputs = []
    _emit_progress_update(f"running_baseline_generation_cases 0/{total_cases}")
    for index, prompt in enumerate(sample["prompt"], start=1):
        baseline_outputs.append(
            _strip_thinking_blocks(
                _predict_generation_response(
                    baseline_model,
                    tokenizer,
                    str(prompt),
                    max_new_tokens=max_new_tokens,
                )
            )
        )
        _emit_progress_update(f"running_baseline_generation_cases {index}/{total_cases}")
    _clear_inference_model(baseline_model)

    owned_candidate_model = candidate_model is None
    candidate_model = candidate_model or _load_adapter_inference_model(config)
    candidate_model.eval()
    candidate_outputs = []
    _emit_progress_update(f"running_candidate_generation_cases 0/{total_cases}")
    for index, prompt in enumerate(sample["prompt"], start=1):
        candidate_outputs.append(
            _strip_thinking_blocks(
                _predict_generation_response(
                    candidate_model,
                    tokenizer,
                    str(prompt),
                    max_new_tokens=max_new_tokens,
                )
            )
        )
        _emit_progress_update(f"running_candidate_generation_cases {index}/{total_cases}")
    if owned_candidate_model:
        _clear_inference_model(candidate_model)

    baseline_wins = 0
    candidate_wins = 0
    ties = 0
    baseline_score_total = 0.0
    candidate_score_total = 0.0
    cases = []
    rubric = [
        "Answers the prompt directly and follows the requested output format and task instructions.",
        "Matches the gold reference closely on key facts, structure, and level of detail without obvious unsupported additions.",
        "Uses clear, coherent wording suitable for the target task style.",
    ]

    judgment_tasks = [
        {
            "index": index,
            "prompt": str(prompt),
            "reference_answer": str(reference_answer),
            "baseline_output": baseline_output,
            "candidate_output": candidate_output,
        }
        for index, (prompt, reference_answer, baseline_output, candidate_output) in enumerate(
            zip(sample["prompt"], sample["completion"], baseline_outputs, candidate_outputs)
        )
    ]
    judgments = _run_generation_eval_openai_tasks_with_progress(
        judgment_tasks,
        progress_prefix="judging_generation_cases",
        worker=lambda task: _judge_generation_outputs(
            prompt=task["prompt"],
            source_text=task["prompt"],
            reference_answer=task["reference_answer"],
            rubric=rubric,
            baseline_output=task["baseline_output"],
            candidate_output=task["candidate_output"],
            judge_model=judge_model,
            flip_order=bool(task["index"] % 2),
        ),
    )
    for prompt, reference_answer, baseline_output, candidate_output, judgment in zip(
        sample["prompt"],
        sample["completion"],
        baseline_outputs,
        candidate_outputs,
        judgments,
    ):
        winner = judgment["winner"]
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "candidate":
            candidate_wins += 1
        else:
            ties += 1

        baseline_score_total += float(judgment["baseline_score"])
        candidate_score_total += float(judgment["candidate_score"])
        cases.append(
            {
                "case_id": index + 1,
                "prompt": str(prompt),
                "prompt_preview": _truncate_preview(prompt, 500),
                "reference_answer": str(reference_answer),
                "rubric": rubric,
                "baseline_output": baseline_output,
                "candidate_output": candidate_output,
                "judgment": judgment,
            }
        )

    total_cases = len(cases)
    comparison = {
        "task_family": "generation",
        "strategy": "sampled_base_vs_tuned_prompt_completion_generation_eval",
        "gold_available": True,
        "judge_model": judge_model,
        "sample_policy": _comparison_sample_policy(
            len(eval_dataset),
            max_examples=comparison_max_examples,
            seed=seed,
        ),
        "baseline": {
            "kind": "base_model",
            "model_id": config.base_model,
            "adapter_path": None,
        },
        "candidate": {
            "kind": "adapter",
            "model_id": config.base_model,
            "adapter_path": str(config.final_adapter_dir),
        },
        "summary": {
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "ties": ties,
            "baseline_win_rate": (baseline_wins / total_cases) if total_cases else 0.0,
            "candidate_win_rate": (candidate_wins / total_cases) if total_cases else 0.0,
            "tie_rate": (ties / total_cases) if total_cases else 0.0,
            "baseline_average_score": (baseline_score_total / total_cases) if total_cases else 0.0,
            "candidate_average_score": (candidate_score_total / total_cases) if total_cases else 0.0,
        },
        "cases": cases,
    }
    if holdout_metadata is not None:
        comparison["holdout"] = holdout_metadata

    return comparison

def _normalize_sft_dataset(dataset: Any, tokenizer: Any, apply_chat_template: Any) -> tuple[Any, str]:
    column_names = set(dataset.column_names)

    if "text" in column_names:
        normalized = dataset.map(
            lambda example: {"text": _normalize_text_value(example["text"], "text")},
            remove_columns=list(dataset.column_names),
            desc="Normalizing SFT text dataset",
        )
        normalized = _drop_blank_normalized_sft_rows(
            normalized,
            ("text",),
            dataset_label="SFT text dataset",
        )
        return normalized, "language_modeling"

    if "messages" in column_names:
        def normalize_messages(example: dict[str, Any]) -> dict[str, str]:
            messages = example["messages"]
            if _is_conversational_value(messages):
                rendered = _render_chat_template(
                    apply_chat_template,
                    {"messages": messages},
                    tokenizer,
                )
                return {"text": rendered["text"]}
            return {"text": _normalize_text_value(messages, "messages")}

        normalized = dataset.map(
            normalize_messages,
            remove_columns=list(dataset.column_names),
            desc="Applying chat template to SFT messages dataset",
        )
        normalized = _drop_blank_normalized_sft_rows(
            normalized,
            ("text",),
            dataset_label="SFT messages dataset",
        )
        return normalized, "language_modeling"

    if {"prompt", "completion"}.issubset(column_names):
        def normalize_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
            prompt = example["prompt"]
            completion = example["completion"]
            if _is_conversational_value(prompt) or _is_conversational_value(completion):
                rendered = _render_chat_template(
                    apply_chat_template,
                    {"prompt": prompt, "completion": completion},
                    tokenizer,
                )
                return {"prompt": rendered["prompt"], "completion": rendered["completion"]}
            return {
                "prompt": _normalize_text_value(prompt, "prompt"),
                "completion": _normalize_text_value(completion, "completion"),
            }

        normalized = dataset.map(
            normalize_prompt_completion,
            remove_columns=list(dataset.column_names),
            desc="Normalizing SFT prompt-completion dataset",
        )
        normalized = _drop_blank_normalized_sft_rows(
            normalized,
            ("prompt", "completion"),
            dataset_label="SFT prompt-completion dataset",
        )
        return normalized, "prompt_completion"

    raise ValueError(
        "SFT expects one of these dataset shapes: {text}, {messages}, or {prompt, completion}."
    )


def _normalize_paired_preference_dataset(
    dataset: Any,
    tokenizer: Any,
    apply_chat_template: Any,
    trainer_name: str,
) -> Any:
    column_names = set(dataset.column_names)
    if not {"prompt", "chosen", "rejected"}.issubset(column_names):
        raise ValueError(
            f"{trainer_name} expects a paired preference dataset with explicit prompt/chosen/rejected columns."
        )

    def normalize_preference_row(example: dict[str, Any]) -> dict[str, str]:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        if prompt is None:
            raise ValueError(
                f"{trainer_name} requires an explicit prompt column. Implicit preference datasets are not supported in this example."
            )
        if (
            _is_conversational_value(prompt)
            or _is_conversational_value(chosen)
            or _is_conversational_value(rejected)
        ):
            rendered = _render_chat_template(
                apply_chat_template,
                {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                tokenizer,
            )
            return {
                "prompt": rendered["prompt"],
                "chosen": rendered["chosen"],
                "rejected": rendered["rejected"],
            }
        return {
            "prompt": _normalize_text_value(prompt, "prompt"),
            "chosen": _normalize_text_value(chosen, "chosen"),
            "rejected": _normalize_text_value(rejected, "rejected"),
        }

    return dataset.map(
        normalize_preference_row,
        remove_columns=list(dataset.column_names),
        desc=f"Normalizing {trainer_name} preference dataset",
    )


def _normalize_unpaired_preference_dataset(
    dataset: Any,
    tokenizer: Any,
    apply_chat_template: Any,
    trainer_name: str,
) -> Any:
    column_names = set(dataset.column_names)

    if {"prompt", "completion", "label"}.issubset(column_names):
        def normalize_unpaired_row(example: dict[str, Any]) -> dict[str, Any]:
            prompt = example["prompt"]
            completion = example["completion"]
            label = example["label"]
            if prompt is None:
                raise ValueError(
                    f"{trainer_name} requires an explicit prompt column. Implicit unpaired preference datasets are not supported in this example."
                )
            if _is_conversational_value(prompt) or _is_conversational_value(completion):
                rendered = _render_chat_template(
                    apply_chat_template,
                    {"prompt": prompt, "completion": completion},
                    tokenizer,
                )
                return {
                    "prompt": rendered["prompt"],
                    "completion": rendered["completion"],
                    "label": _normalize_label_value(label),
                }
            return {
                "prompt": _normalize_text_value(prompt, "prompt"),
                "completion": _normalize_text_value(completion, "completion"),
                "label": _normalize_label_value(label),
            }

        return dataset.map(
            normalize_unpaired_row,
            remove_columns=list(dataset.column_names),
            desc=f"Normalizing {trainer_name} unpaired preference dataset",
        )

    if {"prompt", "chosen", "rejected"}.issubset(column_names):
        def paired_to_unpaired(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            prompts: list[str] = []
            completions: list[str] = []
            labels: list[bool] = []

            for prompt, chosen, rejected in zip(
                batch["prompt"],
                batch["chosen"],
                batch["rejected"],
                strict=True,
            ):
                if prompt is None:
                    raise ValueError(
                        f"{trainer_name} requires an explicit prompt column. Implicit paired preference datasets are not supported in this example."
                    )
                if (
                    _is_conversational_value(prompt)
                    or _is_conversational_value(chosen)
                    or _is_conversational_value(rejected)
                ):
                    rendered = _render_chat_template(
                        apply_chat_template,
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                        tokenizer,
                    )
                    prompt_text = rendered["prompt"]
                    chosen_text = rendered["chosen"]
                    rejected_text = rendered["rejected"]
                else:
                    prompt_text = _normalize_text_value(prompt, "prompt")
                    chosen_text = _normalize_text_value(chosen, "chosen")
                    rejected_text = _normalize_text_value(rejected, "rejected")

                prompts.extend([prompt_text, prompt_text])
                completions.extend([chosen_text, rejected_text])
                labels.extend([True, False])

            return {"prompt": prompts, "completion": completions, "label": labels}

        return dataset.map(
            paired_to_unpaired,
            batched=True,
            remove_columns=list(dataset.column_names),
            desc=f"Converting paired preference data to {trainer_name} unpaired format",
        )

    raise ValueError(
        f"{trainer_name} expects either {{prompt, completion, label}} or {{prompt, chosen, rejected}}."
    )


def _load_tokenizer(config: TrainConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_quantization_config(config: TrainConfig, torch: Any, BitsAndBytesConfig: Any) -> Any | None:
    if not config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _render_chat_template(apply_chat_template: Any, example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    # Qwen 3.5 chat templates emit <think> blocks by default unless explicitly disabled.
    return apply_chat_template(
        example,
        tokenizer=tokenizer,
        **DEFAULT_CHAT_TEMPLATE_KWARGS,
    )


def _load_base_model(config: TrainConfig, for_merge: bool = False) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None if for_merge else _build_quantization_config(config, torch, BitsAndBytesConfig)

    model_kwargs: dict[str, Any] = {
        "revision": config.base_model_revision,
        "cache_dir": str(MODEL_CACHE_DIR),
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}
    elif for_merge:
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False
    return model


def _prepare_model_for_training(config: TrainConfig, model: Any) -> Any:
    if not config.load_in_4bit:
        return model
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(model)


def _load_seeded_peft_model(config: TrainConfig) -> Any:
    from peft import PeftModel

    seed_artifact = _resolve_seed_artifact_path(config)
    if not seed_artifact.exists():
        raise FileNotFoundError(
            f"seed_artifact does not exist: {seed_artifact}. Non-SFT runs must start from a saved adapter."
        )

    model = _prepare_model_for_training(config, _load_base_model(config))
    model = PeftModel.from_pretrained(model, str(seed_artifact), is_trainable=True)
    model.config.use_cache = False
    return model


def _build_lora_config(config: TrainConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )


def _resolve_resume_checkpoint(output_dir: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    if not output_dir.exists():
        return None
    return get_last_checkpoint(str(output_dir))


def _emit_structured_training_metric(state: Any, metrics: dict[str, Any] | None) -> None:
    numeric_metrics: dict[str, float] = {}
    for key, raw_value in (metrics or {}).items():
        if isinstance(raw_value, bool):
            continue
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not isinstance(key, str):
            continue
        numeric_metrics[key] = numeric_value

    if not numeric_metrics:
        return

    epoch_value = getattr(state, "epoch", None)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "step": int(getattr(state, "global_step", 0)),
        "epoch": float(epoch_value) if epoch_value is not None else None,
        "metrics": numeric_metrics,
    }
    print(f"{STRUCTURED_TRAINING_METRIC_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


def _build_structured_metric_callback() -> Any:
    from transformers import TrainerCallback

    class StructuredMetricPrinterCallback(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            _emit_structured_training_metric(state, logs)

    return StructuredMetricPrinterCallback()


def _build_common_trainer_kwargs(
    config: TrainConfig,
    output_dir: Path,
    has_eval: bool,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.resolved_learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": config.eval_steps if has_eval else None,
        "report_to": config.report_to,
        "run_name": config.output_name,
        "seed": 42,
    }


def _build_trainer_bundle(
    config: TrainConfig,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    sft_dataset_style: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    from trl.experimental.bco import BCOConfig, BCOTrainer
    from trl.experimental.cpo import CPOConfig, CPOTrainer
    from trl.experimental.kto import KTOConfig, KTOTrainer
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    has_eval = eval_dataset is not None
    common = _build_common_trainer_kwargs(config, config.checkpoints_dir, has_eval)
    callbacks = [_build_structured_metric_callback()]

    if config.trainer_type == "sft":
        model = _prepare_model_for_training(config, _load_base_model(config))
        training_args = SFTConfig(
            **common,
            max_length=config.max_length,
            completion_only_loss=(sft_dataset_style == "prompt_completion"),
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=_build_lora_config(config),
            callbacks=callbacks,
        )
        return trainer, {"trainer": "SFTTrainer"}

    model = _load_seeded_peft_model(config)

    if config.trainer_type == "dpo":
        training_args = DPOConfig(
            **common,
            max_length=config.max_length,
            beta=config.beta,
            precompute_ref_log_probs=True,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        return trainer, {"trainer": "DPOTrainer"}

    if config.trainer_type == "kto":
        training_args = KTOConfig(
            **common,
            max_length=config.max_length,
            beta=config.beta,
            precompute_ref_log_probs=True,
        )
        trainer = KTOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        return trainer, {"trainer": "KTOTrainer"}

    if config.trainer_type == "orpo":
        training_args = ORPOConfig(
            **common,
            max_length=config.max_length,
            beta=config.beta,
        )
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        return trainer, {"trainer": "ORPOTrainer"}

    if config.trainer_type == "cpo":
        training_args = CPOConfig(
            **common,
            max_length=config.max_length,
            beta=config.beta,
            loss_type="sigmoid",
        )
        trainer = CPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        return trainer, {"trainer": "CPOTrainer"}

    training_args = BCOConfig(
        **common,
        max_length=config.max_length,
        beta=config.beta,
        precompute_ref_log_probs=True,
    )
    trainer = BCOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    return trainer, {"trainer": "BCOTrainer"}


def _normalize_for_trainer(config: TrainConfig, tokenizer: Any, dataset: Any) -> tuple[Any, str | None]:
    from trl import apply_chat_template

    if config.trainer_type == "sft":
        return _normalize_sft_dataset(dataset, tokenizer, apply_chat_template)
    if config.trainer_type in {"dpo", "orpo", "cpo"}:
        return _normalize_paired_preference_dataset(
            dataset,
            tokenizer,
            apply_chat_template,
            trainer_name=config.trainer_type.upper(),
        ), None
    return _normalize_unpaired_preference_dataset(
        dataset,
        tokenizer,
        apply_chat_template,
        trainer_name=config.trainer_type.upper(),
    ), None


def _save_final_adapter(trainer: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _merge_adapter_into_base(config: TrainConfig, tokenizer: Any) -> None:
    from peft import PeftModel

    _emit_progress_update("preparing_merged_deployment_artifact")
    config.merged_dir.mkdir(parents=True, exist_ok=True)
    base_model = _load_base_model(config, for_merge=True)
    merged = PeftModel.from_pretrained(base_model, str(config.final_adapter_dir))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(config.merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(config.merged_dir))
    _emit_progress_update("prepared_merged_deployment_artifact")


def _commit_all_volumes() -> None:
    MODEL_CACHE_VOLUME.commit()
    DATASET_CACHE_VOLUME.commit()
    CHECKPOINTS_VOLUME.commit()


def _load_normalized_training_datasets(
    config: TrainConfig,
    tokenizer: Any,
) -> tuple[Any, Any | None, str | None, dict[str, Any] | None, dict[str, Any] | None]:
    evaluation_plan = config.evaluation_plan or {}
    holdout_metadata = None

    if (
        config.trainer_type == "sft"
        and config.dataset_source_type == "prepared_manifest"
        and evaluation_plan.get("strategy") == "merged_sft_holdout"
    ):
        prepared_train_dataset, prepared_eval_dataset, preprocessing_diagnostics, holdout_metadata = (
            _load_sft_prepared_manifest_datasets(config)
        )
        normalized_train_dataset, sft_dataset_style = _normalize_for_trainer(
            config,
            tokenizer,
            prepared_train_dataset,
        )
        normalized_eval_dataset = None
        if prepared_eval_dataset is not None:
            normalized_eval_dataset, _ = _normalize_for_trainer(
                config,
                tokenizer,
                prepared_eval_dataset,
            )
        return (
            normalized_train_dataset,
            normalized_eval_dataset,
            sft_dataset_style,
            preprocessing_diagnostics,
            holdout_metadata,
        )

    train_dataset, eval_dataset, preprocessing_diagnostics = _load_datasets(config)
    normalized_train_dataset, sft_dataset_style = _normalize_for_trainer(
        config,
        tokenizer,
        train_dataset,
    )
    normalized_eval_dataset = None
    if eval_dataset is not None:
        normalized_eval_dataset, _ = _normalize_for_trainer(config, tokenizer, eval_dataset)

    if (
        config.trainer_type == "sft"
        and (config.task_spec or {}).get("task_family") == "classification"
        and sft_dataset_style == "prompt_completion"
        and normalized_eval_dataset is None
    ):
        normalized_train_dataset, normalized_eval_dataset, holdout_metadata = _create_stratified_holdout(
            normalized_train_dataset,
            fraction=float(evaluation_plan.get("holdout_fraction", 0.1)),
            seed=int(evaluation_plan.get("deterministic_seed", 42)),
        )

    return (
        normalized_train_dataset,
        normalized_eval_dataset,
        sft_dataset_style,
        preprocessing_diagnostics,
        holdout_metadata,
    )


def _release_trainer_model(trainer: Any) -> None:
    model = getattr(trainer, "model", None)
    if model is None:
        return
    try:
        trainer.model = None
    except Exception:
        pass
    _clear_inference_model(model)


def _run_offline_evaluation(
    config: TrainConfig,
    tokenizer: Any,
    normalized_eval_dataset: Any | None,
    *,
    sft_dataset_style: str | None,
    preprocessing_diagnostics: dict[str, Any] | None,
    holdout_metadata: dict[str, Any] | None,
    candidate_model: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if config.trainer_type != "sft":
        raise ValueError("Comparison evaluation currently supports SFT jobs only.")
    if normalized_eval_dataset is None or len(normalized_eval_dataset) <= 0:
        raise ValueError("Evaluation requested but no eval dataset is available after preprocessing.")

    task_family = (config.task_spec or {}).get("task_family")
    if task_family == "classification" and sft_dataset_style == "prompt_completion":
        return _evaluate_classification_model_comparison(
            config,
            tokenizer,
            normalized_eval_dataset,
            holdout_metadata=holdout_metadata,
            candidate_model=candidate_model,
        )
    if task_family == "generation" and sft_dataset_style == "prompt_completion":
        return None, _evaluate_generation_prompt_completion_model_comparison(
            config,
            tokenizer,
            normalized_eval_dataset,
            holdout_metadata=holdout_metadata,
            candidate_model=candidate_model,
        )
    if task_family == "generation" and sft_dataset_style == "language_modeling":
        return None, _evaluate_generation_model_comparison(
            config,
            tokenizer,
            normalized_eval_dataset,
            holdout_metadata=holdout_metadata,
            candidate_model=candidate_model,
        )

    raise ValueError(
        f"Comparison evaluation is not supported for task_family={task_family!r} "
        f"with dataset_style={sft_dataset_style!r}."
    )


def _train_then_evaluate_impl(config: TrainConfig) -> dict[str, Any]:
    MODEL_CACHE_VOLUME.reload()
    DATASET_CACHE_VOLUME.reload()
    CHECKPOINTS_VOLUME.reload()

    _write_run_config(config)

    tokenizer = _load_tokenizer(config)
    tokenizer.padding_side = "right" if config.trainer_type == "sft" else "left"
    (
        normalized_train_dataset,
        normalized_eval_dataset,
        sft_dataset_style,
        preprocessing_diagnostics,
        holdout_metadata,
    ) = _load_normalized_training_datasets(config, tokenizer)

    trainer, trainer_meta = _build_trainer_bundle(
        config,
        tokenizer,
        normalized_train_dataset,
        normalized_eval_dataset,
        sft_dataset_style=sft_dataset_style,
    )

    resume_from_checkpoint = _resolve_resume_checkpoint(config.checkpoints_dir)
    training_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    _save_final_adapter(trainer, tokenizer, config.final_adapter_dir)
    training_result = _build_training_result(
        config,
        trainer_meta=trainer_meta,
        normalized_train_dataset=normalized_train_dataset,
        normalized_eval_dataset=normalized_eval_dataset,
        training_output=training_output,
        preprocessing_diagnostics=preprocessing_diagnostics,
        holdout_metadata=holdout_metadata,
        resumed_from_checkpoint=resume_from_checkpoint,
        include_merged_dir=_should_build_merged_artifact(config),
    )

    CHECKPOINTS_VOLUME.commit()
    _emit_structured_lifecycle_event("training_complete", training_result=training_result)

    candidate_model = getattr(trainer, "model", None)
    if candidate_model is None:
        raise RuntimeError("Trainer did not expose a model for same-container evaluation.")
    candidate_model.eval()
    evaluation_result, comparison_evaluation = _run_offline_evaluation(
        config,
        tokenizer,
        normalized_eval_dataset,
        sft_dataset_style=sft_dataset_style,
        preprocessing_diagnostics=preprocessing_diagnostics,
        holdout_metadata=holdout_metadata,
        candidate_model=candidate_model,
    )

    _release_trainer_model(trainer)
    if _should_build_merged_artifact(config):
        _merge_adapter_into_base(config, tokenizer)

    _commit_all_volumes()

    return {
        "training_result": training_result,
        "evaluation": evaluation_result,
        "comparison_evaluation": comparison_evaluation,
    }


def _training_runner_cls(modal_gpu_type: str):
    return app.cls(
        image=train_image,
        gpu=modal_gpu_type,
        timeout=12 * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=3),
        single_use_containers=True,
        volumes={
            str(MODEL_CACHE_DIR): MODEL_CACHE_VOLUME,
            str(DATASET_CACHE_DIR): DATASET_CACHE_VOLUME,
            str(CHECKPOINTS_DIR): CHECKPOINTS_VOLUME,
        },
        secrets=[HF_SECRET],
    )


class _TrainingRunnerMethods:
    @modal.method()
    def train_then_evaluate(self, config: TrainConfig) -> dict[str, Any]:
        return _train_then_evaluate_impl(config)


@_training_runner_cls("A10G")
class A10GTrainingRunner(_TrainingRunnerMethods):
    pass


@_training_runner_cls("L40S")
class L40STrainingRunner(_TrainingRunnerMethods):
    pass


@_training_runner_cls("H100")
class H100TrainingRunner(_TrainingRunnerMethods):
    pass


TRAINING_RUNNER_CLS_BY_MODAL_GPU = {
    "A10G": A10GTrainingRunner,
    "L40S": L40STrainingRunner,
    "H100": H100TrainingRunner,
}


def _resolve_training_runner(config: TrainConfig) -> tuple[Any, str]:
    modal_gpu_type = config.modal_gpu_type
    runner_cls = TRAINING_RUNNER_CLS_BY_MODAL_GPU.get(modal_gpu_type)
    if runner_cls is None:
        raise ValueError(f"Unsupported Modal runtime GPU type {modal_gpu_type!r}.")
    return runner_cls, modal_gpu_type


def _with_runtime_runner_options(runner_cls: Any, config: TrainConfig) -> Any:
    if hasattr(runner_cls, "with_options"):
        return runner_cls.with_options(secrets=_build_runtime_secrets(config))
    return runner_cls


@app.local_entrypoint()
def main(config: str = "backend/modal_trl_posttrain.example.yaml") -> None:
    raw_config = _load_yaml_mapping(config)
    train_config = _config_from_mapping(raw_config)
    run_mode = (_maybe_none(os.environ.get("POSTTRAINING_RUN_MODE")) or "train_then_evaluate").lower()
    if run_mode != "train_then_evaluate":
        raise ValueError("POSTTRAINING_RUN_MODE must be 'train_then_evaluate'.")
    runner_cls, modal_gpu_type = _resolve_training_runner(train_config)
    print(
        f"PT_GPU_SELECTION public_gpu_type={train_config.gpu_type} "
        f"modal_gpu_type={modal_gpu_type} run_mode={run_mode}",
        flush=True,
    )
    runner_cls = _with_runtime_runner_options(runner_cls, train_config)
    result = runner_cls().train_then_evaluate.spawn(train_config).get()
    print(json.dumps(result, indent=2, sort_keys=True))

    training_result_path = _maybe_none(os.environ.get("TRAIN_RESULT_PATH"))
    evaluation_result_path = _maybe_none(os.environ.get("EVALUATION_RESULT_PATH"))
    comparison_evaluation_path = _maybe_none(os.environ.get("COMPARISON_EVALUATION_PATH"))
    if training_result_path and result.get("training_result") is not None:
        output_path = Path(training_result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result["training_result"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if evaluation_result_path and result.get("evaluation") is not None:
        output_path = Path(evaluation_result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result["evaluation"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if comparison_evaluation_path and result.get("comparison_evaluation") is not None:
        output_path = Path(comparison_evaluation_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result["comparison_evaluation"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
