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
vllm serve Qwen/Qwen3-8B-Base --enable-lora --lora-modules run=/checkpoints/experiments/qwen3-8b-dpo/final_adapter
```

Notes:
- This example is intentionally PEFT/LoRA-first.
- `gpu_type` is part of the public config. Because Modal GPU resources are static on
  plain functions, the runtime GPU override is implemented with `Cls.with_options(...)`.
- For non-SFT trainers, this example expects a prior SFT adapter path in
  `seed_artifact`, following TRL's recommended workflow.
- Create a Modal secret named `huggingface-secret` before running this example.
- If `enable_wandb=True`, also create a Modal secret named `wandb-secret`.
- Pass a YAML file to `--config`.
- In YAML, `target_modules` can be either a proper YAML list or a comma-separated string.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import modal


TrainerType = Literal["sft", "dpo", "kto", "orpo", "cpo", "bco"]

APP_NAME = "trl-posttraining"
DEFAULT_GPU_TYPE = "A10"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MODEL_CACHE_DIR = Path("/model_cache")
DATASET_CACHE_DIR = Path("/dataset_cache")
CHECKPOINTS_DIR = Path("/checkpoints")
EXPERIMENTS_DIR = CHECKPOINTS_DIR / "experiments"

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
) -> str:
    if label_mapping:
        for candidate in _label_mapping_candidates(raw_label):
            if candidate in label_mapping:
                return str(label_mapping[candidate])
        raise ValueError(
            f"Could not map `{field_name}` value `{raw_label}` through label_mapping. "
            "Ensure label_mapping contains string keys for the raw dataset labels."
        )
    return _normalize_text_value(raw_label, field_name)


def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


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


def _transform_prepared_dataset_entry(config: TrainConfig, dataset: Any, entry: dict[str, Any]) -> Any:
    preset = _require_manifest_string(entry, "transform_preset", entry.get("dataset", "dataset"))
    dataset_id = _require_manifest_string(entry, "dataset", entry.get("dataset", "dataset"))
    field_mapping = _require_manifest_mapping(entry, "field_mapping")
    label_mapping = _require_manifest_mapping(entry, "label_mapping")
    prompt_template = entry.get("prompt_template")
    if prompt_template is not None and not isinstance(prompt_template, str):
        raise ValueError(f"Prepared dataset `{dataset_id}` prompt_template must be a string when provided.")

    remove_columns = list(dataset.column_names)

    if preset == "sft_text":
        text_column = _require_source_column(field_mapping, "text", dataset_id)
        _assert_columns_exist(dataset, [text_column], dataset_id)
        return dataset.map(
            lambda example: {"text": example[text_column]},
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with sft_text",
        )

    if preset == "sft_messages":
        messages_column = _require_source_column(field_mapping, "messages", dataset_id)
        _assert_columns_exist(dataset, [messages_column], dataset_id)
        return dataset.map(
            lambda example: {"messages": example[messages_column]},
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with sft_messages",
        )

    if preset == "prompt_completion_passthrough":
        prompt_column = _require_source_column(field_mapping, "prompt", dataset_id)
        completion_column = _require_source_column(field_mapping, "completion", dataset_id)
        _assert_columns_exist(dataset, [prompt_column, completion_column], dataset_id)
        return dataset.map(
            lambda example: {
                "prompt": example[prompt_column],
                "completion": example[completion_column],
            },
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with prompt_completion_passthrough",
        )

    if preset == "qa_to_prompt_completion":
        question_column = _require_source_column(field_mapping, "question", dataset_id)
        answer_column = _require_source_column(field_mapping, "answer", dataset_id)
        context_column = field_mapping.get("context")
        columns = [question_column, answer_column]
        if isinstance(context_column, str) and context_column.strip():
            columns.append(context_column.strip())
        _assert_columns_exist(dataset, columns, dataset_id)
        template = prompt_template or "Question:\n{question}\n\nAnswer:"
        if isinstance(context_column, str) and context_column.strip():
            template = prompt_template or "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            context_column = context_column.strip()

        def qa_to_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
            values = {
                "question": example[question_column],
                "answer": example[answer_column],
                "context": example[context_column] if isinstance(context_column, str) else None,
            }
            return {
                "prompt": _render_prompt_template(template, values),
                "completion": example[answer_column],
            }

        return dataset.map(
            qa_to_prompt_completion,
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with qa_to_prompt_completion",
        )

    if preset == "classification_to_prompt_completion":
        input_column = _require_source_column(field_mapping, "input", dataset_id)
        label_column = _require_source_column(field_mapping, "label", dataset_id)
        _assert_columns_exist(dataset, [input_column, label_column], dataset_id)
        label_space = _build_label_space_text(label_mapping)
        template = (
            prompt_template
            or "Classify the following example. Return only the label.\n\n"
            "Available labels: {label_space}\n\nInput:\n{input}\n\nLabel:"
        )

        def classification_to_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
            values = {
                "input": example[input_column],
                "label": example[label_column],
                "label_space": label_space,
            }
            return {
                "prompt": _render_prompt_template(template, values),
                "completion": _resolve_label_mapping_value(example[label_column], label_mapping, label_column),
            }

        return dataset.map(
            classification_to_prompt_completion,
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with classification_to_prompt_completion",
        )

    if preset in {"paired_preference_passthrough", "paired_preference_chat"}:
        prompt_column = _require_source_column(field_mapping, "prompt", dataset_id)
        chosen_column = _require_source_column(field_mapping, "chosen", dataset_id)
        rejected_column = _require_source_column(field_mapping, "rejected", dataset_id)
        _assert_columns_exist(dataset, [prompt_column, chosen_column, rejected_column], dataset_id)

        if config.trainer_type in {"kto", "bco"}:
            def paired_to_unpaired(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
                prompts: list[Any] = []
                completions: list[Any] = []
                labels: list[bool] = []
                for prompt, chosen, rejected in zip(
                    batch[prompt_column],
                    batch[chosen_column],
                    batch[rejected_column],
                    strict=True,
                ):
                    prompts.extend([prompt, prompt])
                    completions.extend([chosen, rejected])
                    labels.extend([True, False])
                return {"prompt": prompts, "completion": completions, "label": labels}

            return dataset.map(
                paired_to_unpaired,
                batched=True,
                remove_columns=remove_columns,
                desc=f"Preparing {dataset_id} paired preference data for {config.trainer_type.upper()}",
            )

        return dataset.map(
            lambda example: {
                "prompt": example[prompt_column],
                "chosen": example[chosen_column],
                "rejected": example[rejected_column],
            },
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with {preset}",
        )

    if preset == "unpaired_preference_passthrough":
        prompt_column = _require_source_column(field_mapping, "prompt", dataset_id)
        completion_column = _require_source_column(field_mapping, "completion", dataset_id)
        label_column = _require_source_column(field_mapping, "label", dataset_id)
        _assert_columns_exist(dataset, [prompt_column, completion_column, label_column], dataset_id)

        def unpaired_preference(example: dict[str, Any]) -> dict[str, Any]:
            raw_label = example[label_column]
            if label_mapping:
                mapped_label = _resolve_label_mapping_value(raw_label, label_mapping, label_column)
            else:
                mapped_label = raw_label
            return {
                "prompt": example[prompt_column],
                "completion": example[completion_column],
                "label": _normalize_label_value(mapped_label),
            }

        return dataset.map(
            unpaired_preference,
            remove_columns=remove_columns,
            desc=f"Preparing {dataset_id} with unpaired_preference_passthrough",
        )

    raise ValueError(
        f"Unsupported transform_preset `{preset}` for prepared dataset `{dataset_id}`."
    )


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


def _load_prepared_manifest_datasets(config: TrainConfig) -> tuple[Any, Any]:
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
        prepared_train = _transform_prepared_dataset_entry(config, train_dataset, raw_entry)
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
            prepared_eval = _transform_prepared_dataset_entry(config, eval_dataset, raw_entry)
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
    return mixed_train, mixed_eval


def _load_datasets(config: TrainConfig) -> tuple[Any, Any]:
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

    return train_dataset, eval_dataset


def _normalize_sft_dataset(dataset: Any, tokenizer: Any, apply_chat_template: Any) -> tuple[Any, str]:
    column_names = set(dataset.column_names)

    if "text" in column_names:
        normalized = dataset.map(
            lambda example: {"text": _normalize_text_value(example["text"], "text")},
            remove_columns=list(dataset.column_names),
            desc="Normalizing SFT text dataset",
        )
        return normalized, "language_modeling"

    if "messages" in column_names:
        def normalize_messages(example: dict[str, Any]) -> dict[str, str]:
            messages = example["messages"]
            if _is_conversational_value(messages):
                rendered = apply_chat_template({"messages": messages}, tokenizer=tokenizer)
                return {"text": rendered["text"]}
            return {"text": _normalize_text_value(messages, "messages")}

        normalized = dataset.map(
            normalize_messages,
            remove_columns=list(dataset.column_names),
            desc="Applying chat template to SFT messages dataset",
        )
        return normalized, "language_modeling"

    if {"prompt", "completion"}.issubset(column_names):
        def normalize_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
            prompt = example["prompt"]
            completion = example["completion"]
            if _is_conversational_value(prompt) or _is_conversational_value(completion):
                rendered = apply_chat_template(
                    {"prompt": prompt, "completion": completion},
                    tokenizer=tokenizer,
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
            rendered = apply_chat_template(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                tokenizer=tokenizer,
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
                rendered = apply_chat_template(
                    {"prompt": prompt, "completion": completion},
                    tokenizer=tokenizer,
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
                    rendered = apply_chat_template(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                        tokenizer=tokenizer,
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

    config.merged_dir.mkdir(parents=True, exist_ok=True)
    base_model = _load_base_model(config, for_merge=True)
    merged = PeftModel.from_pretrained(base_model, str(config.final_adapter_dir))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(config.merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(config.merged_dir))


def _commit_all_volumes() -> None:
    MODEL_CACHE_VOLUME.commit()
    DATASET_CACHE_VOLUME.commit()
    CHECKPOINTS_VOLUME.commit()


def _train_impl(config: TrainConfig) -> dict[str, Any]:
    MODEL_CACHE_VOLUME.reload()
    DATASET_CACHE_VOLUME.reload()
    CHECKPOINTS_VOLUME.reload()

    _write_run_config(config)

    tokenizer = _load_tokenizer(config)
    tokenizer.padding_side = "right" if config.trainer_type == "sft" else "left"

    train_dataset, eval_dataset = _load_datasets(config)
    normalized_train_dataset, sft_dataset_style = _normalize_for_trainer(config, tokenizer, train_dataset)
    normalized_eval_dataset = None
    if eval_dataset is not None:
        normalized_eval_dataset, _ = _normalize_for_trainer(config, tokenizer, eval_dataset)

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

    if config.merge_after_train:
        _merge_adapter_into_base(config, tokenizer)

    _commit_all_volumes()

    result = {
        "trainer_type": config.trainer_type,
        "trainer_class": trainer_meta["trainer"],
        "base_model": config.base_model,
        "dataset_name": config.dataset_name,
        "dataset_source_type": config.dataset_source_type,
        "selected_datasets": (
            [
                entry.get("dataset")
                for entry in (config.prepared_dataset_manifest or {}).get("selected_datasets", [])
                if isinstance(entry, dict) and entry.get("dataset")
            ]
            if config.dataset_source_type == "prepared_manifest"
            else [config.dataset_name]
        ),
        "output_name": config.output_name,
        "gpu_type": config.gpu_type,
        "learning_rate": config.resolved_learning_rate,
        "train_examples": len(normalized_train_dataset),
        "eval_examples": len(normalized_eval_dataset) if normalized_eval_dataset is not None else 0,
        "checkpoint_dir": str(config.checkpoints_dir),
        "final_adapter_dir": str(config.final_adapter_dir),
        "merged_dir": str(config.merged_dir) if config.merge_after_train else None,
        "resumed_from_checkpoint": resume_from_checkpoint,
        "global_step": int(getattr(training_output, "global_step", 0)),
        "training_loss": float(getattr(training_output, "training_loss", 0.0)),
        "notes": [
            "Use final_adapter_dir for vLLM LoRA serving with --enable-lora.",
            "Use merged_dir for direct vLLM base-model serving when merge_after_train=True.",
        ],
    }
    return result


@app.cls(
    image=train_image,
    gpu=DEFAULT_GPU_TYPE,
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
class TrainingRunner:
    @modal.method()
    def train(self, config: TrainConfig) -> dict[str, Any]:
        return _train_impl(config)


@app.local_entrypoint()
def main(config: str = "backend/modal_trl_posttrain.example.yaml") -> None:
    raw_config = _load_yaml_mapping(config)
    train_config = _config_from_mapping(raw_config)
    runner_cls = TrainingRunner.with_options(
        gpu=train_config.gpu_type,
        secrets=_build_runtime_secrets(train_config),
    )
    result = runner_cls().train.spawn(train_config).get()
    print(json.dumps(result, indent=2, sort_keys=True))
    result_path = os.environ.get("TRAIN_RESULT_PATH")
    if result_path:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
