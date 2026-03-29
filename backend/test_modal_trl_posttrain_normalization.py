import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _DummyImageChain:
    def uv_pip_install(self, *args, **kwargs):
        return self

    def env(self, *args, **kwargs):
        return self


class _DummyImage:
    @staticmethod
    def debian_slim(*args, **kwargs):
        return _DummyImageChain()


class _DummySecret:
    @staticmethod
    def from_name(*args, **kwargs):
        return object()

    @staticmethod
    def from_dict(*args, **kwargs):
        return object()


class _DummyVolume:
    @staticmethod
    def from_name(*args, **kwargs):
        return object()


class _DummyRetries:
    def __init__(self, *args, **kwargs):
        pass


class _DummyApp:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, *args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def _dummy_method(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


sys.modules.setdefault(
    "modal",
    types.SimpleNamespace(
        Secret=_DummySecret,
        Volume=_DummyVolume,
        Image=_DummyImage,
        App=_DummyApp,
        Retries=_DummyRetries,
        method=_dummy_method,
    ),
)

MODULE_PATH = Path(__file__).with_name("modal_trl_posttrain.py")
MODULE_SPEC = importlib.util.spec_from_file_location("modal_trl_posttrain_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


class FakeDataset:
    def __init__(self, rows, column_names=None):
        self.rows = list(rows)
        if column_names is not None:
            self.column_names = list(column_names)
        else:
            self.column_names = list(self.rows[0].keys()) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def map(self, fn, remove_columns=None, desc=None, batched=False):
        if batched:
            raise NotImplementedError("Batched mapping is not needed for these tests.")
        return FakeDataset([fn(dict(row)) for row in self.rows])

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices], column_names=self.column_names)


def _combine_fake_datasets(datasets):
    rows = []
    column_names = []
    for dataset in datasets:
        rows.extend(dataset.rows)
        if not column_names:
            column_names = list(dataset.column_names)
    return FakeDataset(rows, column_names=column_names)


class _FakeModel:
    def eval(self):
        return self


class _FakeVolumeHandle:
    def reload(self):
        pass

    def commit(self):
        pass


class ModalTrlNormalizationTests(unittest.TestCase):
    def test_config_from_mapping_accepts_task_spec_without_beta(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "classification",
                    "target_policy": "single_target",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Classify support tickets by priority.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {"holdout_fraction": 0.1, "max_examples": 64},
                "training_estimate": {"expected_total_steps": 42},
                "output_name": "demo-run",
            }
        )

        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.task_spec["task_family"], "classification")
        self.assertEqual(config.evaluation_plan["max_examples"], 64)

    def test_config_from_mapping_accepts_generation_task_spec_with_null_evaluation_plan(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Act as a contest-math tutor.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": None,
                "training_estimate": {"expected_total_steps": 42},
                "output_name": "generation-demo-run",
            }
        )

        self.assertEqual(config.task_spec["task_family"], "generation")
        self.assertIsNone(config.evaluation_plan)

    def test_config_from_mapping_normalizes_gpu_aliases_case_insensitively(self):
        for raw_gpu_type, expected_public_gpu_type, expected_modal_gpu_type in (
            ("A10", "A10", "A10G"),
            ("a10g", "A10", "A10G"),
            ("h100", "H100", "H100"),
            ("l40s", "L40S", "L40S"),
        ):
            with self.subTest(raw_gpu_type=raw_gpu_type):
                config = MODULE._config_from_mapping(
                    {
                        "trainer_type": "sft",
                        "dataset_name": "prepared_manifest",
                        "dataset_source_type": "prepared_manifest",
                        "prepared_dataset_manifest": {"selected_datasets": []},
                        "output_name": "gpu-normalization-demo",
                        "gpu_type": raw_gpu_type,
                    }
                )

                self.assertEqual(config.gpu_type, expected_public_gpu_type)
                self.assertEqual(config.modal_gpu_type, expected_modal_gpu_type)

    def test_config_from_mapping_rejects_unknown_gpu_type(self):
        with self.assertRaisesRegex(ValueError, r"gpu_type"):
            MODULE._config_from_mapping(
                {
                    "trainer_type": "sft",
                    "dataset_name": "prepared_manifest",
                    "dataset_source_type": "prepared_manifest",
                    "prepared_dataset_manifest": {"selected_datasets": []},
                    "output_name": "bad-gpu-demo",
                    "gpu_type": "b200",
                }
            )

    def test_resolve_training_runner_selects_gpu_specific_runner_class(self):
        for raw_gpu_type, expected_runner_name, expected_modal_gpu_type in (
            ("A10", "A10GTrainingRunner", "A10G"),
            ("a10g", "A10GTrainingRunner", "A10G"),
            ("l40s", "L40STrainingRunner", "L40S"),
            ("h100", "H100TrainingRunner", "H100"),
        ):
            with self.subTest(raw_gpu_type=raw_gpu_type):
                config = MODULE._config_from_mapping(
                    {
                        "trainer_type": "sft",
                        "dataset_name": "prepared_manifest",
                        "dataset_source_type": "prepared_manifest",
                        "prepared_dataset_manifest": {"selected_datasets": []},
                        "output_name": "runner-selection-demo",
                        "gpu_type": raw_gpu_type,
                    }
                )

                runner_cls, modal_gpu_type = MODULE._resolve_training_runner(config)
                self.assertEqual(runner_cls.__name__, expected_runner_name)
                self.assertEqual(modal_gpu_type, expected_modal_gpu_type)

    def test_train_then_evaluate_impl_includes_public_and_modal_gpu_metadata(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "base_model": "meta-llama/Llama-3-8B",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "output_name": "train-result-gpu-demo",
                "gpu_type": "a10g",
            }
        )

        original_model_cache_volume = MODULE.MODEL_CACHE_VOLUME
        original_dataset_cache_volume = MODULE.DATASET_CACHE_VOLUME
        original_checkpoints_volume = MODULE.CHECKPOINTS_VOLUME
        original_write_run_config = MODULE._write_run_config
        original_load_tokenizer = MODULE._load_tokenizer
        original_load_normalized_training_datasets = MODULE._load_normalized_training_datasets
        original_build_trainer_bundle = MODULE._build_trainer_bundle
        original_resolve_resume_checkpoint = MODULE._resolve_resume_checkpoint
        original_save_final_adapter = MODULE._save_final_adapter
        original_run_offline_evaluation = MODULE._run_offline_evaluation
        original_emit_structured_lifecycle_event = MODULE._emit_structured_lifecycle_event
        original_commit_all_volumes = MODULE._commit_all_volumes

        class _FakeTrainer:
            def __init__(self):
                self.model = _FakeModel()

            def train(self, resume_from_checkpoint=None):
                return types.SimpleNamespace(global_step=7, training_loss=0.125)

        fake_volume = _FakeVolumeHandle()
        MODULE.MODEL_CACHE_VOLUME = fake_volume
        MODULE.DATASET_CACHE_VOLUME = fake_volume
        MODULE.CHECKPOINTS_VOLUME = fake_volume
        MODULE._write_run_config = lambda _config: None
        MODULE._load_tokenizer = lambda _config: types.SimpleNamespace(padding_side=None)
        MODULE._load_normalized_training_datasets = (
            lambda _config, _tokenizer: (FakeDataset([{"text": "alpha"}]), None, "language_modeling", {}, None)
        )
        MODULE._build_trainer_bundle = (
            lambda _config, _tokenizer, _train_dataset, _eval_dataset, sft_dataset_style=None: (
                _FakeTrainer(),
                {"trainer": "SFTTrainer"},
            )
        )
        MODULE._resolve_resume_checkpoint = lambda _path: None
        MODULE._save_final_adapter = lambda _trainer, _tokenizer, _path: None
        MODULE._run_offline_evaluation = (
            lambda *_args, **_kwargs: ({"metrics": {"accuracy": 1.0}}, {"task_family": "classification"})
        )
        MODULE._emit_structured_lifecycle_event = lambda *_args, **_kwargs: None
        MODULE._commit_all_volumes = lambda: None
        try:
            result = MODULE._train_then_evaluate_impl(config)
        finally:
            MODULE.MODEL_CACHE_VOLUME = original_model_cache_volume
            MODULE.DATASET_CACHE_VOLUME = original_dataset_cache_volume
            MODULE.CHECKPOINTS_VOLUME = original_checkpoints_volume
            MODULE._write_run_config = original_write_run_config
            MODULE._load_tokenizer = original_load_tokenizer
            MODULE._load_normalized_training_datasets = original_load_normalized_training_datasets
            MODULE._build_trainer_bundle = original_build_trainer_bundle
            MODULE._resolve_resume_checkpoint = original_resolve_resume_checkpoint
            MODULE._save_final_adapter = original_save_final_adapter
            MODULE._run_offline_evaluation = original_run_offline_evaluation
            MODULE._emit_structured_lifecycle_event = original_emit_structured_lifecycle_event
            MODULE._commit_all_volumes = original_commit_all_volumes

        training_result = result["training_result"]
        self.assertEqual(training_result["gpu_type"], "A10")
        self.assertEqual(training_result["modal_gpu_type"], "A10G")
        self.assertEqual(training_result["global_step"], 7)
        self.assertEqual(training_result["training_loss"], 0.125)
        self.assertEqual(result["evaluation"]["metrics"]["accuracy"], 1.0)

    def test_should_build_merged_artifact_only_when_requested(self):
        standard_config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "base_model": "Qwen/Qwen3-8B-Base",
                "dataset_name": "trl-lib/Capybara",
                "output_name": "qwen3-standard-demo",
            }
        )
        merged_config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "base_model": "Qwen/Qwen3-8B-Base",
                "dataset_name": "trl-lib/Capybara",
                "output_name": "qwen3-merged-demo",
                "merge_after_train": True,
            }
        )

        self.assertFalse(MODULE._should_build_merged_artifact(standard_config))
        self.assertTrue(MODULE._should_build_merged_artifact(merged_config))

    def test_train_then_evaluate_impl_reuses_candidate_model_and_builds_merged_dir_when_requested(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "base_model": "Qwen/Qwen3-8B-Base",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Draft clinical notes.",
                    "unsupported_reason": None,
                },
                "output_name": "qwen-merged-demo",
                "merge_after_train": True,
            }
        )

        original_model_cache_volume = MODULE.MODEL_CACHE_VOLUME
        original_dataset_cache_volume = MODULE.DATASET_CACHE_VOLUME
        original_checkpoints_volume = MODULE.CHECKPOINTS_VOLUME
        original_write_run_config = MODULE._write_run_config
        original_load_tokenizer = MODULE._load_tokenizer
        original_load_normalized_training_datasets = MODULE._load_normalized_training_datasets
        original_build_trainer_bundle = MODULE._build_trainer_bundle
        original_resolve_resume_checkpoint = MODULE._resolve_resume_checkpoint
        original_save_final_adapter = MODULE._save_final_adapter
        original_run_offline_evaluation = MODULE._run_offline_evaluation
        original_merge_adapter_into_base = MODULE._merge_adapter_into_base
        original_emit_structured_lifecycle_event = MODULE._emit_structured_lifecycle_event
        original_commit_all_volumes = MODULE._commit_all_volumes

        class _FakeTrainer:
            def __init__(self):
                self.model = _FakeModel()

            def train(self, resume_from_checkpoint=None):
                return types.SimpleNamespace(global_step=11, training_loss=0.25)

        fake_volume = _FakeVolumeHandle()
        lifecycle_events = []
        captured_candidate_models = []
        merge_calls = []
        trainer = _FakeTrainer()
        original_candidate_model = trainer.model

        MODULE.MODEL_CACHE_VOLUME = fake_volume
        MODULE.DATASET_CACHE_VOLUME = fake_volume
        MODULE.CHECKPOINTS_VOLUME = fake_volume
        MODULE._write_run_config = lambda _config: None
        MODULE._load_tokenizer = lambda _config: types.SimpleNamespace(padding_side=None)
        MODULE._load_normalized_training_datasets = (
            lambda _config, _tokenizer: (FakeDataset([{"text": "alpha"}]), FakeDataset([{"text": "beta"}]), "language_modeling", {}, None)
        )
        MODULE._build_trainer_bundle = (
            lambda _config, _tokenizer, _train_dataset, _eval_dataset, sft_dataset_style=None: (
                trainer,
                {"trainer": "SFTTrainer"},
            )
        )
        MODULE._resolve_resume_checkpoint = lambda _path: None
        MODULE._save_final_adapter = lambda _trainer, _tokenizer, _path: None
        MODULE._run_offline_evaluation = (
            lambda *_args, **kwargs: captured_candidate_models.append(kwargs["candidate_model"]) or (None, {"task_family": "generation"})
        )
        MODULE._merge_adapter_into_base = lambda _config, _tokenizer: merge_calls.append(str(_config.merged_dir))
        MODULE._emit_structured_lifecycle_event = lambda event_type, **payload: lifecycle_events.append((event_type, payload))
        MODULE._commit_all_volumes = lambda: None
        try:
            result = MODULE._train_then_evaluate_impl(config)
        finally:
            MODULE.MODEL_CACHE_VOLUME = original_model_cache_volume
            MODULE.DATASET_CACHE_VOLUME = original_dataset_cache_volume
            MODULE.CHECKPOINTS_VOLUME = original_checkpoints_volume
            MODULE._write_run_config = original_write_run_config
            MODULE._load_tokenizer = original_load_tokenizer
            MODULE._load_normalized_training_datasets = original_load_normalized_training_datasets
            MODULE._build_trainer_bundle = original_build_trainer_bundle
            MODULE._resolve_resume_checkpoint = original_resolve_resume_checkpoint
            MODULE._save_final_adapter = original_save_final_adapter
            MODULE._run_offline_evaluation = original_run_offline_evaluation
            MODULE._merge_adapter_into_base = original_merge_adapter_into_base
            MODULE._emit_structured_lifecycle_event = original_emit_structured_lifecycle_event
            MODULE._commit_all_volumes = original_commit_all_volumes

        self.assertEqual(captured_candidate_models, [original_candidate_model])
        self.assertEqual(len(merge_calls), 1)
        self.assertEqual(lifecycle_events[0][0], "training_complete")
        self.assertEqual(
            result["training_result"]["merged_dir"],
            str(config.merged_dir),
        )
        self.assertIsNone(trainer.model)

    def test_main_rejects_standalone_evaluate_mode(self):
        original_load_yaml_mapping = MODULE._load_yaml_mapping
        original_env = MODULE.os.environ.get("POSTTRAINING_RUN_MODE")
        MODULE._load_yaml_mapping = lambda _config: {
            "trainer_type": "sft",
            "dataset_name": "prepared_manifest",
            "dataset_source_type": "prepared_manifest",
            "prepared_dataset_manifest": {"selected_datasets": []},
            "output_name": "main-mode-demo",
        }
        MODULE.os.environ["POSTTRAINING_RUN_MODE"] = "evaluate"
        try:
            with self.assertRaisesRegex(ValueError, "train_then_evaluate"):
                MODULE.main(config="unused.yaml")
        finally:
            MODULE._load_yaml_mapping = original_load_yaml_mapping
            if original_env is None:
                del MODULE.os.environ["POSTTRAINING_RUN_MODE"]
            else:
                MODULE.os.environ["POSTTRAINING_RUN_MODE"] = original_env

    def test_create_stratified_holdout_preserves_each_label(self):
        dataset = FakeDataset(
            [
                {"prompt": "P1", "completion": "high"},
                {"prompt": "P2", "completion": "high"},
                {"prompt": "P3", "completion": "medium"},
                {"prompt": "P4", "completion": "medium"},
                {"prompt": "P5", "completion": "low"},
                {"prompt": "P6", "completion": "low"},
            ]
        )

        train_dataset, eval_dataset, metadata = MODULE._create_stratified_holdout(
            dataset,
            fraction=0.1,
            seed=42,
        )

        self.assertEqual(len(train_dataset) + len(eval_dataset), len(dataset))
        self.assertEqual(sorted(metadata["train_label_distribution"].keys()), ["high", "low", "medium"])
        self.assertEqual(sorted(metadata["eval_label_distribution"].keys()), ["high", "low", "medium"])

    def test_resolve_source_splits_prefers_manifest_values_and_falls_back_to_train_split(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "classification",
                    "target_policy": "single_target",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Classify support tickets by priority.",
                    "unsupported_reason": None,
                },
                "output_name": "demo-run",
            }
        )

        self.assertEqual(
            MODULE._resolve_source_splits_for_entry(
                {
                    "source_splits": ["train", "validation", "train", "test"],
                    "train_split": "train",
                },
                config,
            ),
            ["train", "validation", "test"],
        )
        self.assertEqual(
            MODULE._resolve_source_splits_for_entry(
                {
                    "train_split": "custom-train",
                },
                config,
            ),
            ["custom-train"],
        )

    def test_create_random_holdout_is_deterministic(self):
        dataset = FakeDataset(
            [
                {"text": "a"},
                {"text": "b"},
                {"text": "c"},
                {"text": "d"},
            ]
        )

        train_one, eval_one, metadata_one = MODULE._create_random_holdout(dataset, fraction=0.25, seed=42)
        train_two, eval_two, metadata_two = MODULE._create_random_holdout(dataset, fraction=0.25, seed=42)

        self.assertEqual([row["text"] for row in train_one.rows], [row["text"] for row in train_two.rows])
        self.assertEqual([row["text"] for row in eval_one.rows], [row["text"] for row in eval_two.rows])
        self.assertEqual(metadata_one["strategy"], "deterministic_random_holdout")
        self.assertEqual(metadata_one["eval_examples"], 1)
        self.assertEqual(metadata_one, metadata_two)

    def test_sample_eval_dataset_uses_all_when_small_and_caps_when_large(self):
        small_dataset = FakeDataset(
            [
                {"text": "a"},
                {"text": "b"},
                {"text": "c"},
            ]
        )
        large_dataset = FakeDataset(
            [
                {"text": f"row-{index}"}
                for index in range(40)
            ]
        )

        sampled_small = MODULE._sample_eval_dataset(
            small_dataset,
            max_examples=MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
            seed=42,
        )
        sampled_large = MODULE._sample_eval_dataset(
            large_dataset,
            max_examples=MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
            seed=42,
        )

        self.assertEqual(len(sampled_small), 3)
        self.assertEqual(len(sampled_large), MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES)

    def test_reweight_train_dataset_by_provenance_groups_rows_by_dataset(self):
        train_dataset = FakeDataset(
            [
                {"text": "alpha-1", MODULE.PROVENANCE_DATASET_COLUMN: "alpha"},
                {"text": "beta-1", MODULE.PROVENANCE_DATASET_COLUMN: "beta"},
                {"text": "alpha-2", MODULE.PROVENANCE_DATASET_COLUMN: "alpha"},
            ]
        )

        original_mix = MODULE._mix_datasets
        MODULE._mix_datasets = lambda datasets, probabilities: {
            "datasets": datasets,
            "probabilities": probabilities,
        }
        try:
            remixed = MODULE._reweight_train_dataset_by_provenance(
                train_dataset,
                {"alpha": 0.7, "beta": 0.3, "missing": 0.1},
            )
        finally:
            MODULE._mix_datasets = original_mix

        self.assertEqual(remixed["probabilities"], [0.7, 0.3])
        self.assertEqual([row["text"] for row in remixed["datasets"][0].rows], ["alpha-1", "alpha-2"])
        self.assertEqual([row["text"] for row in remixed["datasets"][1].rows], ["beta-1"])

    def test_passthrough_completion_accepts_unseen_labels_without_mapping(self):
        dataset = FakeDataset(
            [
                {"body": "Router offline", "priority": "high"},
                {"body": "General question", "priority": "low"},
            ]
        )
        entry = {
            "dataset": "acme/priority-dataset",
            "normalization": {
                "shape": "prompt_completion",
                "source_columns": ["body", "priority"],
                "fields": {
                    "text": None,
                    "prompt": {
                        "source_column": None,
                        "template": "Classify the following example.\n\nInput:\n{body}\n\nLabel:",
                        "value_mapping": None,
                    },
                    "completion": {
                        "source_column": "priority",
                        "template": None,
                        "value_mapping": None,
                    },
                },
            },
        }

        transformed = MODULE._transform_dataset_with_normalization("acme/priority-dataset", dataset, entry)

        self.assertEqual(
            [row["completion"] for row in transformed.rows],
            ["high", "low"],
        )

    def test_explicit_value_mapping_remaps_known_labels_and_fails_on_unknown_ones(self):
        known_dataset = FakeDataset(
            [
                {"body": "Router offline", "priority": "high"},
                {"body": "Refund request", "priority": "medium"},
            ]
        )
        entry = {
            "dataset": "acme/priority-dataset",
            "normalization": {
                "shape": "prompt_completion",
                "source_columns": ["body", "priority"],
                "fields": {
                    "text": None,
                    "prompt": {
                        "source_column": None,
                        "template": "Classify the following example.\n\nInput:\n{body}\n\nLabel:",
                        "value_mapping": None,
                    },
                    "completion": {
                        "source_column": "priority",
                        "template": None,
                        "value_mapping": {
                            "high": "urgent",
                            "medium": "normal",
                        },
                    },
                },
            },
        }

        transformed = MODULE._transform_dataset_with_normalization("acme/priority-dataset", known_dataset, entry)
        self.assertEqual(
            [row["completion"] for row in transformed.rows],
            ["urgent", "normal"],
        )

        unknown_dataset = FakeDataset(
            [
                {"body": "General question", "priority": "low"},
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            r"acme/priority-dataset.*normalization\.fields\.completion.*low",
        ):
            MODULE._transform_dataset_with_normalization("acme/priority-dataset", unknown_dataset, entry)

    def test_prepare_prepared_dataset_entry_drops_missing_target_rows_and_reports_counts(self):
        dataset = FakeDataset(
            [
                {"body": "Router offline", "type": "technical"},
                {"body": "Refund request", "type": ""},
                {"body": "Need an update", "type": None},
                {"body": "Billing question", "type": "billing"},
            ]
        )
        entry = {
            "dataset": "acme/support-tickets",
            "selected_target_column": "type",
            "normalization": {
                "shape": "prompt_completion",
                "source_columns": ["body", "type"],
                "fields": {
                    "text": None,
                    "prompt": {
                        "source_column": None,
                        "template": "Classify the following support ticket.\n\nTicket:\n{body}\n\nLabel:",
                        "value_mapping": None,
                    },
                    "completion": {
                        "source_column": "type",
                        "template": None,
                        "value_mapping": None,
                    },
                },
            },
        }

        transformed, diagnostics = MODULE._prepare_prepared_dataset_entry(dataset, entry, split_name="train")

        self.assertEqual(len(transformed), 2)
        self.assertEqual(
            [row["completion"] for row in transformed.rows],
            ["technical", "billing"],
        )
        self.assertEqual(diagnostics["dataset"], "acme/support-tickets")
        self.assertEqual(diagnostics["split"], "train")
        self.assertEqual(diagnostics["selected_target_column"], "type")
        self.assertEqual(diagnostics["total_rows"], 4)
        self.assertEqual(diagnostics["kept_rows"], 2)
        self.assertEqual(diagnostics["dropped_rows_missing_target"], 2)

        summary = MODULE._summarize_preprocessing_diagnostics([diagnostics])
        self.assertEqual(
            summary["missing_target_label_filtering"]["dropped_rows_missing_target"],
            2,
        )

    def test_prepare_prepared_dataset_entry_rejects_image_blob_source_columns(self):
        dataset = FakeDataset(
            [
                {
                    "image": {"bytes": "iVBORw0KGgoAAAANSUhEUgAA", "path": None},
                    "caption": "Portable chest radiograph with right basilar opacity.",
                    "cui": "atelectasis",
                }
            ]
        )
        entry = {
            "dataset": "acme/radiology-images",
            "selected_target_column": "cui",
            "normalization": {
                "shape": "prompt_completion",
                "source_columns": ["image", "caption", "cui"],
                "fields": {
                    "text": None,
                    "prompt": {
                        "source_column": None,
                        "template": "Image bytes: {image}\nCaption: {caption}\n\nLabel:",
                        "value_mapping": None,
                    },
                    "completion": {
                        "source_column": "cui",
                        "template": None,
                        "value_mapping": None,
                    },
                },
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            r"acme/radiology-images.*unsupported image/blob columns \['image'\]",
        ):
            MODULE._prepare_prepared_dataset_entry(dataset, entry, split_name="train")

    def test_load_prepared_manifest_datasets_fails_early_when_all_target_labels_are_missing(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {
                    "selected_datasets": [
                        {
                            "dataset": "acme/support-tickets",
                            "train_split": "train",
                            "weight": 1,
                            "selected_target_column": "type",
                            "normalization": {
                                "shape": "prompt_completion",
                                "source_columns": ["body", "type"],
                                "fields": {
                                    "text": None,
                                    "prompt": {
                                        "source_column": None,
                                        "template": "Classify the following support ticket.\n\nTicket:\n{body}\n\nLabel:",
                                        "value_mapping": None,
                                    },
                                    "completion": {
                                        "source_column": "type",
                                        "template": None,
                                        "value_mapping": None,
                                    },
                                },
                            },
                        }
                    ]
                },
                "task_spec": {
                    "supported": True,
                    "task_family": "classification",
                    "target_policy": "single_target",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Classify support tickets by issue type.",
                    "unsupported_reason": None,
                },
                "output_name": "demo-run",
            }
        )

        original_loader = MODULE._load_single_dataset
        MODULE._load_single_dataset = lambda **kwargs: FakeDataset(
            [
                {"body": "Router offline", "type": None},
                {"body": "Refund request", "type": ""},
            ]
        )
        try:
            with self.assertRaisesRegex(
                ValueError,
                r"acme/support-tickets.*no usable rows after dropping examples with missing target labels.*`type`",
            ):
                MODULE._load_prepared_manifest_datasets(config)
        finally:
            MODULE._load_single_dataset = original_loader

    def test_load_sft_prepared_manifest_datasets_merges_source_splits_for_classification(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {
                    "selected_datasets": [
                        {
                            "dataset": "acme/support-tickets",
                            "dataset_config": "default",
                            "train_split": "train",
                            "source_splits": ["train", "validation", "test"],
                            "weight": 1,
                            "selected_target_column": "priority",
                            "normalization": {
                                "shape": "prompt_completion",
                                "source_columns": ["body", "priority"],
                                "fields": {
                                    "text": None,
                                    "prompt": {
                                        "source_column": None,
                                        "template": "Classify the following support ticket.\n\nTicket:\n{body}\n\nLabel:",
                                        "value_mapping": None,
                                    },
                                    "completion": {
                                        "source_column": "priority",
                                        "template": None,
                                        "value_mapping": None,
                                    },
                                },
                            },
                        }
                    ]
                },
                "task_spec": {
                    "supported": True,
                    "task_family": "classification",
                    "target_policy": "single_target",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Classify support tickets by priority.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "strategy": "merged_sft_holdout",
                    "holdout_fraction": 0.1,
                    "deterministic_seed": 42,
                },
                "output_name": "classification-holdout-demo",
            }
        )

        datasets_by_split = {
            "train": FakeDataset(
                [
                    {"body": "Router offline", "priority": "high"},
                    {"body": "Password reset", "priority": "medium"},
                ]
            ),
            "validation": FakeDataset(
                [
                    {"body": "Refund request", "priority": "low"},
                    {"body": "Need status update", "priority": "high"},
                ]
            ),
            "test": FakeDataset(
                [
                    {"body": "Billing issue", "priority": "medium"},
                    {"body": "General question", "priority": "low"},
                ]
            ),
        }

        original_loader = MODULE._load_single_dataset
        original_concat = MODULE._concat_datasets
        original_mix = MODULE._mix_datasets
        MODULE._load_single_dataset = lambda dataset_name, dataset_config, split: datasets_by_split[split]
        MODULE._concat_datasets = _combine_fake_datasets
        MODULE._mix_datasets = lambda datasets, probabilities: _combine_fake_datasets(datasets)
        try:
            train_dataset, eval_dataset, diagnostics, holdout_metadata = MODULE._load_sft_prepared_manifest_datasets(
                config
            )
        finally:
            MODULE._load_single_dataset = original_loader
            MODULE._concat_datasets = original_concat
            MODULE._mix_datasets = original_mix

        self.assertIsNotNone(eval_dataset)
        self.assertEqual(len(train_dataset) + len(eval_dataset), 6)
        self.assertEqual(holdout_metadata["strategy"], "stratified_completion_holdout")
        self.assertEqual(
            holdout_metadata["source_splits_by_dataset"]["acme/support-tickets"],
            ["train", "validation", "test"],
        )
        self.assertEqual(sorted(holdout_metadata["eval_label_distribution"].keys()), ["high", "low", "medium"])
        self.assertEqual(
            diagnostics["invalid_sft_example_filtering"]["dropped_rows_invalid_examples"],
            0,
        )

    def test_load_sft_prepared_manifest_datasets_creates_holdout_for_generation(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {
                    "selected_datasets": [
                        {
                            "dataset": "acme/tutor-text",
                            "dataset_config": "default",
                            "train_split": "train",
                            "source_splits": ["train", "test"],
                            "weight": 1,
                            "normalization": {
                                "shape": "text",
                                "source_columns": ["text"],
                                "fields": {
                                    "text": {
                                        "source_column": "text",
                                        "template": None,
                                        "value_mapping": None,
                                    },
                                    "prompt": None,
                                    "completion": None,
                                },
                            },
                        }
                    ]
                },
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Act as a contest-math tutor.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "strategy": "merged_sft_holdout",
                    "holdout_fraction": 0.1,
                    "deterministic_seed": 42,
                },
                "output_name": "generation-holdout-demo",
            }
        )

        datasets_by_split = {
            "train": FakeDataset(
                [
                    {"text": "Tutor the student through an AMC problem step by step."},
                    {"text": "Explain why symmetry helps here before doing arithmetic."},
                ]
            ),
            "test": FakeDataset(
                [
                    {"text": "Model a short but thoughtful AIME-style explanation."},
                    {"text": "Encourage the student to sanity-check the answer."},
                ]
            ),
        }

        original_loader = MODULE._load_single_dataset
        original_concat = MODULE._concat_datasets
        original_mix = MODULE._mix_datasets
        MODULE._load_single_dataset = lambda dataset_name, dataset_config, split: datasets_by_split[split]
        MODULE._concat_datasets = _combine_fake_datasets
        MODULE._mix_datasets = lambda datasets, probabilities: _combine_fake_datasets(datasets)
        try:
            train_dataset, eval_dataset, diagnostics, holdout_metadata = MODULE._load_sft_prepared_manifest_datasets(
                config
            )
        finally:
            MODULE._load_single_dataset = original_loader
            MODULE._concat_datasets = original_concat
            MODULE._mix_datasets = original_mix

        self.assertIsNotNone(eval_dataset)
        self.assertGreater(len(eval_dataset), 0)
        self.assertEqual(len(train_dataset) + len(eval_dataset), 4)
        self.assertEqual(holdout_metadata["strategy"], "deterministic_random_holdout")
        self.assertEqual(
            holdout_metadata["source_splits_by_dataset"]["acme/tutor-text"],
            ["train", "test"],
        )
        self.assertEqual(
            diagnostics["invalid_sft_example_filtering"]["dropped_rows_invalid_examples"],
            0,
        )

    def test_evaluate_classification_model_comparison_reports_deltas_and_disagreements(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "classification",
                    "target_policy": "single_target",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Classify support tickets by priority.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "deterministic_seed": 42,
                    "comparison_max_examples": MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
                    "max_examples": 64,
                },
                "output_name": "comparison-demo",
            }
        )
        eval_dataset = FakeDataset(
            [
                {"prompt": "P1", "completion": "high"},
                {"prompt": "P2", "completion": "low"},
                {"prompt": "P3", "completion": "medium"},
            ]
        )

        base_model = _FakeModel()
        candidate_model = _FakeModel()
        original_load_base = MODULE._load_base_model
        original_load_adapter = MODULE._load_adapter_inference_model
        original_predict = MODULE._predict_classification_outputs
        original_eval = MODULE._evaluate_classification_dataset
        original_clear = MODULE._clear_inference_model
        MODULE._load_base_model = lambda config: base_model
        MODULE._load_adapter_inference_model = lambda config: candidate_model
        MODULE._clear_inference_model = lambda model: None

        def fake_predict(model, tokenizer, dataset):
            if model is base_model:
                return {
                    "gold_labels": ["high", "low", "medium"],
                    "labels": ["high", "low", "medium"],
                    "predictions": ["high", None, "low"],
                    "raw_predictions": ["high", "unknown", "low"],
                }
            return {
                "gold_labels": ["high", "low", "medium"],
                "labels": ["high", "low", "medium"],
                "predictions": ["high", "low", "medium"],
                "raw_predictions": ["high", "low", "medium"],
            }

        MODULE._predict_classification_outputs = fake_predict
        MODULE._evaluate_classification_dataset = lambda *args, **kwargs: {
            "metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "invalid_label_rate": 0.0,
                "label_coverage": 1.0,
            },
            "sampled_examples": 3,
        }

        try:
            evaluation_result, comparison = MODULE._evaluate_classification_model_comparison(
                config,
                tokenizer=None,
                eval_dataset=eval_dataset,
                holdout_metadata={"strategy": "test_holdout"},
            )
        finally:
            MODULE._load_base_model = original_load_base
            MODULE._load_adapter_inference_model = original_load_adapter
            MODULE._predict_classification_outputs = original_predict
            MODULE._evaluate_classification_dataset = original_eval
            MODULE._clear_inference_model = original_clear

        self.assertEqual(comparison["summary"]["winner"], "candidate")
        self.assertEqual(comparison["summary"]["disagreement_counts"]["candidate_only_correct"], 2)
        self.assertEqual(comparison["sample_policy"]["sampled_cases"], 3)
        self.assertEqual(comparison["holdout"]["strategy"], "test_holdout")
        self.assertEqual(evaluation_result["holdout"]["strategy"], "test_holdout")

    def test_evaluate_generation_model_comparison_aggregates_judgments(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Act as a contest-math tutor.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "deterministic_seed": 42,
                    "comparison_max_examples": MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
                    "show_evaluation_component": False,
                },
                "output_name": "generation-comparison-demo",
            }
        )
        eval_dataset = FakeDataset(
            [
                {"text": "First tutoring example."},
                {"text": "Second tutoring example."},
            ]
        )

        base_model = _FakeModel()
        candidate_model = _FakeModel()
        original_load_base = MODULE._load_base_model
        original_load_adapter = MODULE._load_adapter_inference_model
        original_generate = MODULE._predict_generation_response
        original_synthesize = MODULE._synthesize_generation_case
        original_score = MODULE._score_generation_output_against_reference
        original_clear = MODULE._clear_inference_model
        MODULE._load_base_model = lambda config: base_model
        MODULE._load_adapter_inference_model = lambda config: candidate_model
        MODULE._clear_inference_model = lambda model: None
        MODULE._synthesize_generation_case = lambda source_text, judge_model: {
            "prompt": f"Prompt for {source_text}",
            "reference_answer": f"Reference for {source_text}",
            "rubric": ["helpful", "faithful", "clear"],
            "source_summary": f"Summary for {source_text}",
        }
        captured_prompts = []

        def fake_generate(model, tokenizer, prompt, max_new_tokens):
            captured_prompts.append(prompt)
            prefix = "base" if model is base_model else "candidate"
            return f"{prefix} output for {prompt}"

        MODULE._predict_generation_response = fake_generate
        captured_scores = []

        def fake_score(**kwargs):
            captured_scores.append(kwargs)
            if "First tutoring example." in kwargs["prompt"]:
                if kwargs["output"].startswith("base output"):
                    return {
                        "score": 6.0,
                        "matches_expected_output": False,
                        "reason": "Base output misses key details.",
                    }
                return {
                    "score": 8.5,
                    "matches_expected_output": True,
                    "reason": "Candidate output matches the expected answer.",
                }
            return {
                "score": 7.0,
                "matches_expected_output": True,
                "reason": "The output is good enough to count as a match.",
            }

        MODULE._score_generation_output_against_reference = fake_score

        try:
            comparison = MODULE._evaluate_generation_model_comparison(
                config,
                tokenizer=None,
                eval_dataset=eval_dataset,
                holdout_metadata={"strategy": "test_holdout"},
            )
        finally:
            MODULE._load_base_model = original_load_base
            MODULE._load_adapter_inference_model = original_load_adapter
            MODULE._predict_generation_response = original_generate
            MODULE._synthesize_generation_case = original_synthesize
            MODULE._score_generation_output_against_reference = original_score
            MODULE._clear_inference_model = original_clear

        self.assertEqual(len(captured_scores), 4)
        self.assertEqual(comparison["summary"]["match_threshold_score"], 7.0)
        self.assertEqual(comparison["summary"]["baseline_match_count"], 1)
        self.assertEqual(comparison["summary"]["candidate_match_count"], 2)
        self.assertEqual(comparison["summary"]["baseline_match_rate"], 0.5)
        self.assertEqual(comparison["summary"]["candidate_match_rate"], 1.0)
        self.assertEqual(comparison["show_evaluation_component"], False)
        self.assertEqual(comparison["sample_policy"]["sampled_cases"], 2)
        self.assertEqual(comparison["holdout"]["strategy"], "test_holdout")
        self.assertEqual(len(captured_prompts), 4)
        self.assertIn("Task:\nPrompt for First tutoring example.", captured_prompts[0])
        self.assertIn("Source text:\nFirst tutoring example.", captured_prompts[0])
        self.assertIn("Return only the final answer.", captured_prompts[0])
        self.assertEqual(comparison["cases"][0]["prompt"], "Prompt for First tutoring example.")
        self.assertIn("Task:\nPrompt for First tutoring example.", comparison["cases"][0]["model_input_preview"])
        self.assertIn("Source text:\nFirst tutoring example.", comparison["cases"][0]["model_input_preview"])
        self.assertEqual(comparison["cases"][1]["baseline_judgment"]["score"], 7.0)
        self.assertTrue(comparison["cases"][1]["baseline_judgment"]["matches_expected_output"])

    def test_evaluate_generation_model_comparison_uses_source_text_without_known_target_section(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Draft clinical notes from encounters.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "deterministic_seed": 42,
                    "comparison_max_examples": MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
                },
                "output_name": "grounded-generation-comparison-demo",
            }
        )
        eval_dataset = FakeDataset(
            [
                {
                    "text": (
                        "### Conversation\n"
                        "Doctor: What happened?\n"
                        "Patient: My head hurts badly.\n"
                        "Guest_family: She has advanced dementia.\n\n"
                        "### Clinical Note\n"
                        "Symptoms: headache. History limited by dementia."
                    )
                }
            ]
        )

        base_model = _FakeModel()
        candidate_model = _FakeModel()
        original_load_base = MODULE._load_base_model
        original_load_adapter = MODULE._load_adapter_inference_model
        original_generate = MODULE._predict_generation_response
        original_synthesize = MODULE._synthesize_generation_case
        original_score = MODULE._score_generation_output_against_reference
        original_clear = MODULE._clear_inference_model
        captured_prompts = []
        MODULE._load_base_model = lambda config: base_model
        MODULE._load_adapter_inference_model = lambda config: candidate_model
        MODULE._clear_inference_model = lambda model: None
        MODULE._synthesize_generation_case = lambda source_text, judge_model: {
            "prompt": "Write a brief clinical note based on this encounter.",
            "reference_answer": "Clinical Note: ...",
            "rubric": ["helpful", "faithful", "clear"],
            "source_summary": "Clinical encounter.",
        }

        def fake_generate(model, tokenizer, prompt, max_new_tokens):
            captured_prompts.append(prompt)
            return "stub output"

        MODULE._predict_generation_response = fake_generate
        MODULE._score_generation_output_against_reference = lambda **kwargs: {
            "score": 7.0,
            "matches_expected_output": True,
            "reason": "The output matches the expected answer.",
        }

        try:
            comparison = MODULE._evaluate_generation_model_comparison(
                config,
                tokenizer=None,
                eval_dataset=eval_dataset,
                holdout_metadata=None,
            )
        finally:
            MODULE._load_base_model = original_load_base
            MODULE._load_adapter_inference_model = original_load_adapter
            MODULE._predict_generation_response = original_generate
            MODULE._synthesize_generation_case = original_synthesize
            MODULE._score_generation_output_against_reference = original_score
            MODULE._clear_inference_model = original_clear

        self.assertEqual(len(captured_prompts), 2)
        self.assertIn("Write a brief clinical note based on this encounter.", captured_prompts[0])
        self.assertIn("Patient: My head hurts badly.", captured_prompts[0])
        self.assertIn("Guest_family: She has advanced dementia.", captured_prompts[0])
        self.assertNotIn("Symptoms: headache. History limited by dementia.", captured_prompts[0])
        self.assertEqual(
            comparison["cases"][0]["prompt"],
            "Write a brief clinical note based on this encounter.",
        )
        self.assertIn("Patient: My head hurts badly.", comparison["cases"][0]["model_input_preview"])
        self.assertNotIn(
            "Symptoms: headache. History limited by dementia.",
            comparison["cases"][0]["model_input_preview"],
        )

    def test_evaluate_generation_model_comparison_strips_thinking_blocks_before_judging(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "text",
                    "objective_summary": "Draft helpful responses.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "deterministic_seed": 42,
                    "comparison_max_examples": MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
                },
                "output_name": "generation-comparison-strip-think",
            }
        )
        eval_dataset = FakeDataset([{"text": "Example source text."}])

        base_model = _FakeModel()
        candidate_model = _FakeModel()
        original_load_base = MODULE._load_base_model
        original_load_adapter = MODULE._load_adapter_inference_model
        original_generate = MODULE._predict_generation_response
        original_synthesize = MODULE._synthesize_generation_case
        original_score = MODULE._score_generation_output_against_reference
        original_clear = MODULE._clear_inference_model
        MODULE._load_base_model = lambda config: base_model
        MODULE._load_adapter_inference_model = lambda config: candidate_model
        MODULE._clear_inference_model = lambda model: None
        MODULE._synthesize_generation_case = lambda source_text, judge_model: {
            "prompt": "Answer the example.",
            "reference_answer": "Visible answer.",
            "rubric": ["helpful", "faithful", "clear"],
            "source_summary": "Example summary.",
        }

        def fake_generate(model, tokenizer, prompt, max_new_tokens):
            if model is base_model:
                return "<think>\ninternal base reasoning\n</think>\n\nVisible base answer."
            return "Visible candidate intro.\n\n<think>\ninternal candidate reasoning\n</think>\n\nVisible candidate answer."

        captured_judgments = []

        def fake_score(**kwargs):
            captured_judgments.append(kwargs)
            return {
                "score": 8.0 if kwargs["output"].startswith("Visible candidate") else 3.0,
                "matches_expected_output": kwargs["output"].startswith("Visible candidate"),
                "reason": "Checked the cleaned output.",
            }

        MODULE._predict_generation_response = fake_generate
        MODULE._score_generation_output_against_reference = fake_score

        try:
            comparison = MODULE._evaluate_generation_model_comparison(
                config,
                tokenizer=None,
                eval_dataset=eval_dataset,
                holdout_metadata=None,
            )
        finally:
            MODULE._load_base_model = original_load_base
            MODULE._load_adapter_inference_model = original_load_adapter
            MODULE._predict_generation_response = original_generate
            MODULE._synthesize_generation_case = original_synthesize
            MODULE._score_generation_output_against_reference = original_score
            MODULE._clear_inference_model = original_clear

        self.assertEqual(len(captured_judgments), 2)
        self.assertEqual(captured_judgments[0]["output"], "Visible base answer.")
        self.assertEqual(
            captured_judgments[1]["output"],
            "Visible candidate intro.\n\nVisible candidate answer.",
        )
        self.assertEqual(comparison["cases"][0]["baseline_output"], "Visible base answer.")
        self.assertEqual(
            comparison["cases"][0]["candidate_output"],
            "Visible candidate intro.\n\nVisible candidate answer.",
        )
        self.assertFalse(comparison["cases"][0]["baseline_judgment"]["matches_expected_output"])
        self.assertTrue(comparison["cases"][0]["candidate_judgment"]["matches_expected_output"])

    def test_evaluate_generation_prompt_completion_model_comparison_uses_gold_completions(self):
        config = MODULE._config_from_mapping(
            {
                "trainer_type": "sft",
                "dataset_name": "prepared_manifest",
                "dataset_source_type": "prepared_manifest",
                "prepared_dataset_manifest": {"selected_datasets": []},
                "task_spec": {
                    "supported": True,
                    "task_family": "generation",
                    "target_policy": "none",
                    "output_shape_preference": "prompt_completion",
                    "objective_summary": "Draft tutoring answers.",
                    "unsupported_reason": None,
                },
                "evaluation_plan": {
                    "deterministic_seed": 42,
                    "comparison_max_examples": MODULE.DEFAULT_COMPARISON_MAX_EXAMPLES,
                },
                "output_name": "generation-prompt-completion-demo",
            }
        )
        eval_dataset = FakeDataset(
            [
                {"prompt": "Solve problem 1", "completion": "Gold answer 1"},
                {"prompt": "Solve problem 2", "completion": "Gold answer 2"},
            ]
        )

        base_model = _FakeModel()
        candidate_model = _FakeModel()
        original_load_base = MODULE._load_base_model
        original_load_adapter = MODULE._load_adapter_inference_model
        original_generate = MODULE._predict_generation_response
        original_score = MODULE._score_generation_output_against_reference
        original_clear = MODULE._clear_inference_model
        MODULE._load_base_model = lambda config: base_model
        MODULE._load_adapter_inference_model = lambda config: candidate_model
        MODULE._clear_inference_model = lambda model: None

        captured_prompts = []
        captured_judgments = []

        def fake_generate(model, tokenizer, prompt, max_new_tokens):
            captured_prompts.append(prompt)
            return ("base" if model is base_model else "candidate") + f" output for {prompt}"

        def fake_score(**kwargs):
            captured_judgments.append(kwargs)
            if kwargs["prompt"] == "Solve problem 1":
                if kwargs["output"].startswith("base output"):
                    return {
                        "score": 2.0,
                        "matches_expected_output": False,
                        "reason": "Base output misses the expected answer.",
                    }
                return {
                    "score": 8.0,
                    "matches_expected_output": True,
                    "reason": "Candidate output matches the expected answer.",
                }
            return {
                "score": 7.0,
                "matches_expected_output": True,
                "reason": "The output is good enough to count as a match.",
            }

        MODULE._predict_generation_response = fake_generate
        MODULE._score_generation_output_against_reference = fake_score

        try:
            comparison = MODULE._evaluate_generation_prompt_completion_model_comparison(
                config,
                tokenizer=None,
                eval_dataset=eval_dataset,
                holdout_metadata={"strategy": "test_holdout"},
            )
        finally:
            MODULE._load_base_model = original_load_base
            MODULE._load_adapter_inference_model = original_load_adapter
            MODULE._predict_generation_response = original_generate
            MODULE._score_generation_output_against_reference = original_score
            MODULE._clear_inference_model = original_clear

        self.assertEqual(len(captured_prompts), 4)
        self.assertEqual(captured_prompts[0], "Solve problem 1")
        self.assertEqual(
            sorted(judgment["reference_answer"] for judgment in captured_judgments),
            ["Gold answer 1", "Gold answer 1", "Gold answer 2", "Gold answer 2"],
        )
        self.assertEqual(comparison["show_evaluation_component"], True)
        self.assertEqual(comparison["summary"]["match_threshold_score"], 7.0)
        self.assertEqual(comparison["summary"]["baseline_match_count"], 1)
        self.assertEqual(comparison["summary"]["candidate_match_count"], 2)
        self.assertEqual(comparison["cases"][0]["prompt"], "Solve problem 1")
        self.assertEqual(comparison["cases"][0]["reference_answer"], "Gold answer 1")
        self.assertEqual(comparison["cases"][0]["baseline_judgment"]["score"], 2.0)
        self.assertTrue(comparison["cases"][0]["candidate_judgment"]["matches_expected_output"])
        self.assertEqual(comparison["holdout"]["strategy"], "test_holdout")


if __name__ == "__main__":
    unittest.main()
