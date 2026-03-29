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


if __name__ == "__main__":
    unittest.main()
