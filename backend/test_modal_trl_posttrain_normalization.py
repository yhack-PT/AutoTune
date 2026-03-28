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
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def map(self, fn, remove_columns=None, desc=None, batched=False):
        if batched:
            raise NotImplementedError("Batched mapping is not needed for these tests.")
        return FakeDataset([fn(dict(row)) for row in self.rows])


class ModalTrlNormalizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
