import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class _DummyImageChain:
    def uv_pip_install(self, *args, **kwargs):
        return self

    def run_commands(self, *args, **kwargs):
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

    def function(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


def _dummy_decorator(*args, **kwargs):
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
        method=_dummy_decorator,
        concurrent=_dummy_decorator,
        web_server=_dummy_decorator,
    ),
)

MODULE_PATH = Path(__file__).with_name("modal_vllm_serve.py")
MODULE_SPEC = importlib.util.spec_from_file_location("modal_vllm_serve_under_test", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


class ModalVllmServeTests(unittest.TestCase):
    def test_default_package_specs_are_vllm_compatible(self):
        self.assertEqual(MODULE.VLLM_PACKAGE_SPEC, "vllm==0.18.0")
        self.assertEqual(MODULE.SERVE_TRANSFORMERS_SPEC, "transformers>=4.56.0,<5")
        self.assertEqual(MODULE.BASE_MODEL, "Qwen/Qwen3-8B-Base")

    def test_build_vllm_cmd_for_merged_model_uses_merged_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoints_dir = Path(tempdir)
            merged_dir = checkpoints_dir / "experiments" / "demo-run" / "merged"
            merged_dir.mkdir(parents=True)

            env = {
                "BASE_MODEL": "Qwen/Qwen3-8B-Base",
                "ADAPTER_PATH": "experiments/demo-run/merged",
                "ADAPTER_NAME": "",
                "MAX_MODEL_LEN": "1024",
                "MERGED": "1",
                "ENABLE_THINKING": "0",
            }

            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(MODULE, "CHECKPOINTS_DIR", checkpoints_dir),
            ):
                cmd = MODULE._build_vllm_cmd()
                env = MODULE._build_vllm_env()

        self.assertIn("--model", cmd)
        self.assertIn(str(merged_dir), cmd)
        self.assertIn("--served-model-name", cmd)
        self.assertIn("demo-run", cmd)
        self.assertNotIn("--enable-lora", cmd)
        self.assertNotIn("VLLM_PACKAGE_SPEC", env)

    def test_build_vllm_cmd_for_qwen3_adapter_model_enables_standard_lora_serving(self):
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoints_dir = Path(tempdir)
            adapter_dir = checkpoints_dir / "experiments" / "demo-run" / "final_adapter"
            adapter_dir.mkdir(parents=True)

            env = {
                "BASE_MODEL": "Qwen/Qwen3-8B-Base",
                "ADAPTER_PATH": "experiments/demo-run/final_adapter",
                "ADAPTER_NAME": "",
                "MAX_MODEL_LEN": "2048",
                "MERGED": "0",
                "ENABLE_THINKING": "0",
            }

            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(MODULE, "CHECKPOINTS_DIR", checkpoints_dir),
            ):
                cmd = MODULE._build_vllm_cmd()
                env = MODULE._build_vllm_env()

        self.assertIn("--model", cmd)
        self.assertIn("Qwen/Qwen3-8B-Base", cmd)
        self.assertIn("--enable-lora", cmd)
        self.assertIn("--lora-modules", cmd)
        self.assertIn(f"demo-run={adapter_dir}", cmd)
        self.assertNotIn("--language-model-only", cmd)
        self.assertNotIn("VLLM_PACKAGE_SPEC", env)


if __name__ == "__main__":
    unittest.main()
