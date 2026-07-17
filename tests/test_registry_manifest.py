import json
import tempfile
import unittest
from pathlib import Path

from sam2_trt.manifest import BundleManifest, EngineRecord, sha256_file
from sam2_trt.model_registry import RegistryError, resolve_model


class RegistryManifestTest(unittest.TestCase):
    def test_resolves_exact_checkpoint_and_hashes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            registry = root / "models.yaml"
            registry.write_text(
                "schema_version: 1\nmodels:\n  test:\n    encoder: sam2\n"
                "    config: cfg.yaml\n    checkpoint_env: TEST_CKPT\n    downstream: test\n",
                encoding="utf-8",
            )
            spec = resolve_model("test", registry_path=registry, environ={"TEST_CKPT": str(checkpoint)})
            self.assertEqual(spec.checkpoint, checkpoint)
            self.assertEqual(spec.downstream_checkpoint, checkpoint)

            manifest = BundleManifest.create(
                model_id="test", checkpoint=checkpoint, downstream="test", source_revisions={"sam2": "abc"}
            )
            manifest_path = root / "manifest.json"
            manifest.write(manifest_path)
            loaded = BundleManifest.read(manifest_path)
            self.assertEqual(loaded.checkpoint_sha256, sha256_file(checkpoint))
            self.assertEqual(loaded.verify_files(root), [])

    def test_missing_checkpoint_is_not_guessed(self):
        with self.assertRaises(RegistryError):
            resolve_model("sam2.1-tinyvit-21m", environ={})


if __name__ == "__main__":
    unittest.main()
