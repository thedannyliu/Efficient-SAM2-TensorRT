import json
import tempfile
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sam2_trt.build import (
    _network_flags,
    _profile_batches,
    _shape_for,
    _validate_builder_options,
    build_bundle,
)
from sam2_trt.manifest import BundleManifest, EngineRecord, sha256_file


class TensorRtNetworkFlagsTest(unittest.TestCase):
    def test_tensorrt_11_uses_strong_typing_without_explicit_batch(self):
        class Flags(IntEnum):
            STRONGLY_TYPED = 0

        fake = SimpleNamespace(NetworkDefinitionCreationFlag=Flags)
        self.assertEqual(_network_flags(fake), 1)

    def test_tensorrt_10_combines_explicit_batch_and_strong_typing(self):
        class Flags(IntEnum):
            EXPLICIT_BATCH = 0
            STRONGLY_TYPED = 1

        fake = SimpleNamespace(NetworkDefinitionCreationFlag=Flags)
        self.assertEqual(_network_flags(fake), 3)

    def test_encoder_dynamic_batch_profile_is_fixed_to_one(self):
        for endpoint in ("min", "opt", "max"):
            self.assertEqual(_shape_for("encoder", "image", 1, endpoint), (1, 3, 1024, 1024))

    def test_track_batch_eight_is_split_across_profiles(self):
        self.assertEqual(_profile_batches("track_step"), (1, 2, 4))
        self.assertEqual(_profile_batches("prompt_point_step"), (1, 2, 4, 8))

    def test_builder_options_reject_invalid_values(self):
        _validate_builder_options(5, 0)
        with self.assertRaises(ValueError):
            _validate_builder_options(6, 0)
        with self.assertRaises(ValueError):
            _validate_builder_options(5, -1)

    def test_bundle_can_reuse_verified_downstream_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "current"
            source = Path(directory) / "source"
            root.mkdir()
            source.mkdir()
            roles = ("prompt_point_step", "prompt_box_step", "track_step")
            records = []
            for role in roles:
                (root / f"{role}.onnx").write_bytes(f"{role}-onnx".encode())
                (source / f"{role}.onnx").write_bytes(f"{role}-onnx".encode())
                engine = source / f"{role}.fp16.engine"
                engine.write_bytes(f"{role}-engine".encode())
                records.append(
                    EngineRecord(
                        role=role,
                        filename=engine.name,
                        sha256=sha256_file(engine),
                        precision="fp16",
                        inputs={},
                        outputs={},
                    )
                )
            (root / "encoder.onnx").write_bytes(b"encoder")
            common = {
                "checkpoint_path": "/checkpoint",
                "checkpoint_sha256": "checkpoint",
                "downstream": "sam2.1-hiera-large",
                "downstream_checkpoint_path": "/downstream",
                "downstream_checkpoint_sha256": "downstream",
                "source_revisions": {},
            }
            BundleManifest(
                model_id="current",
                environment={"export_dtype": "fp16"},
                **common,
            ).write(root / "manifest.json")
            BundleManifest(
                model_id="source",
                engines=records,
                environment={
                    "export_dtype": "fp16",
                    "tensorrt_device_model": "NVIDIA Thor",
                },
                **common,
            ).write(source / "manifest.json")

            def fake_build(_onnx, engine, **_options):
                Path(engine).write_bytes(b"encoder-engine")
                return {"image": [1, 3, 1024, 1024]}, {"image_embedding": [1, 256, 64, 64]}

            with (
                patch("sam2_trt.build.require_thor"),
                patch("sam2_trt.build.device_model", return_value="NVIDIA Thor"),
                patch("sam2_trt.build.build_engine", side_effect=fake_build),
            ):
                build_bundle(
                    root,
                    precision="fp16",
                    reuse_downstream_engines=source,
                )

            result = BundleManifest.read(root / "manifest.json")
            self.assertEqual([record.role for record in result.engines], ["encoder", *roles])
            for role in roles:
                self.assertTrue((root / f"{role}.fp16.engine").samefile(source / f"{role}.fp16.engine"))
            build = json.loads((root / "build.json").read_text())
            self.assertEqual(build["built_engines"], ["encoder.fp16.engine"])
            self.assertEqual(len(build["reused_engines"]), 3)


if __name__ == "__main__":
    unittest.main()
