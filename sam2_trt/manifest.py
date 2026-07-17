from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class EngineRecord:
    role: str
    filename: str
    sha256: str
    precision: str
    inputs: dict[str, list[int | str]]
    outputs: dict[str, list[int | str]]


@dataclass
class BundleManifest:
    model_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    downstream: str
    downstream_checkpoint_path: str
    downstream_checkpoint_sha256: str
    source_revisions: dict[str, str | None]
    engines: list[EngineRecord] = field(default_factory=list)
    accuracy: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        checkpoint: str | Path,
        downstream: str,
        downstream_checkpoint: str | Path | None = None,
        source_revisions: dict[str, str | None],
    ) -> "BundleManifest":
        checkpoint_path = Path(checkpoint).resolve()
        downstream_path = Path(downstream_checkpoint or checkpoint).resolve()
        return cls(
            model_id=model_id,
            checkpoint_path=os.fspath(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            downstream=downstream,
            downstream_checkpoint_path=os.fspath(downstream_path),
            downstream_checkpoint_sha256=sha256_file(downstream_path),
            source_revisions=source_revisions,
            environment={
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            stream.write(payload)
            temporary = Path(stream.name)
        temporary.replace(destination)

    @classmethod
    def read(cls, path: str | Path) -> "BundleManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema {data.get('schema_version')}")
        data["engines"] = [EngineRecord(**record) for record in data.get("engines", [])]
        return cls(**data)

    def verify_files(self, bundle_dir: str | Path) -> list[str]:
        root = Path(bundle_dir)
        errors: list[str] = []
        if not Path(self.checkpoint_path).is_file():
            errors.append("checkpoint is missing")
        elif sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            errors.append("checkpoint SHA256 mismatch")
        if not Path(self.downstream_checkpoint_path).is_file():
            errors.append("downstream checkpoint is missing")
        elif sha256_file(self.downstream_checkpoint_path) != self.downstream_checkpoint_sha256:
            errors.append("downstream checkpoint SHA256 mismatch")
        for engine in self.engines:
            path = root / engine.filename
            if not path.is_file():
                errors.append(f"missing engine: {engine.filename}")
            elif sha256_file(path) != engine.sha256:
                errors.append(f"engine SHA256 mismatch: {engine.filename}")
        return errors
