from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    encoder: str
    checkpoint: Path
    downstream: str
    downstream_checkpoint: Path
    config: str | None = None
    tinyvit_embed_dim: int | None = None


class RegistryError(ValueError):
    pass


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def load_registry(path: str | Path | None = None) -> Mapping[str, Any]:
    registry_path = Path(path) if path else default_registry_path()
    with registry_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), dict):
        raise RegistryError(f"unsupported registry schema: {registry_path}")
    return data


def resolve_model(
    model_id: str,
    *,
    registry_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    downstream_checkpoint: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModelSpec:
    registry = load_registry(registry_path)
    try:
        raw = registry["models"][model_id]
    except KeyError as exc:
        choices = ", ".join(sorted(registry["models"]))
        raise RegistryError(f"unknown model_id {model_id!r}; choose one of: {choices}") from exc

    env = os.environ if environ is None else environ
    selected = Path(checkpoint).expanduser() if checkpoint else None
    if selected is None:
        env_name = raw["checkpoint_env"]
        env_value = env.get(env_name)
        if not env_value:
            filename = raw.get("checkpoint_filename", "checkpoint")
            raise RegistryError(
                f"checkpoint for {model_id} is unset; pass --checkpoint or set "
                f"{env_name} to the exact {filename} path"
            )
        selected = Path(env_value).expanduser()
    selected = selected.resolve()
    if not selected.is_file():
        raise RegistryError(f"checkpoint does not exist: {selected}")

    selected_downstream = selected
    downstream_env = raw.get("downstream_checkpoint_env")
    if downstream_env:
        selected_downstream = (
            Path(downstream_checkpoint).expanduser()
            if downstream_checkpoint
            else Path(env[downstream_env]).expanduser()
            if env.get(downstream_env)
            else None
        )
        if selected_downstream is None:
            raise RegistryError(
                f"downstream checkpoint for {model_id} is unset; pass "
                f"--downstream-checkpoint or set {downstream_env}"
            )
        selected_downstream = selected_downstream.resolve()
        if not selected_downstream.is_file():
            raise RegistryError(f"downstream checkpoint does not exist: {selected_downstream}")

    return ModelSpec(
        model_id=model_id,
        encoder=raw["encoder"],
        checkpoint=selected,
        downstream=raw["downstream"],
        downstream_checkpoint=selected_downstream,
        config=raw.get("config"),
        tinyvit_embed_dim=raw.get("tinyvit_embed_dim"),
    )
