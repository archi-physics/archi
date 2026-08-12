from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    model: str
    timeout: Optional[float] = None

    def provider_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"temperature": 0}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        descriptor: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.timeout is not None:
            descriptor["timeout"] = self.timeout
        return descriptor


@dataclass(frozen=True)
class EvaluatorProfile:
    version: int
    atoms_extractor: ModelDescriptor
    evaluator: ModelDescriptor

    def components(self) -> Tuple[Tuple[str, ModelDescriptor], ...]:
        return (
            ("atoms_extractor", self.atoms_extractor),
            ("evaluator", self.evaluator),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "qa": {
                component: descriptor.to_dict()
                for component, descriptor in self.components()
            },
        }


DEFAULT_DESCRIPTOR = ModelDescriptor(provider="openai", model="gpt-5.6-terra")
DEFAULT_PROFILE = EvaluatorProfile(
    version=1,
    atoms_extractor=DEFAULT_DESCRIPTOR,
    evaluator=DEFAULT_DESCRIPTOR,
)

PROFILE_FIELDS = {"version", "qa"}
QA_COMPONENTS = {"atoms_extractor", "evaluator"}
DESCRIPTOR_FIELDS = {
    "provider",
    "model",
    "timeout",
}


def _validate_descriptor(value: object, context: str) -> ModelDescriptor:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    unknown = sorted(value.keys() - DESCRIPTOR_FIELDS)
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")

    provider = value.get("provider")
    model = value.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"{context}.provider must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{context}.model must be a non-empty string")

    timeout = value.get("timeout")
    if "timeout" in value and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(f"{context}.timeout must be a positive number")

    return ModelDescriptor(
        provider=provider.strip(),
        model=model.strip(),
        timeout=timeout,
    )


def _validate_profile(value: object) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("evaluator profile must be an object")
    unknown = sorted(value.keys() - PROFILE_FIELDS)
    if unknown:
        raise ValueError(
            f"evaluator profile has unknown field(s): {', '.join(unknown)}"
        )
    if value.get("version") != 1:
        raise ValueError("evaluator profile version must be 1")

    qa = value.get("qa")
    if not isinstance(qa, dict):
        raise ValueError("evaluator profile qa must be an object")
    unknown_qa = sorted(qa.keys() - QA_COMPONENTS)
    missing_qa = sorted(QA_COMPONENTS - qa.keys())
    if unknown_qa:
        raise ValueError(
            f"evaluator profile qa has unknown component(s): {', '.join(unknown_qa)}"
        )
    if missing_qa:
        raise ValueError(
            f"evaluator profile qa is missing component(s): {', '.join(missing_qa)}"
        )
    return qa


def load_profile(profile_path: Optional[Path]) -> EvaluatorProfile:
    if profile_path is None:
        return DEFAULT_PROFILE
    if not profile_path.exists() or not profile_path.is_file():
        raise ValueError(f"evaluator profile must be an existing file: {profile_path}")
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid evaluator profile YAML: {exc}") from exc

    return _parse_profile(raw)


def _parse_profile(value: object) -> EvaluatorProfile:
    qa = _validate_profile(value)
    return EvaluatorProfile(
        version=1,
        atoms_extractor=_validate_descriptor(
            qa["atoms_extractor"], "qa.atoms_extractor"
        ),
        evaluator=_validate_descriptor(qa["evaluator"], "qa.evaluator"),
    )
