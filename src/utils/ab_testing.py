"""
A/B Testing Pool — config-driven champion/challenger agent variant pools.

Provides:
- ABVariant dataclass for variant definitions
- ABPool for loading/validating config and sampling challengers
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ABVariant:
    """A single agent variant in the A/B testing pool."""

    name: str
    agent_spec: Optional[str] = None          # path to agent spec .md file
    provider: Optional[str] = None            # e.g. "anthropic", "openai"
    model: Optional[str] = None               # e.g. "claude-sonnet-4-20250514"
    num_documents_to_retrieve: Optional[int] = None
    recursion_limit: Optional[int] = None

    def to_meta(self) -> Dict[str, Any]:
        """Serialise variant config for JSONB storage in comparison records."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_meta_json(self) -> str:
        return json.dumps(self.to_meta(), default=str)


class ABPoolError(ValueError):
    """Raised when the ab_testing config is invalid."""
    pass


class ABPool:
    """
    Manages the set of agent variants and champion designation.

    Loaded from the ``ab_testing`` section of config.yaml:

    .. code-block:: yaml

        ab_testing:
          enabled: true
          champion: "production-v2"
          pool:
            - name: "production-v2"
              agent_spec: "agents/cms-comp-ops.md"
              provider: "anthropic"
              model: "claude-sonnet-4-20250514"
            - name: "gpt4o-candidate"
              provider: "openai"
              model: "gpt-4o"
    """

    def __init__(self, variants: List[ABVariant], champion_name: str) -> None:
        if len(variants) < 2:
            raise ABPoolError("ABPool requires at least 2 variants for A/B comparison.")
        if champion_name not in {v.name for v in variants}:
            raise ABPoolError(f"Champion '{champion_name}' not found in variant list.")
        self.variants = variants
        self.champion_name = champion_name
        self._variant_map: Dict[str, ABVariant] = {v.name: v for v in variants}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, ab_config: Dict[str, Any]) -> "ABPool":
        """
        Build an ABPool from the ``ab_testing`` config dict.

        Expected structure::

            ab_testing:
              enabled: true
              pool:
                champion: "variant-name"
                variants:
                  - name: variant-name
                    provider: local
                    model: "qwen3:32b"
                  - name: other-variant
                    model: "gpt-oss:120b"

        Raises ABPoolError on validation failures.
        """
        if not isinstance(ab_config, dict):
            raise ABPoolError("ab_testing config must be a mapping.")

        pool_config = ab_config.get("pool")
        if not pool_config or not isinstance(pool_config, dict):
            raise ABPoolError("ab_testing.pool must be a mapping with 'champion' and 'variants'.")

        champion_name = pool_config.get("champion")
        if not champion_name or not isinstance(champion_name, str):
            raise ABPoolError("ab_testing.pool.champion must be a non-empty string.")

        variant_list = pool_config.get("variants")
        if not variant_list or not isinstance(variant_list, list):
            raise ABPoolError("ab_testing.pool.variants must be a non-empty list of variants.")

        variants: List[ABVariant] = []
        seen_names: set = set()
        for idx, entry in enumerate(variant_list):
            if not isinstance(entry, dict):
                raise ABPoolError(f"ab_testing.pool.variants[{idx}] must be a mapping.")
            name = entry.get("name")
            if not name or not isinstance(name, str):
                raise ABPoolError(f"ab_testing.pool.variants[{idx}] must include a string 'name'.")
            if name in seen_names:
                raise ABPoolError(f"Duplicate variant name '{name}' in ab_testing.pool.variants.")
            seen_names.add(name)

            variants.append(ABVariant(
                name=name,
                agent_spec=entry.get("agent_spec"),
                provider=entry.get("provider"),
                model=entry.get("model"),
                num_documents_to_retrieve=entry.get("num_documents_to_retrieve"),
                recursion_limit=entry.get("recursion_limit"),
            ))

        if champion_name not in seen_names:
            raise ABPoolError(
                f"Champion '{champion_name}' not found in pool. "
                f"Available: {sorted(seen_names)}"
            )

        if len(variants) < 2:
            raise ABPoolError("ab_testing.pool.variants must contain at least 2 variants for A/B comparison.")

        logger.info(
            "Loaded A/B pool: %d variants, champion='%s'",
            len(variants), champion_name,
        )
        return cls(variants=variants, champion_name=champion_name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def champion(self) -> ABVariant:
        return self._variant_map[self.champion_name]

    @property
    def challengers(self) -> List[ABVariant]:
        return [v for v in self.variants if v.name != self.champion_name]

    def get_variant(self, name: str) -> Optional[ABVariant]:
        return self._variant_map.get(name)

    def sample_challenger(self) -> ABVariant:
        """Return a random challenger (any variant that is not the champion)."""
        pool = self.challengers
        if not pool:
            raise ABPoolError("No challengers available (pool has only the champion).")
        return random.choice(pool)

    def sample_matchup(self) -> Tuple[ABVariant, ABVariant, bool]:
        """
        Return (arm_a_variant, arm_b_variant, is_champion_first).

        The champion is always one arm. Position is randomised.
        """
        challenger = self.sample_challenger()
        is_champion_first = random.random() < 0.5
        if is_champion_first:
            return self.champion, challenger, True
        else:
            return challenger, self.champion, False

    def pool_info(self) -> Dict[str, Any]:
        """Return serialisable pool metadata for the /api/ab/pool endpoint."""
        return {
            "enabled": True,
            "champion": self.champion_name,
            "variants": [v.name for v in self.variants],
            "variant_count": len(self.variants),
        }


def load_ab_pool(config: Dict[str, Any]) -> Optional[ABPool]:
    """
    Load the A/B pool from the full config dict.

    Looks for ab_testing config at:
      - config["ab_testing"]  (top-level)
      - config["services"]["ab_testing"]  (under services)

    Returns None if ab_testing is not configured or disabled.
    """
    ab_config = config.get("ab_testing")
    if not ab_config or not isinstance(ab_config, dict):
        # Try under services (the key lives there in deployed configs)
        services = config.get("services") or {}
        ab_config = services.get("ab_testing")
    if not ab_config or not isinstance(ab_config, dict):
        return None
    if not ab_config.get("enabled", False):
        return None
    return ABPool.from_config(ab_config)
