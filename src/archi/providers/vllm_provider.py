"""vLLM provider -- thin client for OpenAI-compatible vLLM servers.

Wraps a locally hosted vLLM instance whose ``/v1`` API is wire-compatible
with OpenAI.  No real API key is required; the placeholder ``"not-needed"``
is sent instead.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.archi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


class VLLMProvider(BaseProvider):
    """
    Provider for vLLM inference servers.

    Communicates with a vLLM server via its OpenAI-compatible API.
    The base URL can be configured via:
      1. VLLM_BASE_URL environment variable (highest priority)
      2. ProviderConfig.base_url
      3. Default: http://localhost:8000/v1
    """

    provider_type = ProviderType.VLLM
    display_name = "vLLM"

    @staticmethod
    def _normalize_base_url(url: Optional[str]) -> Optional[str]:
        """Ensure the base URL has a scheme so urllib requests succeed."""
        if not url:
            return url
        if url.startswith(("http://", "https://")):
            return url
        return f"http://{url}"

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize the vLLM provider.

        Resolves the server URL in priority order: ``VLLM_BASE_URL`` env
        var > ``config.base_url`` > ``DEFAULT_VLLM_BASE_URL``.  Bare
        ``host:port`` URLs are normalised with an ``http://`` scheme.

        Args:
            config: Optional provider configuration.  When *None*, a
                default config targeting ``localhost:8000`` is created.
        """
        env_base_url = self._normalize_base_url(os.environ.get("VLLM_BASE_URL"))

        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.VLLM,
                base_url=env_base_url or DEFAULT_VLLM_BASE_URL,
                api_key="not-needed",
                enabled=True,
            )
        else:
            if env_base_url:
                config.base_url = env_base_url
            elif not config.base_url:
                config.base_url = DEFAULT_VLLM_BASE_URL
            config.base_url = self._normalize_base_url(config.base_url)

        super().__init__(config)

    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Create a ChatOpenAI instance pointed at the vLLM server.

        Args:
            model_name: HuggingFace model ID served by vLLM
                (e.g. ``"Qwen/Qwen3-8B"``).
            **kwargs: Extra arguments forwarded to ChatOpenAI.

        Returns:
            A ChatOpenAI instance configured for the vLLM endpoint.
        """
        from langchain_openai import ChatOpenAI

        model_kwargs = {
            "model": model_name,
            "base_url": self.config.base_url,
            "api_key": self._api_key or "not-needed",
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }

        return ChatOpenAI(**model_kwargs)

    def list_models(self) -> List[ModelInfo]:
        """Return available models, querying the server first.

        Falls back to statically configured models if the server is
        unreachable.

        Returns:
            A list of :class:`ModelInfo` discovered from the server or
            from config, or an empty list if neither yields results.
        """
        fetched = self._fetch_vllm_models()
        if fetched:
            return fetched
        if self.config.models:
            return self.config.models
        return []

    def _fetch_vllm_models(self) -> List[ModelInfo]:
        """Fetch models from the vLLM ``/v1/models`` endpoint.

        Returns:
            A list of :class:`ModelInfo`, or an empty list if the
            server is unreachable or returns an unexpected payload.
        """
        try:
            url = f"{self.config.base_url}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    models = []
                    for model_data in data.get("data", []):
                        model_id = model_data.get("id", "")
                        models.append(ModelInfo(
                            id=model_id,
                            name=model_id,
                            display_name=model_id,
                            supports_tools=True,
                            supports_streaming=True,
                        ))
                    logger.debug(
                        "[VLLMProvider] Discovered %d models: %s",
                        len(models),
                        [m.id for m in models],
                    )
                    return models
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("[VLLMProvider] Failed to fetch models from %s: %s", self.config.base_url, e)

        return []

    def validate_connection(self) -> bool:
        """Check whether the vLLM server is reachable.

        Sends a GET to ``/v1/models`` with a short timeout.

        Returns:
            True if the server responds with HTTP 200, False otherwise.
        """
        try:
            url = f"{self.config.base_url}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning("[VLLMProvider] Connection failed: %s", e)
            return False
