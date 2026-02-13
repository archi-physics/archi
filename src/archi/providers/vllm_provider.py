"""vLLM provider implementation for OpenAI-compatible vLLM servers."""

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
        """Get a ChatOpenAI instance pointing at the vLLM server."""
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
        """Query the vLLM server's /v1/models endpoint for available models."""
        fetched = self._fetch_vllm_models()
        if fetched:
            return fetched
        if self.config.models:
            return self.config.models
        return []

    def _fetch_vllm_models(self) -> List[ModelInfo]:
        """Fetch models from the vLLM /v1/models endpoint."""
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
        """Check if the vLLM server is reachable by hitting /v1/models."""
        try:
            url = f"{self.config.base_url}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning("[VLLMProvider] Connection failed: %s", e)
            return False
