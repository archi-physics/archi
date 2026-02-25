"""Local provider implementation for Ollama and OpenAI-compatible local servers."""

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


class LocalProvider(BaseProvider):
    """
    Provider for local LLM servers (Ollama, vLLM, LM Studio, etc.)
    
    Supports two modes:
    1. Ollama mode (default): Uses ChatOllama from langchain_ollama
    2. OpenAI-compatible mode: Uses ChatOpenAI for vLLM, LM Studio, etc.
    
    The mode is determined by the 'local_mode' setting in extra_kwargs.
    """
    
    provider_type = ProviderType.LOCAL
    display_name = "Local Server"
    
    DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_OPENAI_COMPAT_BASE_URL = "http://localhost:8000/v1"

    @staticmethod
    def _normalize_base_url(url: Optional[str]) -> Optional[str]:
        """Ensure the base URL has a scheme so urllib requests succeed."""
        if not url:
            return url
        if url.startswith(("http://", "https://")):
            return url
        return f"http://{url}"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        import os
        
        # Check for OLLAMA_HOST environment variable (supports Docker/Podman deployments)
        # If set, prefer it over the config value so runners can override host/port
        env_ollama_host = self._normalize_base_url(os.environ.get("OLLAMA_HOST"))
        default_ollama_host = env_ollama_host or self.DEFAULT_OLLAMA_BASE_URL

        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.LOCAL,
                base_url=default_ollama_host,
                models=[],  # dynamic fetch
                default_model="",  # set from first available model if present
                # Default to Ollama mode
                extra_kwargs={"local_mode": "ollama"},
            )
        else:
            # Let env override the config base_url when provided (useful in CI)
            if env_ollama_host:
                config.base_url = env_ollama_host
            elif not config.base_url:
                config.base_url = default_ollama_host
            config.base_url = self._normalize_base_url(config.base_url)
        super().__init__(config)
    
    @property
    def local_mode(self) -> str:
        """Get the local server mode (ollama or openai_compat)."""
        return self.config.extra_kwargs.get("local_mode", "ollama")
    
    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Get a local chat model instance."""
        mode = kwargs.pop("local_mode", self.local_mode)
        
        if mode == "openai_compat":
            return self._get_openai_compat_model(model_name, **kwargs)
        else:
            return self._get_ollama_model(model_name, **kwargs)
    
    def _get_ollama_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Get a ChatOllama instance."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain_ollama is required for Ollama support. "
                "Install it with: pip install langchain-ollama"
            )
        
        model_kwargs = {
            "model": model_name,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }

        # Remove local_mode from kwargs as ChatOllama doesn't accept it
        model_kwargs.pop("local_mode", None)
        
        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url
            
        return ChatOllama(**model_kwargs)
    
    def _get_openai_compat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """Get a ChatOpenAI instance for OpenAI-compatible servers (vLLM, LM Studio)."""
        from langchain_openai import ChatOpenAI
        
        base_url = self.config.base_url or self.DEFAULT_OPENAI_COMPAT_BASE_URL
        
        model_kwargs = {
            "model": model_name,
            "base_url": base_url,
            "streaming": True,
            # Most local servers don't require an API key, but some do
            "api_key": self._api_key or "not-needed",
            **{k: v for k, v in self.config.extra_kwargs.items() if k != "local_mode"},
            **kwargs,
        }
        
        return ChatOpenAI(**model_kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """
        List available local models.

        For Ollama, dynamically queries the Ollama API to get installed models.
        Falls back to configured models if the query fails; returns empty list otherwise.
        """
        if self.local_mode == "ollama":
            # Try to get installed models dynamically
            installed = self._fetch_ollama_models()
            if installed:
                return installed

        # Fall back to configured models
        if self.config.models:
            return self.config.models
        return []
    
    def _fetch_ollama_models(self) -> List[ModelInfo]:
        """
        Fetch installed models from Ollama API and convert to ModelInfo objects.
        
        Returns empty list if fetch fails.
        """
        import json
        import urllib.request
        import urllib.error
        
        try:
            base_url = self.config.base_url or self.DEFAULT_OLLAMA_BASE_URL
            logger.debug(f"[LocalProvider] Fetching Ollama models from {base_url}")
            url = f"{base_url}/api/tags"
            
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    models = []
                    for model_data in data.get("models", []):
                        name = model_data.get("name", "")
                        # Extract parameter size if available
                        details = model_data.get("details", {})
                        param_size = details.get("parameter_size", "")
                        family = details.get("family", "")
                        
                        # Create a user-friendly display name
                        display_name = name
                        if param_size:
                            display_name = f"{name} ({param_size})"
                        
                        # Infer capabilities from model family
                        supports_tools = family.lower() in ["qwen2", "llama", "mistral"]
                        supports_vision = "vision" in name.lower() or "vl" in name.lower()
                        
                        models.append(ModelInfo(
                            id=name,
                            name=name,
                            display_name=display_name,
                            context_window=32768,  # Default, actual varies by model
                            supports_tools=supports_tools,
                            supports_streaming=True,
                            supports_vision=supports_vision,
                            max_output_tokens=8192,
                        ))
                    logger.debug(
                        f"[LocalProvider] Discovered {len(models)} models from Ollama: "
                        f"{[m.id for m in models]}"
                    )
                    return models
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning(f"[LocalProvider] Failed to fetch Ollama models from {self.config.base_url}: {e}")
        
        return []
    
    def validate_connection(self) -> bool:
        """
        Validate connection to the local server.
        
        For Ollama, checks if the server is running by hitting the /api/tags endpoint.
        For OpenAI-compatible, checks the /models endpoint.
        """
        import urllib.request
        import urllib.error
        
        try:
            if self.local_mode == "ollama":
                base_url = self.config.base_url or self.DEFAULT_OLLAMA_BASE_URL
                url = f"{base_url}/api/tags"
            else:
                base_url = self.config.base_url or self.DEFAULT_OPENAI_COMPAT_BASE_URL
                url = f"{base_url}/models"
            
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning(f"Local server connection failed: {e}")
            return False
    
    def list_installed_models(self) -> List[str]:
        """
        Query Ollama for installed models (Ollama mode only).
        
        Returns a list of model names that are currently installed locally.
        
        Deprecated: Use list_models() instead which returns full ModelInfo objects.
        """
        models = self.list_models()
        return [m.id for m in models]
