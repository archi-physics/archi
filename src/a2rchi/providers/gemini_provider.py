"""Google Gemini provider implementation."""

from typing import Any, Dict, List, Optional

from src.a2rchi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default models available from Google Gemini
DEFAULT_GEMINI_MODELS = [
    ModelInfo(
        id="gemini-2.0-flash",
        name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        context_window=1048576,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="gemini-2.0-flash-thinking",
        name="gemini-2.0-flash-thinking",
        display_name="Gemini 2.0 Flash Thinking",
        context_window=1048576,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="gemini-1.5-pro",
        name="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        context_window=2097152,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="gemini-1.5-flash",
        name="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        context_window=1048576,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=True,
        max_output_tokens=8192,
    ),
]


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models."""
    
    provider_type = ProviderType.GEMINI
    display_name = "Google Gemini"
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.GEMINI,
                api_key_env="GOOGLE_API_KEY",
                models=DEFAULT_GEMINI_MODELS,
                default_model="gemini-2.0-flash",
            )
        super().__init__(config)
    
    def get_chat_model(self, model_name: str, **kwargs):
        """Get a Gemini chat model instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini provider. "
                "Install with: pip install langchain-google-genai"
            )
        
        model_kwargs = {
            "model": model_name,
            "streaming": True,
            **self.config.extra_kwargs,
            **kwargs,
        }
        
        if self._api_key:
            model_kwargs["google_api_key"] = self._api_key
            
        return ChatGoogleGenerativeAI(**model_kwargs)
    
    def list_models(self) -> List[ModelInfo]:
        """List available Gemini models."""
        if self.config.models:
            return self.config.models
        return DEFAULT_GEMINI_MODELS
