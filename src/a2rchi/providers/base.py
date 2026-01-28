"""
Provider abstraction layer for multi-model support.

This module provides a unified interface for working with different LLM providers
(OpenAI, Anthropic, Gemini, OpenRouter, Local servers) in a consistent way.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ProviderType(str, Enum):
    """Enumeration of supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    LOCAL = "local"


@dataclass
class ModelInfo:
    """Information about a specific model offered by a provider."""
    id: str
    name: str
    display_name: str
    context_window: int = 128000
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    max_output_tokens: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "supports_vision": self.supports_vision,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    provider_type: ProviderType
    api_key_env: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    models: List[ModelInfo] = field(default_factory=list)
    default_model: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Each provider implementation wraps a specific LLM service (OpenAI, Anthropic, etc.)
    and provides a consistent interface for:
    - Getting chat models
    - Listing available models
    - Validating API credentials
    """
    
    provider_type: ProviderType
    display_name: str
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._api_key: Optional[str] = None
        self._load_api_key()
    
    def _load_api_key(self) -> None:
        """Load API key from config or environment."""
        if self.config.api_key:
            self._api_key = self.config.api_key
        elif self.config.api_key_env:
            self._api_key = read_secret(self.config.api_key_env)
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key for this provider."""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        """Set the API key for this provider dynamically."""
        self._api_key = value
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key for this provider.
        
        This allows runtime configuration of API keys without requiring
        environment variables.
        
        Args:
            api_key: The API key to use for this provider
        """
        self._api_key = api_key
    
    @property
    def is_configured(self) -> bool:
        """Check if the provider has necessary credentials configured."""
        # Local providers may not need an API key
        if self.provider_type == ProviderType.LOCAL:
            return bool(self.config.base_url)
        return bool(self._api_key)
    
    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled and properly configured."""
        return self.config.enabled and self.is_configured
    
    @abstractmethod
    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get a chat model instance for the specified model.
        
        Args:
            model_name: The name/ID of the model to use
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            A BaseChatModel instance configured for the specified model
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """
        List all available models for this provider.
        
        Returns:
            List of ModelInfo objects describing available models
        """
        pass
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        for model in self.list_models():
            if model.id == model_name or model.name == model_name:
                return model
        return None
    
    def validate_connection(self) -> bool:
        """
        Test the connection to the provider.
        
        Returns:
            True if the provider is accessible and credentials are valid
        """
        if not self.is_configured:
            return False
        try:
            # Try to instantiate a model with a simple test
            default_model = self.config.default_model or (
                self.config.models[0].id if self.config.models else None
            )
            if not default_model:
                return False
            model = self.get_chat_model(default_model)
            return model is not None
        except Exception as e:
            logger.warning(f"Provider {self.display_name} validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize provider info to a dictionary."""
        return {
            "type": self.provider_type.value,
            "name": self.display_name,
            "configured": self.is_configured,
            "enabled": self.config.enabled,
            "models": [m.to_dict() for m in self.list_models()],
            "default_model": self.config.default_model,
        }
