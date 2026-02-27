"""
archi Providers Package

This package provides a unified interface for accessing various LLM providers.
Each provider wraps a specific LLM service (OpenAI, Anthropic, Google, OpenRouter, Local).

Usage:
    from src.archi.providers import get_provider, list_enabled_providers
    
    # Get a specific provider
    provider = get_provider("openai")
    model = provider.get_chat_model("gpt-4o")
    
    # List all enabled providers
    providers = list_enabled_providers()
"""

import os
from typing import Dict, List, Optional, Type

from src.archi.providers.base import (
    BaseProvider,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Provider registry - maps provider type to provider class
_PROVIDER_REGISTRY: Dict[ProviderType, Type[BaseProvider]] = {}

# Cached provider instances
_PROVIDER_INSTANCES: Dict[ProviderType, BaseProvider] = {}

_DEFAULT_API_KEY_ENV_BY_PROVIDER: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GEMINI: "GEMINI_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
}


def _ensure_provider_config_api_key_env(
    provider_type: ProviderType,
    config: Optional[ProviderConfig],
) -> Optional[ProviderConfig]:
    """
    Fill missing api_key_env on custom ProviderConfig.

    Passing a custom config should not disable environment-based API key loading.
    """
    if config is None:
        return None
    if not getattr(config, "api_key_env", None):
        config.api_key_env = _DEFAULT_API_KEY_ENV_BY_PROVIDER.get(provider_type, "")
    return config


def register_provider(provider_type: ProviderType, provider_class: Type[BaseProvider]) -> None:
    """Register a provider class for a provider type."""
    _PROVIDER_REGISTRY[provider_type] = provider_class


def _ensure_providers_registered() -> None:
    """Lazily register all built-in providers."""
    if _PROVIDER_REGISTRY:
        return
    
    # Import and register all providers
    from src.archi.providers.openai_provider import OpenAIProvider
    from src.archi.providers.anthropic_provider import AnthropicProvider
    from src.archi.providers.gemini_provider import GeminiProvider
    from src.archi.providers.openrouter_provider import OpenRouterProvider
    from src.archi.providers.local_provider import LocalProvider
    
    register_provider(ProviderType.OPENAI, OpenAIProvider)
    register_provider(ProviderType.ANTHROPIC, AnthropicProvider)
    register_provider(ProviderType.GEMINI, GeminiProvider)
    register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
    register_provider(ProviderType.LOCAL, LocalProvider)


def get_provider(
    provider_type: str | ProviderType,
    config: Optional[ProviderConfig] = None,
    use_cache: bool = True
) -> BaseProvider:
    """
    Get a provider instance by type.
    
    Args:
        provider_type: The provider type (string or ProviderType enum)
        config: Optional provider configuration. If not provided, uses defaults.
        use_cache: Whether to use cached provider instances. Default True.
    
    Returns:
        A provider instance
    
    Raises:
        ValueError: If the provider type is unknown
    """
    _ensure_providers_registered()
    
    # Convert string to ProviderType if needed
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {[p.value for p in ProviderType]}"
            )
    
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(f"No provider registered for type: {provider_type}")
    
    config = _ensure_provider_config_api_key_env(provider_type, config)

    # Return cached instance if available and no custom config
    if use_cache and config is None and provider_type in _PROVIDER_INSTANCES:
        return _PROVIDER_INSTANCES[provider_type]
    
    # Create new instance
    provider_class = _PROVIDER_REGISTRY[provider_type]
    provider = provider_class(config)
    
    # Cache if using default config
    if use_cache and config is None:
        _PROVIDER_INSTANCES[provider_type] = provider
    
    return provider


def get_provider_by_name(name: str, **kwargs) -> BaseProvider:
    """
    Get a provider instance by display name or type name.
    
    This is a convenience function that accepts common names like:
    - "openai", "OpenAI"
    - "anthropic", "claude", "Anthropic"
    - "gemini", "google", "Gemini"
    - "openrouter", "OpenRouter"
    - "local", "ollama", "Local"
    
    Args:
        name: The provider name
        **kwargs: Additional arguments passed to get_provider
    
    Returns:
        A provider instance
    """
    name_lower = name.lower()
    
    # Map common names to provider types
    name_map = {
        "openai": ProviderType.OPENAI,
        "gpt": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "claude": ProviderType.ANTHROPIC,
        "gemini": ProviderType.GEMINI,
        "google": ProviderType.GEMINI,
        "openrouter": ProviderType.OPENROUTER,
        "local": ProviderType.LOCAL,
        "ollama": ProviderType.LOCAL,
        "vllm": ProviderType.LOCAL,
    }
    
    provider_type = name_map.get(name_lower)
    if provider_type is None:
        # Try direct conversion
        try:
            provider_type = ProviderType(name_lower)
        except ValueError:
            raise ValueError(
                f"Unknown provider name: {name}. "
                f"Try one of: {list(name_map.keys())}"
            )
    
    return get_provider(provider_type, **kwargs)


def list_provider_types() -> List[ProviderType]:
    """List all registered provider types."""
    _ensure_providers_registered()
    return list(_PROVIDER_REGISTRY.keys())


def list_enabled_providers() -> List[BaseProvider]:
    """
    List all providers that have valid API keys or connections configured.
    
    This checks each provider's is_enabled property to determine if it can be used.
    
    Returns:
        List of enabled provider instances
    """
    _ensure_providers_registered()
    
    enabled = []
    for provider_type in _PROVIDER_REGISTRY.keys():
        try:
            provider = get_provider(provider_type)
            if provider.is_enabled:
                enabled.append(provider)
        except Exception as e:
            logger.debug(f"Provider {provider_type} not available: {e}")
    
    return enabled


def list_all_models() -> Dict[str, List[ModelInfo]]:
    """
    List all models from all enabled providers.
    
    Returns:
        Dict mapping provider display name to list of ModelInfo
    """
    result = {}
    for provider in list_enabled_providers():
        try:
            models = provider.list_models()
            result[provider.display_name] = models
        except Exception as e:
            logger.warning(f"Failed to list models from {provider.display_name}: {e}")
    
    return result


def get_model(provider_type: str | ProviderType, model_name: str, provider_config: dict, **kwargs):
    """
    Convenience function to get a chat model directly.
    
    Args:
        provider_type: The provider type
        model_name: The model name
        **kwargs: Additional model configuration
    
    Returns:
        A LangChain chat model instance
    """
    extra_kwargs = {}
    if isinstance(provider_config, dict):
        extra_kwargs.update(provider_config.get("extra_kwargs", {}) or {})
        if "mode" in provider_config and "local_mode" not in extra_kwargs:
            extra_kwargs["local_mode"] = provider_config.get("mode")

    if isinstance(provider_type, str):
        try:
            provider_type_enum = ProviderType(provider_type.lower())
        except ValueError as exc:
            valid_types = ", ".join(t.value for t in ProviderType)
            message = f"Invalid provider type '{provider_type}'. Must be one of: {valid_types}"
            logger.error(message)
            raise ValueError(message) from exc
    else:
        provider_type_enum = provider_type

    config = ProviderConfig(
        provider_type=provider_type_enum,
        api_key_env=_DEFAULT_API_KEY_ENV_BY_PROVIDER.get(provider_type_enum, ""),
        base_url=provider_config.get("base_url", None) if isinstance(provider_config, dict) else None,
        enabled=True,
        models=provider_config.get("models", []) if isinstance(provider_config, dict) else [],
        default_model=provider_config.get("default_model", None) if isinstance(provider_config, dict) else None,
        extra_kwargs=extra_kwargs,
    )
    provider = get_provider(provider_type_enum, config)
    return provider.get_chat_model(model_name, **kwargs)


def clear_provider_cache() -> None:
    """Clear all cached provider instances."""
    _PROVIDER_INSTANCES.clear()


def get_provider_with_api_key(
    provider_type: str | ProviderType,
    api_key: str,
) -> BaseProvider:
    """
    Get a provider instance with a custom API key.
    
    This creates a new provider instance with the specified API key,
    bypassing the cache and environment variable lookup.
    
    Args:
        provider_type: The provider type (string or ProviderType enum)
        api_key: The API key to use for this provider
    
    Returns:
        A provider instance configured with the specified API key
    """
    _ensure_providers_registered()
    
    # Convert string to ProviderType if needed
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {[p.value for p in ProviderType]}"
            )
    
    if provider_type not in _PROVIDER_REGISTRY:
        raise ValueError(f"No provider registered for type: {provider_type}")
    
    # Create new instance with custom config containing the API key
    provider_class = _PROVIDER_REGISTRY[provider_type]
    config = ProviderConfig(
        provider_type=provider_type,
        api_key=api_key,
        enabled=True,
    )
    return provider_class(config)


def get_chat_model_with_api_key(
    provider_type: str | ProviderType,
    model_name: str,
    api_key: str,
    **kwargs
):
    """
    Get a chat model directly with a custom API key.
    
    This is a convenience function that creates a provider with the specified
    API key and immediately returns a chat model.
    
    Args:
        provider_type: The provider type
        model_name: The model name
        api_key: The API key to use
        **kwargs: Additional model configuration
    
    Returns:
        A LangChain chat model instance
    """
    provider = get_provider_with_api_key(provider_type, api_key)
    return provider.get_chat_model(model_name, **kwargs)


# Export public API
__all__ = [
    # Core classes
    "BaseProvider",
    "ModelInfo",
    "ProviderConfig",
    "ProviderType",
    # Factory functions
    "get_provider",
    "get_provider_by_name",
    "get_provider_with_api_key",
    "get_model",
    "get_chat_model_with_api_key",
    # Listing functions
    "list_provider_types",
    "list_enabled_providers",
    "list_all_models",
    # Registry management
    "register_provider",
    "clear_provider_cache",
]
