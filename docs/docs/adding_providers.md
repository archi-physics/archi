# Adding a New Provider to Archi

This comprehensive guide walks you through adding a new LLM (Large Language Model) provider to Archi. It's designed for developers who may be new to Python or the Archi codebase.

## Table of Contents

1. [Understanding Providers](#understanding-providers)
2. [The Provider Architecture](#the-provider-architecture)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing and Validation](#testing-and-validation)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

## Understanding Providers

### What is a Provider?

In Archi, a **provider** is an abstraction layer that connects the application to different LLM services. Think of it as a translator that allows Archi to communicate with various AI services (OpenAI, Anthropic, Google, etc.) using a single, consistent interface.

**Why do we need providers?**

Different LLM services have different:
- Authentication methods (API keys, tokens)
- Model naming conventions (e.g., "gpt-4" vs "claude-3")
- API endpoints and request formats
- Capabilities (streaming, vision, tool usage)
- Configuration requirements

Without providers, every part of Archi that uses an LLM would need to know how to interact with each service individually. Providers solve this by providing a **unified interface**.

### How Archi Uses Providers

```
User Request
    ↓
Archi Pipeline (e.g)
    ↓
Provider (e.g., OpenAIProvider)
    ↓
LLM API (e.g., OpenAI GPT-4)
    ↓
Response flows back up
```

## The Provider Architecture

### Core Components

Archi's provider system consists of several Python files and classes:

#### 1. **base.py** - The Foundation

[src/archi/providers/base.py](../../src/archi/providers/base.py) contains the abstract base classes and data structures:

**ProviderType Enum**
```python
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
```

An `Enum` (enumeration) is a special Python class that defines a set of named constants. It's like a menu of options. Here, it lists all available provider types. Using an enum prevents typos—you can't accidentally write "openai" as "opanai" when using `ProviderType.OPENAI`.

**ModelInfo Dataclass**
```python
@dataclass
class ModelInfo:
    id: str
    name: str
    display_name: str
    context_window: int = 128000
```

A `dataclass` is a Python class that's primarily used to store data. The `@dataclass` decorator automatically generates common methods like `__init__()`. `ModelInfo` describes a specific AI model's capabilities:

- `id`: The exact identifier the API uses (e.g., "gpt-4o")
- `name`: Alternative name for the model
- `display_name`: Human-readable name shown to users (e.g., "GPT-4o")
- `context_window`: How many tokens (roughly words) the model can process at once
- `supports_tools`: Can the model call functions/tools? (True/False)
- `supports_streaming`: Can responses stream in real-time? (True/False)
- `supports_vision`: Can it process images? (True/False)
- `max_output_tokens`: Maximum length of generated response

**ProviderConfig Dataclass**
```python
@dataclass
class ProviderConfig:
    provider_type: ProviderType
    api_key_env: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    models: List[ModelInfo] = field(default_factory=list)
    default_model: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
```

This stores configuration for a provider:
- `provider_type`: Which provider this is (from the enum)
- `api_key_env`: Name of the environment variable containing the API key (e.g., "OPENAI_API_KEY")
- `api_key`: Direct API key value (optional, used for BYOK - Bring Your Own Key)
- `base_url`: Custom API endpoint (optional, for self-hosted or proxy services)
- `enabled`: Is this provider turned on?
- `models`: List of available models
- `default_model`: Which model to use if none specified
- `extra_kwargs`: Additional provider-specific settings

**BaseProvider Abstract Class**
```python
class BaseProvider(ABC):
    @abstractmethod
    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        pass
```

An `abstract base class` (ABC) is a template that defines what methods all providers must implement, but doesn't implement them itself. It's like a contract that says "every provider must have these methods."

The `@abstractmethod` decorator marks methods that **must** be implemented by subclasses. If you create a provider and forget to implement `get_chat_model()`, Python will raise an error.

**Why use abstract classes?**
- Enforces consistency—all providers have the same interface
- Documents what methods are required
- Catches errors early (at class definition time, not runtime)

#### 2. **__init__.py** - The Registry

[src/archi/providers/__init__.py](../../src/archi/providers/__init__.py) manages provider registration and access:

**The Registry Pattern**
```python
_PROVIDER_REGISTRY: Dict[ProviderType, Type[BaseProvider]] = {}
```

A registry is a dictionary that maps provider types to their implementation classes. Think of it as a phone book: you look up a name (ProviderType.OPENAI) and get back contact information (the OpenAIProvider class).

**Why use a registry?**
- Central location to find all providers
- Easy to add new providers without modifying existing code
- Supports dynamic provider loading

**The Factory Pattern**
```python
def get_provider(provider_type: str | ProviderType, config: Optional[ProviderConfig] = None) -> BaseProvider:
```

This is a "factory function"—it creates and returns provider objects. Instead of calling `OpenAIProvider()` directly, you call `get_provider("openai")`, and it figures out which class to instantiate.

**Benefits:**
- Centralizes provider creation logic
- Handles caching (reuses the same instance)
- Supports multiple ways to specify providers (string or enum)
- Manages error handling in one place

#### 3. **Individual Provider Files**

Each provider (e.g., [openai_provider.py](../../src/archi/providers/openai_provider.py)) implements the `BaseProvider` interface for a specific service.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Archi Application                         │
│  (Pipelines, Chat Interface, Agents)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Uses
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              Provider Factory (__init__.py)                  │
│  - get_provider()                                            │
│  - Registry of all providers                                 │
│  - Caching and lifecycle management                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Returns instance of
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                BaseProvider (base.py)                        │
│  - Abstract interface                                        │
│  - Common functionality (API key loading, validation)        │
└─────────────┬──────────────────────────────────┬────────────┘
              │                                   │
         Implements                          Implements
              │                                   │
              ↓                                   ↓
┌─────────────────────────┐         ┌─────────────────────────┐
│   OpenAIProvider        │         │  AnthropicProvider      │
│   - OpenAI-specific     │   ...   │  - Anthropic-specific   │
│   - Uses langchain-     │         │  - Uses langchain-      │
│     openai              │         │    anthropic            │
└────────────┬────────────┘         └────────────┬────────────┘
             │                                    │
             │ Creates                            │ Creates
             ↓                                    ↓
┌─────────────────────────┐         ┌─────────────────────────┐
│   LangChain ChatOpenAI  │         │  LangChain Chat         │
│   - Handles API calls   │         │  Anthropic              │
│   - Manages streaming   │         │  - Handles API calls    │
└────────────┬────────────┘         └────────────┬────────────┘
             │                                    │
             │ HTTP Requests                      │ HTTP Requests
             ↓                                    ↓
┌─────────────────────────┐         ┌─────────────────────────┐
│   OpenAI API            │         │   Anthropic API         │
└─────────────────────────┘         └─────────────────────────┘
```

## Prerequisites

Before implementing a new provider, you need:

### 1. **Python Knowledge**

Basic understanding of:
- **Classes and Inheritance**: Providers extend (inherit from) `BaseProvider`
- **Type Hints**: Modern Python uses type annotations like `str`, `Optional[str]`, `List[ModelInfo]`
- **Dataclasses**: Convenient way to create data-holding classes
- **Abstract Classes**: Templates that enforce method implementation
- **Decorators**: Functions that modify other functions (like `@abstractmethod`)
- **Enums**: Type-safe named constants
- **Imports**: How to organize and import code across modules

### 2. **Provider Information**

Research about your target provider:
- **API Documentation**: Read the provider's official API docs
- **Authentication**: How does authentication work? (API key, OAuth, token)
- **Available Models**: What models are offered? What are their capabilities?
- **LangChain Integration**: Does a LangChain package exist? (e.g., `langchain-openai`)
- **Special Requirements**: Any unique configuration needs?

Example: For OpenAI
- API docs: https://platform.openai.com/docs
- Auth: API key in header (`Authorization: Bearer YOUR_KEY`)
- Models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- LangChain: `langchain-openai` package exists
- Special: Standard REST API, supports streaming

### 3. **Development Environment**

- Python 3.7+
- Archi source code cloned locally
- Dependencies installed (`pip install -r requirements/requirements-base.txt`)
- IDE with Python support (VS Code, PyCharm)
- Access to the provider's API (account and API key for testing)

### 4. **Understanding LangChain**

Archi uses [LangChain](https://python.langchain.com/), a framework for building LLM applications. Each provider wraps a LangChain chat model class:

```python
from langchain_openai import ChatOpenAI  # OpenAI integration
from langchain_anthropic import ChatAnthropic  # Anthropic integration
```

These classes handle the low-level API communication, streaming, error handling, and response parsing. Your provider creates and configures these objects.

## Step-by-Step Implementation

Let's implement a fictional provider called "AcmeAI" step by step.

### Step 1: Update the ProviderType Enum

**File**: [src/archi/providers/base.py](../../src/archi/providers/base.py)

**Find this code:**
```python
class ProviderType(str, Enum):
    """Enumeration of supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    VLLM = "vllm"
```

**Add your provider:**
```python
class ProviderType(str, Enum):
    """Enumeration of supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    VLLM = "vllm"
    ACMEAI = "acmeai"  # ← Add this line
```

**Why this works:**
- `ACMEAI` is the constant name (use uppercase by convention)
- `"acmeai"` is the string value (what users will type)
- Inheriting from `str` means `ProviderType.ACMEAI == "acmeai"` is True

**What this enables:**
- Type-safe provider references throughout the codebase
- Prevents typos when referring to your provider
- Auto-completion in IDEs

### Step 2: Create the Provider Implementation File

**File**: Create `src/archi/providers/acmeai_provider.py`

Let's build this file section by section:

#### 2.1: File Header and Imports

```python
"""AcmeAI provider implementation.

This module provides integration with AcmeAI's LLM service.
It wraps the langchain-acmeai package and exposes AcmeAI models
through Archi's unified provider interface.
"""

from typing import List, Optional

# Import the LangChain integration for your provider
# This package handles the actual API communication
from langchain_acmeai import ChatAcmeAI

# Import base classes and utilities from Archi
from src.archi.providers.base import (
    BaseProvider,      # The abstract class we'll extend
    ModelInfo,         # Dataclass to describe model capabilities
    ProviderConfig,    # Configuration for the provider
    ProviderType,      # Enum we updated in Step 1
)
from src.utils.logging import get_logger

# Create a logger for this module
# This allows us to output debug info and errors
logger = get_logger(__name__)
```

**Understanding imports:**
- `typing`: Provides type hints like `List`, `Optional`
- `langchain_acmeai`: The LangChain integration package (you'll need to install this)
- `src.archi.providers.base`: Core provider infrastructure from Archi
- `src.utils.logging`: Archi's logging system

#### 2.2: Define Available Models

```python
# Define the models available from AcmeAI
# This is a constant list that describes each model's capabilities
DEFAULT_ACMEAI_MODELS = [
    ModelInfo(
        # The exact model ID used in API calls
        id="acme-ultra-1",
        # Alternative name (often same as id)
        name="acme-ultra-1",
        # Human-readable name shown in UI
        display_name="Acme Ultra 1",
        # Maximum input tokens the model can handle
        context_window=200000,
        # Can this model use tools/functions?
        supports_tools=True,
        # Can responses stream in real-time?
        supports_streaming=True,
        # Can it process images?
        supports_vision=True,
        # Maximum tokens in generated responses
        max_output_tokens=8192,
    ),
    ModelInfo(
        id="acme-fast-1",
        name="acme-fast-1",
        display_name="Acme Fast 1",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
        supports_vision=False,  # This model doesn't support vision
        max_output_tokens=4096,
    ),
    # Add more models as needed
]
```

**How to find this information:**
- Check the provider's API documentation
- Look for model specifications or pricing pages
- Test models to confirm capabilities
- Contact provider support if unclear

**Why define models upfront:**
- Allows UI to show available options before making API calls
- Documents model capabilities for other developers
- Provides fallback if dynamic model listing fails

#### 2.3: Implement the Provider Class

```python
class AcmeAIProvider(BaseProvider):
    """Provider for AcmeAI models.

    This class wraps the AcmeAI API and provides a consistent
    interface for using AcmeAI models within Archi.

    Example:
        >>> provider = AcmeAIProvider()
        >>> model = provider.get_chat_model("acme-ultra-1")
        >>> response = model.invoke("Hello!")
    """

    # Class attributes - these are shared by all instances
    provider_type = ProviderType.ACMEAI
    display_name = "AcmeAI"

    def __init__(self, config: Optional[ProviderConfig] = None):
        """Initialize the AcmeAI provider.

        Args:
            config: Optional custom configuration. If not provided,
                   uses default configuration with environment-based
                   API key loading.
        """
        # If no config provided, create a default one
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.ACMEAI,
                # Environment variable name for API key
                api_key_env="ACMEAI_API_KEY",
                # List of available models
                models=DEFAULT_ACMEAI_MODELS,
                # Default model if none specified
                default_model="acme-ultra-1",
            )

        # Call the parent class constructor
        # This handles API key loading and basic setup
        super().__init__(config)
```

**Understanding `__init__`:**
- This is the constructor—called when you create a new provider instance
- `Optional[ProviderConfig]` means config can be `None` or a `ProviderConfig` object
- We provide sensible defaults so the provider "just works" most of the time
- `super().__init__(config)` calls the parent class's constructor, which:
  - Stores the config
  - Loads the API key from the environment or config
  - Sets up basic provider state

#### 2.4: Implement get_chat_model()

This is the most important method—it creates a configured LangChain model:

```python
    def get_chat_model(self, model_name: str, **kwargs) -> ChatAcmeAI:
        """Get an AcmeAI chat model instance.

        This method creates a LangChain ChatAcmeAI instance configured
        with the appropriate API key, model name, and settings.

        Args:
            model_name: The ID of the model to use (e.g., "acme-ultra-1")
            **kwargs: Additional arguments to pass to ChatAcmeAI constructor.
                     These can include temperature, top_p, max_tokens, etc.

        Returns:
            A configured ChatAcmeAI instance ready to use.

        Example:
            >>> provider = AcmeAIProvider()
            >>> model = provider.get_chat_model(
            ...     "acme-ultra-1",
            ...     temperature=0.7,
            ...     max_tokens=2048
            ... )
        """
        # Start with the model name and enable streaming
        model_kwargs = {
            "model": model_name,
            "streaming": True,  # Enable real-time response streaming
        }

        # Merge in any extra kwargs from the provider config
        # These are default settings for all models from this provider
        model_kwargs.update(self.config.extra_kwargs)

        # Merge in any kwargs passed to this method
        # These override both defaults and config settings
        model_kwargs.update(kwargs)

        # Add the API key if it's been loaded
        if self._api_key:
            model_kwargs["api_key"] = self._api_key

        # Add custom base URL if configured
        # This is useful for self-hosted instances or proxies
        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url

        # Some providers require certain parameters to be set
        # For example, Anthropic requires max_tokens
        # Check your provider's requirements and add similar logic
        if "max_tokens" not in model_kwargs:
            # Try to get max_tokens from model info
            model_info = self.get_model_info(model_name)
            if model_info and model_info.max_output_tokens:
                model_kwargs["max_tokens"] = model_info.max_output_tokens
            else:
                # Fallback default
                model_kwargs["max_tokens"] = 4096

        # Create and return the LangChain chat model
        return ChatAcmeAI(**model_kwargs)
```

**Understanding `**kwargs`:**
- The `**` operator collects extra keyword arguments into a dictionary
- `def func(model, **kwargs)` called as `func("acme-ultra-1", temperature=0.7, max_tokens=2048)` results in `kwargs = {"temperature": 0.7, "max_tokens": 2048}`
- This allows callers to pass arbitrary extra arguments
- We forward these to the LangChain model constructor

**Understanding the priority order:**
1. Start with defaults (streaming=True)
2. Apply provider config's extra_kwargs
3. Apply method's kwargs (highest priority)
4. Add required parameters (API key, base_url, max_tokens)

**Why this ordering matters:**
- More specific settings should override more general ones
- User's explicit arguments should win
- Required parameters are always set

#### 2.5: Implement list_models()

```python
    def list_models(self) -> List[ModelInfo]:
        """List all available AcmeAI models.

        Returns:
            A list of ModelInfo objects describing each available model.

        Note:
            If custom models were provided in the config, those are
            returned instead of the defaults. This allows users to
            configure custom model lists.
        """
        # If config has custom models, use those
        if self.config.models:
            return self.config.models

        # Otherwise, return the default model list
        return DEFAULT_ACMEAI_MODELS
```

**Why check `self.config.models` first:**
- Users can override the model list in configuration
- Useful for testing or when new models are released
- Provides flexibility without code changes

### Step 3: Register the Provider

**File**: [src/archi/providers/__init__.py](../../src/archi/providers/__init__.py)

**Find the `_ensure_providers_registered()` function:**

```python
def _ensure_providers_registered() -> None:
    """Lazily register all built-in providers.

    This function imports all provider classes and registers them
    with the provider registry. It's called automatically the first
    time someone tries to get a provider.

    Lazy registration means providers are only loaded when needed,
    improving startup time and reducing memory usage.
    """
    # If already registered, don't do it again
    if _PROVIDER_REGISTRY:
        return

    # Import all provider classes
    from src.archi.providers.openai_provider import OpenAIProvider
    from src.archi.providers.anthropic_provider import AnthropicProvider
    from src.archi.providers.gemini_provider import GeminiProvider
    from src.archi.providers.openrouter_provider import OpenRouterProvider
    from src.archi.providers.local_provider import LocalProvider
    from src.archi.providers.vllm_provider import VLLMProvider
    from src.archi.providers.acmeai_provider import AcmeAIProvider  # ← Add this

    # Register each provider with its type
    register_provider(ProviderType.OPENAI, OpenAIProvider)
    register_provider(ProviderType.ANTHROPIC, AnthropicProvider)
    register_provider(ProviderType.GEMINI, GeminiProvider)
    register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
    register_provider(ProviderType.LOCAL, LocalProvider)
    register_provider(ProviderType.VLLM, VLLMProvider)
    register_provider(ProviderType.ACMEAI, AcmeAIProvider)  # ← Add this
```

**Understanding lazy registration:**
- Providers are only imported when first accessed
- Saves time during application startup
- Reduces memory usage if some providers aren't used
- Prevents errors from missing dependencies until they're actually needed

**Why imports are inside the function:**
- In Python, imports at module level execute immediately
- Putting imports inside a function delays them
- This is called "lazy loading"

#### Optional: Add Name Aliases

Find the `get_provider_by_name()` function and add convenient aliases:

```python
def get_provider_by_name(name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by display name or type name.

    This convenience function accepts various ways to refer to a provider:
    - Official provider type: "acmeai"
    - Display name: "AcmeAI"
    - Common abbreviations: "acme"

    Args:
        name: The provider name (case-insensitive)
        **kwargs: Additional arguments passed to get_provider()

    Returns:
        A provider instance

    Raises:
        ValueError: If the provider name is not recognized
    """
    name_lower = name.lower()

    # Map common names to provider types
    name_map = {
        # Existing providers...
        "openai": ProviderType.OPENAI,
        "gpt": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "claude": ProviderType.ANTHROPIC,
        # ... more existing mappings ...
        "vllm": ProviderType.VLLM,

        # Add your provider aliases
        "acmeai": ProviderType.ACMEAI,
        "acme": ProviderType.ACMEAI,
        "acme-ai": ProviderType.ACMEAI,
    }

    # Look up the provider type
    provider_type = name_map.get(name_lower)
    if provider_type is None:
        # Try direct conversion as a last resort
        try:
            provider_type = ProviderType(name_lower)
        except ValueError:
            raise ValueError(
                f"Unknown provider name: {name}. "
                f"Try one of: {list(name_map.keys())}"
            )

    return get_provider(provider_type, **kwargs)
```

**Why add aliases:**
- Users might not know the exact provider type name
- Makes the API more user-friendly
- Allows for common abbreviations and variations

### Step 4: Install Dependencies

Your provider needs the LangChain integration package. This is typically a separate package.

**Check if it exists:**
- Search PyPI: https://pypi.org/search/?q=langchain-acmeai
- Check LangChain integrations: https://python.langchain.com/docs/integrations/chat/
- Look for provider's official Python SDK

**Add to requirements:**

Option 1: Edit `requirements/requirements-base.txt`
```bash
# Add to requirements/requirements-base.txt
langchain-acmeai>=0.1.0
```

Option 2: Edit `pyproject.toml` (if using)
```toml
[project.dependencies]
# ... existing dependencies ...
"langchain-acmeai>=0.1.0"
```

**Install the dependency:**
```bash
pip install langchain-acmeai
```

**If no LangChain integration exists:**

You have two options:

1. **Create your own wrapper** (advanced):
```python
from langchain_core.language_models.chat_models import BaseChatModel

class ChatAcmeAI(BaseChatModel):
    # Implement the BaseChatModel interface
    # This is complex and beyond this guide's scope
```

2. **Request integration** from LangChain team:
   - Open an issue: https://github.com/langchain-ai/langchain/issues
   - Explain your use case
   - LangChain team may create an official integration

### Step 5: Configure Environment Variables

Users need to set their API key. Document this clearly:

**Environment Variable Setup:**

Create or update documentation (e.g., in README or setup guide):

```markdown
## AcmeAI Configuration

To use AcmeAI models, you need an API key from [AcmeAI](https://acmeai.example.com).

### Option 1: Environment Variable

Set the API key in your environment:

**Linux/Mac:**
```bash
export ACMEAI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:ACMEAI_API_KEY = "your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set ACMEAI_API_KEY=your-api-key-here
```

### Option 2: Configuration File

Add to your Archi configuration YAML (e.g., `config.yaml`):

```yaml
providers:
  acmeai:
    api_key: "your-api-key-here"
    enabled: true
    default_model: "acme-ultra-1"
```

### Option 3: Runtime Configuration

Set the API key programmatically:

```python
from src.archi.providers import get_provider

provider = get_provider("acmeai")
provider.set_api_key("your-api-key-here")
```
```

**Security notes to include:**
- Never commit API keys to version control
- Use environment variables or secret management in production
- Rotate keys regularly
- Use least-privilege API keys (read-only when possible)

### Step 6: Test Your Implementation

Testing is critical. Let's write comprehensive tests.

#### 6.1: Unit Tests

**File**: Create or update `tests/unit/test_providers.py`

```python
"""Unit tests for provider system."""

import os
import pytest

from src.archi.providers import (
    get_provider,
    get_provider_by_name,
    list_provider_types,
)
from src.archi.providers.base import ProviderType
from src.archi.providers.acmeai_provider import AcmeAIProvider


class TestAcmeAIProvider:
    """Tests for AcmeAI provider implementation."""

    def test_provider_instantiation(self):
        """Test that AcmeAI provider can be instantiated."""
        # Create provider instance
        provider = get_provider(ProviderType.ACMEAI)

        # Verify it's the right type
        assert isinstance(provider, AcmeAIProvider)
        assert provider.provider_type == ProviderType.ACMEAI
        assert provider.display_name == "AcmeAI"

    def test_provider_by_string(self):
        """Test getting provider by string name."""
        provider = get_provider("acmeai")
        assert isinstance(provider, AcmeAIProvider)

    def test_provider_by_alias(self):
        """Test getting provider by common alias."""
        provider = get_provider_by_name("acme")
        assert isinstance(provider, AcmeAIProvider)

    def test_list_models(self):
        """Test that provider lists available models."""
        provider = get_provider(ProviderType.ACMEAI)
        models = provider.list_models()

        # Should have at least one model
        assert len(models) > 0

        # Each model should have required fields
        for model in models:
            assert hasattr(model, 'id')
            assert hasattr(model, 'name')
            assert hasattr(model, 'display_name')
            assert hasattr(model, 'context_window')
            assert isinstance(model.context_window, int)
            assert model.context_window > 0

    def test_get_model_info(self):
        """Test retrieving specific model information."""
        provider = get_provider(ProviderType.ACMEAI)

        # Get info for a known model
        model_info = provider.get_model_info("acme-ultra-1")

        assert model_info is not None
        assert model_info.id == "acme-ultra-1"
        assert model_info.display_name == "Acme Ultra 1"

    def test_provider_in_registry(self):
        """Test that provider is registered in the global registry."""
        provider_types = list_provider_types()
        assert ProviderType.ACMEAI in provider_types

    def test_provider_configuration(self):
        """Test provider with custom configuration."""
        from src.archi.providers.base import ProviderConfig, ModelInfo

        # Create custom config
        config = ProviderConfig(
            provider_type=ProviderType.ACMEAI,
            api_key="test-key-123",
            default_model="acme-fast-1",
            extra_kwargs={"temperature": 0.5},
        )

        # Create provider with custom config
        provider = AcmeAIProvider(config)

        assert provider.api_key == "test-key-123"
        assert provider.config.default_model == "acme-fast-1"

    @pytest.mark.skipif(
        not os.getenv("ACMEAI_API_KEY"),
        reason="ACMEAI_API_KEY environment variable not set"
    )
    def test_chat_model_creation(self):
        """Test creating a chat model (requires API key)."""
        provider = get_provider(ProviderType.ACMEAI)

        # This test only runs if API key is available
        if not provider.is_configured:
            pytest.skip("Provider not configured with API key")

        # Create a chat model
        model = provider.get_chat_model("acme-ultra-1")

        assert model is not None
        # The model should be a LangChain chat model
        from langchain_core.language_models.chat_models import BaseChatModel
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(
        not os.getenv("ACMEAI_API_KEY"),
        reason="ACMEAI_API_KEY environment variable not set"
    )
    def test_model_inference(self):
        """Test that the model can actually generate responses."""
        provider = get_provider(ProviderType.ACMEAI)

        if not provider.is_configured:
            pytest.skip("Provider not configured with API key")

        # Create model and generate response
        model = provider.get_chat_model("acme-ultra-1")
        response = model.invoke("Say 'Hello, World!' and nothing else.")

        # Check that we got a response
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
        print(f"Model response: {response.content}")
```

**Understanding pytest:**
- `pytest` is Python's most popular testing framework
- Test functions start with `test_`
- `assert` statements check conditions—if False, the test fails
- `@pytest.mark.skipif` conditionally skips tests
- `pytest.skip()` dynamically skips a test

**Running tests:**
```bash
# Run all tests
pytest tests/unit/test_providers.py

# Run specific test class
pytest tests/unit/test_providers.py::TestAcmeAIProvider

# Run specific test
pytest tests/unit/test_providers.py::TestAcmeAIProvider::test_provider_instantiation

# Run with verbose output
pytest tests/unit/test_providers.py -v

# Run with print statements visible
pytest tests/unit/test_providers.py -s
```

#### 6.2: Manual Testing

Create a test script to manually verify your provider:

**File**: Create `scripts/test_acmeai_provider.py`

```python
#!/usr/bin/env python3
"""Manual test script for AcmeAI provider.

This script helps you interactively test the AcmeAI provider integration.
Run it to verify that your provider works correctly.

Usage:
    python scripts/test_acmeai_provider.py
"""

import os
import sys

# Add src to path so we can import Archi modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.archi.providers import get_provider, list_enabled_providers
from src.archi.providers.base import ProviderType


def main():
    """Main test function."""
    print("=" * 70)
    print("AcmeAI Provider Test Script")
    print("=" * 70)

    # Step 1: Check if provider is registered
    print("\n1. Checking provider registration...")
    try:
        provider = get_provider(ProviderType.ACMEAI)
        print(f"   ✓ Provider registered: {provider.display_name}")
    except Exception as e:
        print(f"   ✗ Failed to get provider: {e}")
        return

    # Step 2: Check configuration
    print("\n2. Checking provider configuration...")
    print(f"   - Is configured: {provider.is_configured}")
    print(f"   - Is enabled: {provider.is_enabled}")
    print(f"   - Has API key: {bool(provider.api_key)}")

    if not provider.is_configured:
        print("\n   ⚠ Provider not configured!")
        print("   Set ACMEAI_API_KEY environment variable:")
        print("   export ACMEAI_API_KEY='your-key-here'")
        return

    # Step 3: List models
    print("\n3. Listing available models...")
    models = provider.list_models()
    print(f"   Found {len(models)} models:")
    for model in models:
        print(f"   - {model.display_name} ({model.id})")
        print(f"     Context: {model.context_window:,} tokens")
        print(f"     Tools: {model.supports_tools}, "
              f"Streaming: {model.supports_streaming}, "
              f"Vision: {model.supports_vision}")

    # Step 4: Create a chat model
    print("\n4. Creating chat model...")
    try:
        default_model = provider.config.default_model
        print(f"   Using model: {default_model}")
        model = provider.get_chat_model(default_model)
        print(f"   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test basic inference
    print("\n5. Testing basic inference...")
    test_prompt = "Say 'Hello from AcmeAI!' and nothing else."
    print(f"   Prompt: {test_prompt}")

    try:
        response = model.invoke(test_prompt)
        print(f"   ✓ Response received:")
        print(f"   {response.content}")
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 6: Test streaming
    print("\n6. Testing streaming response...")
    stream_prompt = "Count from 1 to 5, one number per line."
    print(f"   Prompt: {stream_prompt}")

    try:
        print("   Response: ", end="", flush=True)
        for chunk in model.stream(stream_prompt):
            # Print each chunk as it arrives
            print(chunk.content, end="", flush=True)
        print()  # New line after streaming completes
        print("   ✓ Streaming works!")
    except Exception as e:
        print(f"\n   ✗ Streaming failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 7: Check if provider appears in enabled list
    print("\n7. Checking enabled providers list...")
    enabled = list_enabled_providers()
    provider_names = [p.display_name for p in enabled]

    if provider.display_name in provider_names:
        print(f"   ✓ {provider.display_name} is in enabled providers")
    else:
        print(f"   ⚠ {provider.display_name} not in enabled providers")
        print(f"   Enabled: {provider_names}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

**Run the test script:**
```bash
# Set your API key first
export ACMEAI_API_KEY="your-actual-api-key"

# Run the script
python scripts/test_acmeai_provider.py
```

**What to look for:**
- All checks should pass (✓)
- No error messages or tracebacks
- Models listed correctly
- Responses generated successfully
- Streaming works

### Step 7: Update Documentation

Documentation is essential for users to discover and use your provider.

#### 7.1: User Guide

**File**: [docs/docs/user_guide.md](user_guide.md)

Add a section about your provider:

```markdown
### AcmeAI Provider

Archi supports AcmeAI's language models through the `acmeai` provider.

#### Setup

1. Get an API key from [AcmeAI](https://acmeai.example.com)
2. Set your API key:
   ```bash
   export ACMEAI_API_KEY="your-api-key"
   ```

3. Configure in your `config.yaml`:
   ```yaml
   providers:
     acmeai:
       enabled: true
       default_model: "acme-ultra-1"
   ```

#### Available Models

- **Acme Ultra 1** (`acme-ultra-1`): Most capable model, supports vision
- **Acme Fast 1** (`acme-fast-1`): Faster, lower-cost option

#### Usage Example

```python
from src.archi.providers import get_provider

# Get the provider
provider = get_provider("acmeai")

# Create a model
model = provider.get_chat_model("acme-ultra-1")

# Generate a response
response = model.invoke("Explain quantum computing")
print(response.content)
```
```

#### 7.2: API Reference

**File**: [docs/docs/api_reference.md](api_reference.md)

Add your provider to the API reference:

```markdown
## AcmeAI Provider

### Class: `AcmeAIProvider`

Provider implementation for AcmeAI models.

**Location**: `src.archi.providers.acmeai_provider`

#### Methods

##### `get_chat_model(model_name: str, **kwargs) -> ChatAcmeAI`

Creates a configured AcmeAI chat model.

**Parameters:**
- `model_name` (str): Model ID (e.g., "acme-ultra-1")
- `**kwargs`: Additional model configuration
  - `temperature` (float): Sampling temperature (0.0 to 2.0)
  - `max_tokens` (int): Maximum response length
  - `top_p` (float): Nucleus sampling parameter

**Returns:** ChatAcmeAI instance

**Example:**
```python
provider = get_provider("acmeai")
model = provider.get_chat_model(
    "acme-ultra-1",
    temperature=0.7,
    max_tokens=2048
)
```

##### `list_models() -> List[ModelInfo]`

Returns list of available AcmeAI models.

**Returns:** List of ModelInfo objects

---
```

#### 7.3: Update Navigation

**File**: [docs/mkdocs.yml](../mkdocs.yml)

Make sure your new sections are visible in the docs navigation:

```yaml
nav:
  - Home: index.md
  - Install: install.md
  - Quickstart: quickstart.md
  - User Guide: user_guide.md
  - Adding Providers: adding_providers.md  # ← This guide
  - Advanced Setup: advanced_setup_deploy.md
  - API Reference: api_reference.md
  - Developer Guide: developer_guide.md
```

## Testing and Validation

### Comprehensive Testing Checklist

Before considering your provider complete, test these scenarios:

#### ✅ Basic Functionality
- [ ] Provider instantiates without errors
- [ ] Provider appears in registry
- [ ] `list_models()` returns correct models
- [ ] `get_model_info()` returns correct details
- [ ] Can create chat model instance
- [ ] Model generates responses

#### ✅ Configuration
- [ ] Works with environment variable API key
- [ ] Works with config file API key
- [ ] Works with runtime API key (`set_api_key()`)
- [ ] Respects `enabled: false` setting
- [ ] Custom `base_url` works (if applicable)
- [ ] `extra_kwargs` are passed through

#### ✅ Error Handling
- [ ] Graceful failure without API key
- [ ] Clear error message for invalid API key
- [ ] Clear error message for invalid model name
- [ ] Handles network errors gracefully
- [ ] Handles API rate limits appropriately

#### ✅ Advanced Features
- [ ] Streaming responses work
- [ ] Tool/function calling works (if supported)
- [ ] Vision inputs work (if supported)
- [ ] Token counting/limits respected
- [ ] Concurrent requests work

#### ✅ Integration
- [ ] Works in QA pipeline
- [ ] Works in chat interface
- [ ] Works with BYOK (Bring Your Own Key)
- [ ] Appears in provider listing APIs
- [ ] Can be set as default provider

### Integration Testing Example

Test your provider in an actual Archi pipeline:

```python
"""Integration test: Use AcmeAI provider in a pipeline."""

from src.archi.pipelines.agents.cms_comp_ops_agent import CMSCompOpsAgent
from src.archi.providers import get_provider


def test_acmeai_in_pipeline():
    """Test AcmeAI provider in a pipeline."""
    # Get provider and model
    provider = get_provider("acmeai")
    model = provider.get_chat_model("acme-ultra-1")

    # Create pipeline with AcmeAI model
    pipeline = CMSCompOpsAgent(model=model)

    # Ask a question
    result = pipeline.run("What is the capital of France?")

    # Verify response
    assert "Paris" in result.answer
    print(f"✓ Pipeline test passed: {result.answer}")


if __name__ == "__main__":
    test_acmeai_in_pipeline()
```

## Advanced Topics

### Dynamic Model Discovery

Instead of hardcoding models, query the provider's API:

```python
def list_models(self) -> List[ModelInfo]:
    """List models by querying AcmeAI API."""
    # If custom models configured, use those
    if self.config.models:
        return self.config.models

    try:
        # Query API for available models
        models_data = self._fetch_models_from_api()
        return [self._parse_model_info(m) for m in models_data]
    except Exception as e:
        logger.warning(f"Failed to fetch models from API: {e}")
        # Fall back to defaults
        return DEFAULT_ACMEAI_MODELS

def _fetch_models_from_api(self):
    """Fetch model list from AcmeAI API."""
    import requests

    response = requests.get(
        "https://api.acmeai.example.com/v1/models",
        headers={"Authorization": f"Bearer {self.api_key}"}
    )
    response.raise_for_status()
    return response.json()["models"]

def _parse_model_info(self, model_data: dict) -> ModelInfo:
    """Convert API response to ModelInfo."""
    return ModelInfo(
        id=model_data["id"],
        name=model_data["name"],
        display_name=model_data["display_name"],
        context_window=model_data.get("context_length", 128000),
        supports_tools=model_data.get("supports_functions", False),
        supports_streaming=True,  # Assume true
        supports_vision=model_data.get("supports_vision", False),
        max_output_tokens=model_data.get("max_output_tokens"),
    )
```

**Pros:**
- Always up-to-date with latest models
- No code changes needed for new models

**Cons:**
- Adds network latency
- Fails if API is down
- Need error handling and fallback

### Custom Validation

Override `validate_connection()` for provider-specific checks:

```python
def validate_connection(self) -> bool:
    """Validate connection to AcmeAI."""
    if not self.is_configured:
        return False

    try:
        # Make a simple API call to verify credentials
        import requests

        response = requests.get(
            "https://api.acmeai.example.com/v1/user",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=5
        )

        # Check for successful response
        if response.status_code == 200:
            logger.info("AcmeAI connection validated successfully")
            return True
        elif response.status_code == 401:
            logger.error("AcmeAI API key is invalid")
            return False
        else:
            logger.warning(f"AcmeAI validation returned {response.status_code}")
            return False

    except Exception as e:
        logger.warning(f"AcmeAI validation failed: {e}")
        return False
```

### Usage Tracking

Track API usage and costs:

```python
from src.archi.providers.base import BaseProvider
from typing import Optional
import time


class AcmeAIProvider(BaseProvider):
    """AcmeAI provider with usage tracking."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        super().__init__(config)
        self._call_count = 0
        self._total_tokens = 0

    def get_chat_model(self, model_name: str, **kwargs):
        """Get chat model with usage tracking."""
        model = super().get_chat_model(model_name, **kwargs)

        # Wrap model to track usage
        return self._wrap_model_with_tracking(model)

    def _wrap_model_with_tracking(self, model):
        """Wrap model to track API calls and token usage."""
        original_invoke = model.invoke

        def tracked_invoke(*args, **kwargs):
            self._call_count += 1
            start = time.time()

            result = original_invoke(*args, **kwargs)

            # Track tokens if available
            if hasattr(result, 'response_metadata'):
                usage = result.response_metadata.get('usage', {})
                self._total_tokens += usage.get('total_tokens', 0)

            duration = time.time() - start
            logger.info(
                f"AcmeAI call #{self._call_count}: "
                f"{duration:.2f}s, {self._total_tokens} total tokens"
            )

            return result

        model.invoke = tracked_invoke
        return model

    def get_usage_stats(self):
        """Get usage statistics."""
        return {
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
        }
```

### Multi-Region Support

Support different API endpoints by region:

```python
ACMEAI_REGIONS = {
    "us": "https://api.us.acmeai.example.com",
    "eu": "https://api.eu.acmeai.example.com",
    "asia": "https://api.asia.acmeai.example.com",
}

class AcmeAIProvider(BaseProvider):
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            # Default to US region
            region = os.getenv("ACMEAI_REGION", "us")
            base_url = ACMEAI_REGIONS.get(region, ACMEAI_REGIONS["us"])

            config = ProviderConfig(
                provider_type=ProviderType.ACMEAI,
                api_key_env="ACMEAI_API_KEY",
                base_url=base_url,
                models=DEFAULT_ACMEAI_MODELS,
                default_model="acme-ultra-1",
            )

        super().__init__(config)
```

## Troubleshooting

### Common Issues and Solutions

#### Import Error: "No module named 'langchain_acmeai'"

**Problem:** The LangChain integration package isn't installed.

**Solution:**
```bash
pip install langchain-acmeai
```

If the package doesn't exist:
- Check if there's an official LangChain integration
- You may need to use a different package name
- Consider creating a custom wrapper

#### Provider Not Found in Registry

**Problem:** `get_provider("acmeai")` raises ValueError: "Unknown provider type"

**Possible causes:**
1. Forgot to add to `ProviderType` enum
2. Didn't register in `_ensure_providers_registered()`
3. Typo in provider type name

**Solution:**
- Check Step 1: Verify enum entry exists
- Check Step 3: Verify registration call exists
- Check spelling matches exactly

#### API Key Not Loading

**Problem:** `provider.is_configured` returns `False`

**Check:**
```python
import os
print(os.getenv("ACMEAI_API_KEY"))  # Should print your key
```

**Solutions:**
- Verify environment variable is set
- Restart your Python session after setting the variable
- Check for typos in variable name
- Make sure you're setting it in the right shell/terminal

#### Authentication Failures

**Problem:** API returns 401 Unauthorized

**Check:**
- Is the API key correct?
- Does the key have necessary permissions?
- Is the key format correct (with/without "Bearer" prefix)?
- Has the key expired?

**Debug:**
```python
provider = get_provider("acmeai")
print(f"API Key set: {bool(provider.api_key)}")
print(f"Key starts with: {provider.api_key[:10] if provider.api_key else 'None'}")
```

#### Model Not Found

**Problem:** "Model 'acme-ultra-1' not found"

**Solutions:**
- Verify model ID matches provider's documentation exactly
- Check if model requires special access or API tier
- Confirm model is available in your region
- Try listing models: `provider.list_models()`

#### Streaming Not Working

**Problem:** Responses don't stream, arrive all at once

**Check:**
```python
model = provider.get_chat_model("acme-ultra-1")

# Test streaming explicitly
for chunk in model.stream("Hello"):
    print(chunk.content, end="", flush=True)
```

**Possible causes:**
- Model doesn't support streaming
- LangChain integration doesn't implement streaming
- Network/proxy buffering responses

#### Response Format Issues

**Problem:** Responses are malformed or missing fields

**Debug:**
```python
response = model.invoke("Test")
print(type(response))
print(dir(response))
print(response.__dict__)
```

**Solution:**
- Check LangChain integration's response parsing
- Verify API responses match expected format
- May need to customize response handling

### Getting Help

If you're stuck:

1. **Check existing providers:**
   - Look at [openai_provider.py](../../src/archi/providers/openai_provider.py) for reference
   - Compare your implementation to working examples

2. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Test the LangChain integration directly:**
   ```python
   from langchain_acmeai import ChatAcmeAI

   model = ChatAcmeAI(api_key="your-key", model="acme-ultra-1")
   response = model.invoke("Test")
   print(response)
   ```

4. **Ask for help:**
   - Open an issue on Archi's GitHub
   - Include error messages, stack traces, and what you've tried
   - Describe your environment (Python version, OS, package versions)

## Contributing Your Provider

Ready to contribute your provider to Archi?

### Pre-Submission Checklist

- [ ] All tests pass
- [ ] Code follows existing style patterns
- [ ] Documentation is complete and accurate
- [ ] No hardcoded secrets or API keys in code
- [ ] Provider works with BYOK (Bring Your Own Key)
- [ ] Error messages are clear and helpful
- [ ] Logging is appropriate (not too verbose, not too quiet)
- [ ] Type hints are correct and complete
- [ ] No unnecessary dependencies added

### Submission Process

1. **Fork the repository:**
   ```bash
   git clone https://github.com/archi-physics/archi.git
   cd archi
   git checkout -b add-acmeai-provider
   ```

2. **Implement your provider:**
   - Follow all steps in this guide
   - Ensure all tests pass: `pytest tests/unit/test_providers.py`

3. **Commit your changes:**
   ```bash
   git add src/archi/providers/acmeai_provider.py
   git add src/archi/providers/__init__.py
   git add src/archi/providers/base.py
   git add tests/unit/test_providers.py
   git add docs/docs/
   git commit -m "feat: add AcmeAI provider support"
   ```

4. **Push and create PR:**
   ```bash
   git push origin add-acmeai-provider
   ```
   Then open a pull request on GitHub.

5. **PR Description Template:**
   ```markdown
   ## Description
   Adds support for AcmeAI language models through a new provider.

   ## Changes
   - Added AcmeAIProvider class
   - Updated ProviderType enum
   - Registered provider in __init__.py
   - Added comprehensive tests
   - Updated documentation

   ## Testing
   - [x] Unit tests pass
   - [x] Integration tests pass
   - [x] Manual testing completed
   - [x] Tested with actual API

   ## Documentation
   - [x] User guide updated
   - [x] API reference updated
   - [x] Setup instructions provided
   - [x] Examples included

   ## Dependencies
   - Added: langchain-acmeai>=0.1.0

   ## Notes
   AcmeAI provides powerful language models with vision capabilities.
   This integration makes them available through Archi's unified provider interface.
   ```

### Code Review

Expect reviewers to check:
- Code quality and consistency
- Test coverage
- Documentation completeness
- Error handling
- Security (no exposed secrets)
- Performance implications

Be responsive to feedback and ready to make changes.

## Conclusion

You've learned how to add a new LLM provider to Archi! This comprehensive guide covered:

- Understanding the provider architecture
- Implementing the provider interface
- Registering and configuring the provider
- Writing comprehensive tests
- Documenting your work
- Troubleshooting common issues
- Contributing back to the project

### Key Takeaways

1. **Providers abstract differences** between LLM services
2. **Inheritance and interfaces** ensure consistency
3. **Configuration flexibility** supports multiple use cases
4. **Testing is essential** for reliability
5. **Documentation helps users** adopt your provider

### Next Steps

- Implement your provider following this guide
- Test thoroughly in different scenarios
- Get feedback from other users
- Consider adding advanced features
- Contribute back to help others

## Additional Resources

- **Archi Documentation**: https://archi-physics.github.io/archi/
- **LangChain Docs**: https://python.langchain.com/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **pytest Documentation**: https://docs.pytest.org/
- **Archi GitHub**: https://github.com/archi-physics/archi

---

**Questions or feedback on this guide?**
Open an issue on GitHub or reach out to the Archi development team.

Happy coding!
