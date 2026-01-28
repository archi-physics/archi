"""
Simple YAML configuration loader for A2rchi.

This module provides direct YAML access for:
1. CLI commands that run before PostgreSQL exists
2. Complex nested configuration that isn't in the database yet
3. Bootstrapping ConfigService initialization

For runtime configuration with user preferences and dynamic settings,
use ConfigService instead.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Default config path - can be overridden with A2RCHI_CONFIGS_PATH env var
CONFIGS_PATH = os.environ.get("A2RCHI_CONFIGS_PATH", "/root/A2rchi/configs/")


def _get_configs_path() -> str:
    """Get the configs path, with environment override support."""
    return os.environ.get("A2RCHI_CONFIGS_PATH", CONFIGS_PATH)


def list_config_names() -> List[str]:
    """
    Get available configuration names (without .yaml extension).
    
    Returns:
        List of config names found in CONFIGS_PATH.
    """
    configs_path = _get_configs_path()
    if not os.path.exists(configs_path):
        return []
    return [n.replace('.yaml', '') for n in os.listdir(configs_path) if n.endswith('.yaml')]


def load_yaml_config(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        name: Config name (without .yaml extension). If None, loads the first config found.
        
    Returns:
        Dictionary containing the full configuration.
        
    Raises:
        FileNotFoundError: If no config file is found.
    """
    configs_path = _get_configs_path()
    
    if name is None:
        # Find first YAML file in configs directory
        files = [f for f in os.listdir(configs_path) if f.endswith('.yaml')]
        if not files:
            raise FileNotFoundError(f"No config files found in {configs_path}")
        config_path = os.path.join(configs_path, files[0])
    else:
        config_path = os.path.join(configs_path, f"{name}.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_global_config(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the 'global' section of the config.
    
    Args:
        name: Config name (without .yaml extension).
        
    Returns:
        Dictionary containing global configuration.
    """
    config = load_yaml_config(name)
    return config.get("global", {})


def load_services_config(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the 'services' section of the config.
    
    Args:
        name: Config name (without .yaml extension).
        
    Returns:
        Dictionary containing services configuration.
    """
    config = load_yaml_config(name)
    return config.get("services", {})


def load_data_manager_config(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the 'data_manager' section of the config.
    
    Args:
        name: Config name (without .yaml extension).
        
    Returns:
        Dictionary containing data_manager configuration.
    """
    config = load_yaml_config(name)
    return config.get("data_manager", {})


def load_a2rchi_config(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the 'a2rchi' section of the config.
    
    Args:
        name: Config name (without .yaml extension).
        
    Returns:
        Dictionary containing a2rchi configuration.
    """
    config = load_yaml_config(name)
    return config.get("a2rchi", {})


def get_model_class_map(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get model class mapping from config and resolve to actual classes.
    
    Uses ModelRegistry for class resolution.
    
    Args:
        config: Full configuration dictionary.
        
    Returns:
        Model class map with 'class' values resolved to actual Python classes.
    """
    from src.a2rchi.models.registry import ModelRegistry
    
    model_class_map = config.get("a2rchi", {}).get("model_class_map", {})
    result = {}
    
    for model_name, model_config in model_class_map.items():
        result[model_name] = model_config.copy()
        # Resolve class name to actual class using registry
        model_class = ModelRegistry.get(model_name)
        if model_class:
            result[model_name]["class"] = model_class
            
    return result


def get_embedding_class_map(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get embedding class mapping from config and resolve to actual classes.
    
    Uses EmbeddingRegistry for class resolution.
    
    Args:
        config: Full configuration dictionary.
        
    Returns:
        Embedding class map with 'class' values resolved to actual Python classes.
    """
    from src.a2rchi.models.registry import EmbeddingRegistry
    
    embedding_class_map = config.get("data_manager", {}).get("embedding_class_map", {})
    result = {}
    
    for embedding_name, embedding_config in embedding_class_map.items():
        result[embedding_name] = embedding_config.copy()
        # Resolve class name to actual class using registry
        embedding_class = EmbeddingRegistry.get(embedding_name)
        if embedding_class:
            result[embedding_name]["class"] = embedding_class
            
    return result


def load_config_with_class_mapping(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load config and resolve model/embedding class names to actual classes.
    
    This is the replacement for `load_config(map=True)` from config_loader.
    
    Args:
        name: Config name (without .yaml extension).
        
    Returns:
        Full configuration with class mappings resolved.
    """
    config = load_yaml_config(name)
    
    # Resolve model classes
    if "a2rchi" in config and "model_class_map" in config["a2rchi"]:
        config["a2rchi"]["model_class_map"] = get_model_class_map(config)
        
    # Resolve embedding classes
    if "data_manager" in config and "embedding_class_map" in config["data_manager"]:
        config["data_manager"]["embedding_class_map"] = get_embedding_class_map(config)
        
    return config
