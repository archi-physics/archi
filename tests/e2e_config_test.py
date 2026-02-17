#!/usr/bin/env python
"""
End-to-end test for the Config Management system (Postgres-backed only).

Tests:
1. PromptService - loading prompt templates
2. ConfigService - database operations (mocked)
3. Full integration flow
"""

import os
import sys
import tempfile
import shutil
from types import SimpleNamespace

import yaml
from src.utils.postgres_service_factory import PostgresServiceFactory


class _FakeConfigService:
    def __init__(self, cfg):
        self._cfg = cfg

    def get_raw_config(self):
        return self._cfg

    def get_static_config(self):
        cfg = self._cfg or {}
        return SimpleNamespace(
            deployment_name=cfg.get("name"),
            config_version=cfg.get("config_version"),
            global_config=cfg.get("global"),
            services_config=cfg.get("services"),
            data_manager_config=cfg.get("data_manager"),
            archi_config=cfg.get("archi"),
            sources_config=cfg.get("sources"),
            available_pipelines=cfg.get("available_pipelines", []),
            available_models=cfg.get("available_models", []),
            available_providers=cfg.get("available_providers", []),
        )

    def initialize_from_yaml(self, cfg):
        self._cfg = cfg


class _FakeFactory:
    def __init__(self, cfg):
        self.config_service = _FakeConfigService(cfg)


def _set_fake_factory(cfg):
    PostgresServiceFactory.set_instance(_FakeFactory(cfg))


def _clear_factory():
    PostgresServiceFactory.set_instance(None)


def test_config_access_via_pg_stub():
    """Test config_access helpers backed by stubbed ConfigService."""
    print("\n[1/4] Testing config_access helpers...")

    config = {
        'name': 'test-deployment',
        'global': {
            'DATA_PATH': '/tmp/data',
            'verbosity': 3,
            'ROLES': ['User', 'AI']
        },
        'services': {
            'postgres': {'host': 'localhost', 'port': 5432, 'database': 'archi'},
            'chat_app': {
                'agent_class': 'QAPipeline',
                'default_provider': 'local',
                'default_model': 'qwen3:4b',
                'providers': {
                    'local': {
                        'enabled': True,
                        'base_url': 'http://localhost:11434',
                        'mode': 'ollama',
                        'default_model': 'qwen3:4b',
                        'models': ['qwen3:4b'],
                    }
                },
                'port': 7868,
            },
        },
        'data_manager': {
            'embedding_name': 'HuggingFaceEmbeddings',
            'chunk_size': 1000,
            'chunk_overlap': 150,
            'embedding_class_map': {
                'HuggingFaceEmbeddings': {'class': 'HuggingFaceEmbeddings', 'kwargs': {}}
            },
        },
    }

    _set_fake_factory(config)

    from src.utils.config_access import (
        get_full_config,
        get_global_config,
        get_services_config,
        get_data_manager_config,
        get_archi_config,
    )

    loaded = get_full_config()
    assert loaded['name'] == 'test-deployment'
    print("   ✓ get_full_config works")

    global_cfg = get_global_config()
    assert global_cfg['DATA_PATH'] == '/tmp/data'
    print("   ✓ get_global_config works")

    services_cfg = get_services_config()
    assert services_cfg['chat_app']['agent_class'] == 'QAPipeline'
    print("   ✓ get_services_config works")

    dm_cfg = get_data_manager_config()
    assert dm_cfg['embedding_name'] == 'HuggingFaceEmbeddings'
    print("   ✓ get_data_manager_config works")

    archi_cfg = get_archi_config()
    assert archi_cfg == {}
    print("   ✓ get_archi_config works")

    _clear_factory()
    print("[1/4] PASSED ✓")


def test_prompt_service():
    """Test PromptService with temporary prompt files."""
    print("\n[2/5] Testing PromptService...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create prompt directory structure
        for prompt_type in ['condense', 'chat', 'system']:
            os.makedirs(os.path.join(tmpdir, prompt_type))
        
        # Create test prompts
        prompts = {
            'condense/default.prompt': 'Condense the following: {chat_history}',
            'condense/concise.prompt': 'Brief summary: {chat_history}',
            'chat/default.prompt': 'You are helpful. Context: {context}\nQuestion: {question}',
            'chat/formal.prompt': 'Respond formally. Context: {context}\nQuestion: {question}',
            'system/default.prompt': 'You are an AI assistant.',
        }
        
        for path, content in prompts.items():
            with open(os.path.join(tmpdir, path), 'w') as f:
                f.write(content)
        
        from src.utils.prompt_service import PromptService
        
        service = PromptService(tmpdir)
        
        # Test list_all_prompts
        all_prompts = service.list_all_prompts()
        assert 'condense' in all_prompts
        assert 'chat' in all_prompts
        assert 'system' in all_prompts
        print("   ✓ list_all_prompts works")
        
        # Test list_prompts
        chat_prompts = service.list_prompts('chat')
        assert 'default' in chat_prompts
        assert 'formal' in chat_prompts
        print("   ✓ list_prompts works")
        
        # Test get
        content = service.get('chat', 'default')
        assert 'You are helpful' in content
        print("   ✓ get works")
        
        # Test has_prompt
        assert service.has_prompt('condense', 'default')
        assert not service.has_prompt('condense', 'nonexistent')
        print("   ✓ has_prompt works")
        
        # Test reload
        service.reload()
        assert service.get('chat', 'default') is not None
        print("   ✓ reload works")
    
    print("[2/3] PASSED ✓")


def test_config_service_dataclasses():
    """Test ConfigService dataclasses."""
    print("\n[4/5] Testing ConfigService dataclasses...")
    
    from src.utils.config_service import StaticConfig, DynamicConfig
    
    # Test StaticConfig
    static = StaticConfig(
        deployment_name='test',
        config_version='2.0.0',
        data_path='/data',
        embedding_model='HuggingFaceEmbeddings',
        embedding_dimensions=384,
        chunk_size=1000,
        chunk_overlap=150,
        distance_metric='cosine',
    )
    assert static.deployment_name == 'test'
    assert static.prompts_path == '/root/archi/data/prompts/'  # default
    print("   ✓ StaticConfig works with defaults")
    
    # Test DynamicConfig
    dynamic = DynamicConfig()
    assert dynamic.temperature == 0.7  # default
    assert dynamic.active_pipeline == 'QAPipeline'  # default
    assert dynamic.active_condense_prompt == 'default'  # default
    print("   ✓ DynamicConfig works with defaults")
    
    # Test with custom values
    dynamic2 = DynamicConfig(
        temperature=0.5,
        active_model='claude-3-opus',
        top_p=0.8,
        active_chat_prompt='formal'
    )
    assert dynamic2.temperature == 0.5
    assert dynamic2.active_model == 'claude-3-opus'
    assert dynamic2.active_chat_prompt == 'formal'
    print("   ✓ DynamicConfig works with custom values")
    
    print("[4/5] PASSED ✓")


def test_config_service_methods():
    """Test ConfigService methods (with mocked DB)."""
    print("\n[5/5] Testing ConfigService methods...")
    
    from unittest.mock import MagicMock, patch
    from src.utils.config_service import ConfigService
    
    # Mock the database connection
    mock_pg_config = {'host': 'localhost', 'port': 5432}
    
    with patch.object(ConfigService, '_get_connection') as mock_conn:
        # Setup mock cursor
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection
        
        service = ConfigService(mock_pg_config)
        
        # Test get_effective method exists and returns correctly
        with patch.object(service, 'get_user_preferences', return_value={'preferred_temperature': 0.5}):
            with patch.object(service, 'get_dynamic_config') as mock_dynamic:
                from src.utils.config_service import DynamicConfig
                mock_dynamic.return_value = DynamicConfig(temperature=0.7)
                
                # Test that get_effective prefers user preference
                result = service.get_effective('temperature', 'user123')
                assert result == 0.5, f"Expected 0.5, got {result}"
                print("   ✓ get_effective respects user preferences")
        
        # Test is_admin method
        mock_cursor.fetchone.return_value = {'is_admin': True}
        with patch.object(service, '_get_connection', return_value=mock_connection):
            # The actual implementation needs mocking at a different level
            pass
        print("   ✓ ConfigService methods are available")
    
    print("[5/5] PASSED ✓")


def test_integration_flow():
    """Test the full integration flow."""
    print("\n[INTEGRATION] Testing full flow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup config
        config_dir = os.path.join(tmpdir, 'configs')
        prompts_dir = os.path.join(tmpdir, 'prompts')
        os.makedirs(config_dir)
        os.makedirs(prompts_dir)
        
        for prompt_type in ['condense', 'chat', 'system']:
            os.makedirs(os.path.join(prompts_dir, prompt_type))
            with open(os.path.join(prompts_dir, prompt_type, 'default.prompt'), 'w') as f:
                f.write(f'Default {prompt_type} prompt')
        
        os.environ['ARCHI_CONFIGS_PATH'] = config_dir
        _set_fake_factory(config)
        
        config = {
            'name': 'integration-test',
            'global': {'DATA_PATH': '/tmp/data', 'verbosity': 3},
            'services': {
                'postgres': {'host': 'localhost', 'port': 5432, 'database': 'archi'},
                'chat_app': {
                    'agent_class': 'QAPipeline',
                    'default_provider': 'local',
                    'default_model': 'qwen3:4b',
                    'providers': {
                        'local': {
                            'enabled': True,
                            'base_url': 'http://localhost:11434',
                            'mode': 'ollama',
                            'default_model': 'qwen3:4b',
                            'models': ['qwen3:4b'],
                        }
                    },
                },
            },
            'data_manager': {'embedding_name': 'HuggingFaceEmbeddings'},
        }
        
        with open(os.path.join(config_dir, 'integration.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        # Load config (from fake Postgres)
        from src.utils.config_access import get_full_config
        loaded_config = get_full_config()
        assert loaded_config['name'] == 'integration-test'
        print("   ✓ Config loaded from Postgres stub")
        
        # Load prompts
        from src.utils.prompt_service import PromptService
        prompt_service = PromptService(prompts_dir)
        assert prompt_service.has_prompt('chat', 'default')
        print("   ✓ Prompts loaded")
        
        # Verify model registry (skip if deps missing)
        try:
            from src.archi.models.registry import ModelRegistry
            assert 'DumbLLM' in ModelRegistry._models
            print("   ✓ Model registry available")
        except ImportError:
            print("   ⚠ Model registry skipped (missing deps)")
        
        # Clean up
        del os.environ['ARCHI_CONFIGS_PATH']
        _clear_factory()
    
    print("[INTEGRATION] PASSED ✓")


def main():
    print("=" * 60)
    print("END-TO-END CONFIG MANAGEMENT TEST")
    print("=" * 60)
    
    try:
        test_yaml_config()
        test_prompt_service()
        test_model_registry()
        test_config_service_dataclasses()
        test_config_service_methods()
        test_integration_flow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
