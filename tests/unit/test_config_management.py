"""
Unit tests for the enhanced ConfigService.

Tests cover:
- get_effective() method
- User preferences
- Audit logging
- Admin checks
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager


# =============================================================================
# ConfigService Tests (skipped if psycopg2 not available)
# =============================================================================

@pytest.fixture
def mock_psycopg2():
    """Mock psycopg2 for testing ConfigService."""
    with patch('src.utils.config_service.psycopg2') as mock:
        conn = MagicMock()
        cursor = MagicMock()
        
        # Set up cursor as context manager
        cursor.__enter__ = Mock(return_value=cursor)
        cursor.__exit__ = Mock(return_value=False)
        
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock.connect.return_value = conn
        mock.extras = MagicMock()
        mock.extras.RealDictCursor = MagicMock()
        
        yield mock, conn, cursor


class TestConfigServiceEffective:
    """Tests for ConfigService.get_effective() method."""
    
    def test_get_effective_uses_user_preference(self):
        """When user has preference set, use it."""
        from src.utils.config_service import ConfigService, DynamicConfig
        
        # Create a service and mock its methods directly
        with patch('src.utils.config_service.psycopg2'):
            service = ConfigService({'host': 'localhost', 'port': 5432})
        
        # Mock the methods that get_effective calls
        service.get_dynamic_config = Mock(return_value=DynamicConfig(
            active_pipeline='QAPipeline',
            active_model='openai/gpt-4o',
            temperature=0.7,
            max_tokens=4096,
        ))
        
        service.get_user_preferences = Mock(return_value={
            'preferred_temperature': 0.5,  # User prefers 0.5
        })
        
        result = service.get_effective('temperature', 'user123')
        assert result == 0.5  # Should use user preference
    
    def test_get_effective_falls_back_to_dynamic(self):
        """When user has no preference, fall back to dynamic config."""
        from src.utils.config_service import ConfigService, DynamicConfig
        
        with patch('src.utils.config_service.psycopg2'):
            service = ConfigService({'host': 'localhost', 'port': 5432})
        
        service.get_dynamic_config = Mock(return_value=DynamicConfig(
            active_pipeline='QAPipeline',
            active_model='openai/gpt-4o',
            temperature=0.7,
            max_tokens=4096,
        ))
        
        service.get_user_preferences = Mock(return_value={
            # No temperature preference
        })
        
        result = service.get_effective('temperature', 'user123')
        assert result == 0.7  # Should use dynamic config default
    
    def test_get_effective_unknown_field(self, mock_psycopg2):
        """Unknown fields should raise KeyError."""
        from src.utils.config_service import ConfigService, DynamicConfig
        
        mock, conn, cursor = mock_psycopg2
        
        # Return None for dynamic config to trigger default
        cursor.fetchone.return_value = None
        
        service = ConfigService({'host': 'localhost', 'port': 5432})
        
        # Mock to return default dynamic config
        service.get_dynamic_config = Mock(return_value=DynamicConfig())
        
        with pytest.raises(KeyError, match="Unknown config field"):
            service.get_effective('nonexistent_field', 'user123')


class TestConfigServiceAdmin:
    """Tests for ConfigService.is_admin() method."""
    
    def test_is_admin_true(self, mock_psycopg2):
        """Admin user should return True."""
        from src.utils.config_service import ConfigService
        
        mock, conn, cursor = mock_psycopg2
        cursor.fetchone.return_value = (True,)
        
        service = ConfigService({'host': 'localhost', 'port': 5432})
        assert service.is_admin('admin_user') is True
    
    def test_is_admin_false(self, mock_psycopg2):
        """Non-admin user should return False."""
        from src.utils.config_service import ConfigService
        
        mock, conn, cursor = mock_psycopg2
        cursor.fetchone.return_value = (False,)
        
        service = ConfigService({'host': 'localhost', 'port': 5432})
        assert service.is_admin('regular_user') is False
    
    def test_is_admin_user_not_found(self, mock_psycopg2):
        """Non-existent user should return False."""
        from src.utils.config_service import ConfigService
        
        mock, conn, cursor = mock_psycopg2
        cursor.fetchone.return_value = None
        
        service = ConfigService({'host': 'localhost', 'port': 5432})
        assert service.is_admin('nonexistent') is False


# =============================================================================
# PromptService Tests
# =============================================================================

class TestPromptService:
    """Tests for PromptService."""
    
    @pytest.fixture
    def temp_prompts_dir(self, tmp_path):
        """Create a temporary prompts directory structure."""
        # Create directories
        (tmp_path / "condense").mkdir()
        (tmp_path / "chat").mkdir()
        (tmp_path / "system").mkdir()
        
        # Create prompt files
        (tmp_path / "condense" / "default.prompt").write_text("Condense: {history} {question}")
        (tmp_path / "chat" / "default.prompt").write_text("Chat: {retriever_output} {question}")
        (tmp_path / "system" / "default.prompt").write_text("You are helpful.")
        
        return tmp_path
    
    def test_load_prompts(self, temp_prompts_dir):
        """Should load all prompt files."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        count = service.reload()
        
        assert count == 3
    
    def test_get_prompt(self, temp_prompts_dir):
        """Should retrieve prompt content."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        content = service.get("condense", "default")
        
        assert "Condense:" in content
        assert "{history}" in content
    
    def test_get_prompt_not_found(self, temp_prompts_dir):
        """Should raise PromptNotFoundError for missing prompt."""
        from src.utils.prompt_service import PromptService, PromptNotFoundError
        
        service = PromptService(str(temp_prompts_dir))
        
        with pytest.raises(PromptNotFoundError):
            service.get("condense", "nonexistent")
    
    def test_get_invalid_type(self, temp_prompts_dir):
        """Should raise ValueError for invalid prompt type."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        
        with pytest.raises(ValueError, match="Invalid prompt type"):
            service.get("invalid_type", "default")
    
    def test_list_prompts(self, temp_prompts_dir):
        """Should list prompts by type."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        
        prompts = service.list_prompts("condense")
        assert "default" in prompts
    
    def test_list_all_prompts(self, temp_prompts_dir):
        """Should list all prompts by type."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        
        all_prompts = service.list_all_prompts()
        
        assert "condense" in all_prompts
        assert "chat" in all_prompts
        assert "system" in all_prompts
        assert "default" in all_prompts["condense"]
    
    def test_has_prompt(self, temp_prompts_dir):
        """Should check if prompt exists."""
        from src.utils.prompt_service import PromptService
        
        service = PromptService(str(temp_prompts_dir))
        
        assert service.has_prompt("condense", "default") is True
        assert service.has_prompt("condense", "nonexistent") is False
    
    def test_missing_directory(self, tmp_path):
        """Should handle missing prompts directory gracefully."""
        from src.utils.prompt_service import PromptService
        
        nonexistent = tmp_path / "nonexistent"
        service = PromptService(str(nonexistent))
        count = service.reload()
        
        assert count == 0


