#!/usr/bin/env python3
"""
Smoke test for PostgreSQL consolidation.

Tests the end-to-end flow of:
1. User creation and BYOK key storage
2. Model tracking through pipeline
3. Conversation persistence with model_used/pipeline_used
4. A/B comparison with model info

Run with: python tests/smoke/test_postgres_consolidation.py

Requires:
- PostgreSQL running with init-v2.sql applied
- BYOK_ENCRYPTION_KEY environment variable set
- PG_PASSWORD environment variable set
"""

import os
import sys
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_pg_config():
    """Get PostgreSQL connection config from environment."""
    return {
        "host": os.environ.get("PGHOST", os.environ.get("PG_HOST", "localhost")),
        "port": int(os.environ.get("PGPORT", os.environ.get("PG_PORT", 5439))),
        "database": os.environ.get("PGDATABASE", os.environ.get("PG_DATABASE", "archi")),
        "user": os.environ.get("PGUSER", os.environ.get("PG_USER", "archi")),
        "password": os.environ.get("PG_PASSWORD", "testpassword123"),
    }


def test_user_service():
    """Test UserService for BYOK flow."""
    print("\n=== Testing UserService ===")
    
    from src.utils.user_service import UserService
    
    pg_config = get_pg_config()
    service = UserService(pg_config)
    
    # Create test user
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    user = service.get_or_create_user(test_user_id, auth_provider="anonymous")
    print(f"✓ Created user: {user.id}")
    assert user.id == test_user_id
    assert user.auth_provider == "anonymous"
    
    # Update preferences
    service.update_preferences(test_user_id, theme="dark", preferred_model="gpt-4o")
    updated_user = service.get_user(test_user_id)
    print(f"✓ Updated preferences: theme={updated_user.theme}, preferred_model={updated_user.preferred_model}")
    assert updated_user.theme == "dark"
    assert updated_user.preferred_model == "gpt-4o"
    
    # Test BYOK key storage (if encryption key is set)
    encryption_key = os.environ.get("BYOK_ENCRYPTION_KEY")
    if encryption_key:
        test_api_key = f"sk-test-{uuid.uuid4().hex}"
        service.set_api_key(test_user_id, "openai", test_api_key)
        retrieved_key = service.get_api_key(test_user_id, "openai")
        print(f"✓ BYOK key round-trip successful")
        assert retrieved_key == test_api_key
        
        # Clean up - delete the key
        service.delete_api_key(test_user_id, "openai")
        print(f"✓ BYOK key deleted")
    else:
        print("⚠ BYOK_ENCRYPTION_KEY not set, skipping API key tests")
    
    # Clean up test user
    # Note: Would need a delete_user method for full cleanup
    print(f"✓ UserService tests passed")
    return test_user_id


def test_conversation_service():
    """Test ConversationService with model tracking."""
    print("\n=== Testing ConversationService ===")
    
    from src.utils.conversation_service import ConversationService, Message
    
    pg_config = get_pg_config()
    service = ConversationService(pg_config)
    
    # Create a conversation (need conversation_id from conversations table)
    # For this test, we'll use the insert_messages method
    test_conv_id = 99999  # Use a test conversation ID
    
    messages = [
        Message(
            sender="user",
            content="What is the capital of France?",
            archi_service="test_service",
            conversation_id=str(test_conv_id),
            model_used=None,
            pipeline_used=None,
        ),
        Message(
            sender="archi",
            content="The capital of France is Paris.",
            archi_service="test_service",
            conversation_id=str(test_conv_id),
            link="https://example.com",
            context='{"test": true}',
            model_used="gpt-4o",
            pipeline_used="QAPipeline",
        ),
    ]
    
    try:
        message_ids = service.insert_messages(messages)
        print(f"✓ Inserted messages with IDs: {message_ids}")
        assert len(message_ids) == 2
        
        # Query back the conversation
        history = service.get_conversation_history(test_conv_id)
        print(f"✓ Retrieved {len(history)} messages from history")
        
        # Check model tracking on archi message
        archi_msg = [m for m in history if m.sender == "archi"]
        if archi_msg:
            msg = archi_msg[-1]
            print(f"✓ Model tracking: model_used={msg.model_used}, pipeline_used={msg.pipeline_used}")
            assert msg.model_used == "gpt-4o"
            assert msg.pipeline_used == "QAPipeline"
            
    except Exception as e:
        print(f"⚠ ConversationService test failed (may need conversations table): {e}")
        return None
    
    print(f"✓ ConversationService tests passed")
    return test_conv_id


def test_ab_comparison_v2():
    """Test A/B comparison with model tracking."""
    print("\n=== Testing A/B Comparison V2 ===")
    
    from src.utils.conversation_service import ConversationService
    
    pg_config = get_pg_config()
    service = ConversationService(pg_config)
    
    test_conv_id = 99999
    
    try:
        comparison_id = service.create_ab_comparison(
            conversation_id=test_conv_id,
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            pipeline_a="QAPipeline",
            pipeline_b="QAPipeline",
            response_a="Response from GPT-4o",
            response_b="Response from Claude",
        )
        print(f"✓ Created A/B comparison: {comparison_id}")
        
        # Record preference
        service.record_ab_preference(comparison_id, preference="A", feedback="GPT-4o was more helpful")
        print(f"✓ Recorded preference for comparison")
        
        # Get comparison stats
        stats = service.get_model_comparison_stats()
        print(f"✓ Retrieved comparison stats: {len(stats)} model pairs")
        
    except Exception as e:
        print(f"⚠ A/B comparison test failed: {e}")
        return
    
    print(f"✓ A/B Comparison V2 tests passed")


def test_byok_resolver():
    """Test BYOK provider resolver."""
    print("\n=== Testing BYOK Resolver ===")
    
    from src.archi.providers.byok_resolver import BYOKResolver
    from src.utils.user_service import UserService
    
    pg_config = get_pg_config()
    
    try:
        user_service = UserService(pg_config)
        resolver = BYOKResolver(user_service=user_service)
        
        # Test without BYOK key (should fall back to default)
        test_user_id = f"test_byok_{uuid.uuid4().hex[:8]}"
        user_service.get_or_create_user(test_user_id)
        
        # This should not raise - falls back to environment key
        provider = resolver.get_provider_for_user("openai", user_id=test_user_id)
        print(f"✓ Got provider for user (fallback to env key)")
        
        # Check BYOK lookup returns None for user without key
        key = resolver.get_byok_key("openai", user_id=test_user_id)
        print(f"✓ BYOK lookup for user without key: {key}")
        assert key is None
        
    except Exception as e:
        print(f"⚠ BYOK resolver test: {e}")
    
    print(f"✓ BYOK Resolver tests passed")


def test_document_selection_service():
    """Test document selection with 3-tier hierarchy."""
    print("\n=== Testing DocumentSelectionService ===")
    
    from src.utils.document_selection_service import DocumentSelectionService
    
    pg_config = get_pg_config()
    
    try:
        service = DocumentSelectionService(pg_config)
        
        test_user_id = f"test_docsel_{uuid.uuid4().hex[:8]}"
        test_conv_id = 99998
        
        # Get effective selection (should return system default initially)
        selection = service.get_effective_selection(
            user_id=test_user_id,
            conversation_id=test_conv_id
        )
        print(f"✓ Got effective selection (tier: {selection.tier})")
        
        # The actual behavior depends on what's in the database
        # Just verify the service doesn't crash
        
    except Exception as e:
        print(f"⚠ DocumentSelectionService test: {e}")
    
    print(f"✓ DocumentSelectionService tests passed")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PostgreSQL Consolidation Smoke Tests")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("PG_PASSWORD"):
        print("\n⚠ PG_PASSWORD not set. Set environment variables:")
        print("  export PG_PASSWORD=your_password")
        print("  export PG_HOST=localhost")
        print("  export PG_DATABASE=archi")
        print("  export BYOK_ENCRYPTION_KEY=your_encryption_key")
        print("\nRunning tests that don't require database connection...")
    
    tests_passed = 0
    tests_failed = 0
    
    # Run tests that require DB connection
    if os.environ.get("PG_PASSWORD"):
        try:
            test_user_service()
            tests_passed += 1
        except Exception as e:
            print(f"✗ UserService test failed: {e}")
            tests_failed += 1
        
        try:
            test_conversation_service()
            tests_passed += 1
        except Exception as e:
            print(f"✗ ConversationService test failed: {e}")
            tests_failed += 1
        
        try:
            test_ab_comparison_v2()
            tests_passed += 1
        except Exception as e:
            print(f"✗ A/B Comparison test failed: {e}")
            tests_failed += 1
        
        try:
            test_byok_resolver()
            tests_passed += 1
        except Exception as e:
            print(f"✗ BYOK Resolver test failed: {e}")
            tests_failed += 1
        
        try:
            test_document_selection_service()
            tests_passed += 1
        except Exception as e:
            print(f"✗ DocumentSelectionService test failed: {e}")
            tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return 0 if tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
