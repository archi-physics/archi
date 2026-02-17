"""
API Blueprint - PostgreSQL-consolidated endpoints.

Provides REST API endpoints for:
- User management (preferences, BYOK API keys)
- Configuration (static/dynamic settings)
- Document selection (3-tier system)
- Analytics (model usage, A/B comparison stats)
"""
import os
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import List, Optional

from flask import Blueprint, jsonify, request, g, current_app

from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.env import read_secret
from src.utils.logging import get_logger
from src.utils.config_access import get_full_config
from src.archi.pipelines.agents.agent_spec import AgentSpecError, load_agent_spec

logger = get_logger(__name__)

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')


def _get_agents_dir_from_config() -> str:
    config = get_full_config()
    services_cfg = config.get("services", {}) if isinstance(config, dict) else {}
    chat_cfg = services_cfg.get("chat_app", {}) if isinstance(services_cfg, dict) else {}
    return chat_cfg.get("agents_dir") or "/root/archi/agents"


def _get_agent_class_name_from_config() -> Optional[str]:
    config = get_full_config()
    services_cfg = config.get("services", {}) if isinstance(config, dict) else {}
    chat_cfg = services_cfg.get("chat_app", {}) if isinstance(services_cfg, dict) else {}
    return chat_cfg.get("agent_class") or chat_cfg.get("pipeline")


def _get_agent_tool_registry(agent_class_name: Optional[str]) -> List[str]:
    if not agent_class_name:
        return []
    try:
        from src.archi import pipelines
    except Exception as exc:
        logger.warning("Failed to import pipelines module: %s", exc)
        return []
    agent_cls = getattr(pipelines, agent_class_name, None)
    if not agent_cls or not hasattr(agent_cls, "get_tool_registry"):
        return []
    try:
        dummy = agent_cls.__new__(agent_cls)
        registry = agent_cls.get_tool_registry(dummy) or {}
        return sorted([name for name in registry.keys() if isinstance(name, str)])
    except Exception as exc:
        logger.warning("Failed to read tool registry for %s: %s", agent_class_name, exc)
        return []


def _build_agent_template(name: str, tools: List[str]) -> str:
    tools_block = "\n".join(f"- {tool}" for tool in tools) if tools else "- <tool_name>"
    tools_comment = "\n".join(f"- {tool}" for tool in tools) if tools else "- (no tools available)"
    return (
        f"# {name}\n\n"
        "## Tools\n"
        f"{tools_block}\n\n"
        "## Prompt\n"
        "Write your system prompt here.\n\n"
        "<!--\n"
        "Available tools (registry):\n"
        f"{tools_comment}\n"
        "-->\n"
    )


def _sanitize_filename(filename: str) -> Optional[str]:
    import re
    if not isinstance(filename, str):
        return None
    name = filename.strip().replace("\\", "/")
    name = name.split("/")[-1]
    if not name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return None
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name


def get_services() -> PostgresServiceFactory:
    """Get or create the service factory for this request."""
    if 'services' not in g:
        # Check if factory already exists at app level
        if hasattr(current_app, 'pg_services'):
            g.services = current_app.pg_services
        else:
            # Prefer singleton if set by service entrypoint
            factory = PostgresServiceFactory.get_instance()
            if not factory:
                encryption_key = read_secret("BYOK_ENCRYPTION_KEY", default="")
                raw_config = get_full_config()
                factory = PostgresServiceFactory.from_yaml_config(
                    config=raw_config,
                    encryption_key=encryption_key,
                )
                current_app.pg_services = factory
            g.services = factory
    
    return g.services


def get_client_id() -> str:
    """Get client ID from request (session, header, or generate)."""
    # Check session first
    from flask import session
    if 'client_id' in session:
        return session['client_id']
    
    # Check header
    client_id = request.headers.get('X-Client-ID')
    if client_id:
        return client_id
    
    # Check JSON body
    if request.is_json and request.json:
        client_id = request.json.get('client_id')
        if client_id:
            return client_id
    
    # Generate anonymous client ID from request
    import hashlib
    import uuid
    user_agent = request.headers.get('User-Agent', '')
    remote_addr = request.remote_addr or ''
    fingerprint = f"{user_agent}:{remote_addr}:{uuid.uuid4()}"
    return f"anon_{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}"


def require_client_id(f):
    """Decorator to require and inject client_id."""
    @wraps(f)
    def decorated(*args, **kwargs):
        g.client_id = get_client_id()
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# User Endpoints
# =============================================================================

@api.route('/users/me', methods=['GET'])
@require_client_id
def get_current_user():
    """
    Get or create user for current client.
    
    Returns:
        User object with preferences
    """
    try:
        services = get_services()
        user = services.user_service.get_or_create_user(
            user_id=g.client_id,
            auth_provider='anonymous',
        )
        
        return jsonify({
            'id': user.id,
            'display_name': user.display_name,
            'email': user.email,
            'auth_provider': user.auth_provider,
            'theme': user.theme,
            'preferred_model': user.preferred_model,
            'preferred_temperature': float(user.preferred_temperature) if user.preferred_temperature else None,
            'has_openrouter_key': user.api_key_openrouter is not None,
            'has_openai_key': user.api_key_openai is not None,
            'has_anthropic_key': user.api_key_anthropic is not None,
            'created_at': user.created_at.isoformat() if user.created_at else None,
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/users/me/preferences', methods=['PATCH'])
@require_client_id
def update_user_preferences():
    """
    Update user preferences.
    
    Request body:
        theme: str (optional)
        preferred_model: str (optional)
        preferred_temperature: float (optional)
        display_name: str (optional)
    
    Returns:
        Updated user object
    """
    try:
        data = request.get_json() or {}
        
        # Validate temperature if provided
        if 'preferred_temperature' in data:
            temp = data['preferred_temperature']
            if temp is not None and (temp < 0 or temp > 2):
                return jsonify({'error': 'Temperature must be between 0 and 2'}), 400
        
        services = get_services()
        user = services.user_service.update_preferences(
            user_id=g.client_id,
            display_name=data.get('display_name'),
            theme=data.get('theme'),
            preferred_model=data.get('preferred_model'),
            preferred_temperature=data.get('preferred_temperature'),
        )
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'id': user.id,
            'display_name': user.display_name,
            'theme': user.theme,
            'preferred_model': user.preferred_model,
            'preferred_temperature': float(user.preferred_temperature) if user.preferred_temperature else None,
            'updated_at': user.updated_at.isoformat() if user.updated_at else None,
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/users/me/api-keys/<provider>', methods=['PUT'])
@require_client_id
def set_api_key(provider: str):
    """
    Set BYOK API key for a provider.
    
    URL params:
        provider: 'openrouter', 'openai', or 'anthropic'
    
    Request body:
        api_key: str (the API key to store)
    
    Returns:
        Success confirmation
    """
    if provider not in ('openrouter', 'openai', 'anthropic'):
        return jsonify({'error': f'Invalid provider: {provider}'}), 400
    
    try:
        data = request.get_json() or {}
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'api_key is required'}), 400
        
        services = get_services()
        services.user_service.set_api_key(
            user_id=g.client_id,
            provider=provider,
            api_key=api_key,
        )
        
        return jsonify({
            'success': True,
            'provider': provider,
            'message': f'{provider} API key stored securely',
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting API key: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/users/me/api-keys/<provider>', methods=['DELETE'])
@require_client_id
def delete_api_key(provider: str):
    """
    Delete BYOK API key for a provider.
    
    URL params:
        provider: 'openrouter', 'openai', or 'anthropic'
    
    Returns:
        Success confirmation
    """
    if provider not in ('openrouter', 'openai', 'anthropic'):
        return jsonify({'error': f'Invalid provider: {provider}'}), 400
    
    try:
        services = get_services()
        services.user_service.set_api_key(
            user_id=g.client_id,
            provider=provider,
            api_key=None,  # Setting to None deletes
        )
        
        return jsonify({
            'success': True,
            'provider': provider,
            'message': f'{provider} API key deleted',
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Configuration Endpoints
# =============================================================================

@api.route('/config/static', methods=['GET'])
def get_static_config():
    """
    Get static (deploy-time) configuration.
    
    Returns:
        Static config object
    """
    try:
        services = get_services()
        config = services.config_service.get_static_config()
        
        if not config:
            return jsonify({'error': 'Static config not initialized'}), 500
        
        return jsonify({
            'deployment_name': config.deployment_name,
            'config_version': config.config_version,
            'data_path': config.data_path,
            'prompts_path': config.prompts_path,
            'embedding_model': config.embedding_model,
            'embedding_dimensions': config.embedding_dimensions,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'distance_metric': config.distance_metric,
            'available_pipelines': config.available_pipelines,
            'available_models': config.available_models,
            'available_providers': config.available_providers,
            'auth_enabled': config.auth_enabled,
            'session_lifetime_days': config.session_lifetime_days,
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting static config: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/config/dynamic', methods=['GET'])
def get_dynamic_config():
    """
    Get dynamic (runtime) configuration.
    
    Returns:
        Dynamic config object
    """
    try:
        services = get_services()
        config = services.config_service.get_dynamic_config()
        
        if not config:
            return jsonify({'error': 'Dynamic config not initialized'}), 500
        
        return jsonify({
            'active_pipeline': config.active_pipeline,
            'active_model': config.active_model,
            'temperature': float(config.temperature),
            'max_tokens': config.max_tokens,
            'system_prompt': config.system_prompt,
            'top_p': float(config.top_p),
            'top_k': config.top_k,
            'repetition_penalty': float(config.repetition_penalty),
            'active_condense_prompt': config.active_condense_prompt,
            'active_chat_prompt': config.active_chat_prompt,
            'active_system_prompt': config.active_system_prompt,
            'num_documents_to_retrieve': config.num_documents_to_retrieve,
            'use_hybrid_search': config.use_hybrid_search,
            'bm25_weight': float(config.bm25_weight),
            'semantic_weight': float(config.semantic_weight),
            'ingestion_schedule': config.ingestion_schedule,
            'verbosity': config.verbosity,
            'updated_at': config.updated_at if config.updated_at else None,
            'updated_by': config.updated_by,
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting dynamic config: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/config/dynamic', methods=['PATCH'])
@require_client_id
def update_dynamic_config():
    """
    Update dynamic configuration. Admin only.
    
    Request body (all optional):
        active_pipeline: str
        active_model: str
        temperature: float (0-2)
        max_tokens: int
        system_prompt: str
        top_p: float (0-1)
        top_k: int
        repetition_penalty: float
        active_condense_prompt: str
        active_chat_prompt: str
        active_system_prompt: str
        num_documents_to_retrieve: int
        use_hybrid_search: bool
        bm25_weight: float (0-1)
        semantic_weight: float (0-1)
        ingestion_schedule: str
        verbosity: int (1-5)
    
    Returns:
        Updated dynamic config
    """
    try:
        data = request.get_json() or {}
        
        services = get_services()
        
        # Check if user is admin
        if not services.config_service.is_admin(g.client_id):
            return jsonify({'error': 'Admin access required'}), 403
        
        # Build kwargs from provided fields
        kwargs = {}
        field_mapping = {
            'active_pipeline': 'active_pipeline',
            'active_model': 'active_model',
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'system_prompt': 'system_prompt',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'repetition_penalty': 'repetition_penalty',
            'active_condense_prompt': 'active_condense_prompt',
            'active_chat_prompt': 'active_chat_prompt',
            'active_system_prompt': 'active_system_prompt',
            'num_documents_to_retrieve': 'num_documents_to_retrieve',
            'use_hybrid_search': 'use_hybrid_search',
            'bm25_weight': 'bm25_weight',
            'semantic_weight': 'semantic_weight',
            'ingestion_schedule': 'ingestion_schedule',
            'verbosity': 'verbosity',
        }
        
        for json_key, db_key in field_mapping.items():
            if json_key in data:
                kwargs[db_key] = data[json_key]
        
        config = services.config_service.update_dynamic_config(
            updated_by=g.client_id,
            **kwargs
        )
        
        return jsonify({
            'success': True,
            'active_pipeline': config.active_pipeline,
            'active_model': config.active_model,
            'temperature': float(config.temperature),
            'max_tokens': config.max_tokens,
            'updated_at': config.updated_at if config.updated_at else None,
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error updating dynamic config: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/config/effective', methods=['GET'])
@require_client_id
def get_effective_config():
    """
    Get effective configuration for the current user.
    
    Merges user preferences with deployment defaults.
    
    Returns:
        Effective configuration values
    """
    try:
        services = get_services()
        config = services.config_service.get_effective_config(g.client_id)
        return jsonify(config), 200
    except Exception as e:
        logger.error(f"Error getting effective config: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/config/audit', methods=['GET'])
@require_client_id
def get_config_audit():
    """
    Get configuration audit log. Admin only.
    
    Query params:
        user_id: Filter by user (optional)
        config_type: Filter by type ('dynamic' or 'user_pref') (optional)
        limit: Max entries (default 100)
    
    Returns:
        List of audit log entries
    """
    try:
        services = get_services()
        
        # Check if user is admin
        if not services.config_service.is_admin(g.client_id):
            return jsonify({'error': 'Admin access required'}), 403
        
        user_id = request.args.get('user_id')
        config_type = request.args.get('config_type')
        limit = int(request.args.get('limit', 100))
        
        entries = services.config_service.get_audit_log(
            user_id=user_id,
            config_type=config_type,
            limit=limit,
        )
        
        return jsonify({'entries': entries}), 200
        
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Document Selection Endpoints
# =============================================================================

@api.route('/documents/selection', methods=['GET'])
@require_client_id
def get_document_selection():
    """
    Get enabled documents for a conversation.
    
    Query params:
        conversation_id: int (required)
    
    Returns:
        List of enabled document IDs
    """
    try:
        conversation_id = request.args.get('conversation_id', type=int)
        if not conversation_id:
            return jsonify({'error': 'conversation_id is required'}), 400
        
        services = get_services()
        doc_ids = services.document_selection_service.get_enabled_document_ids(
            user_id=g.client_id,
            conversation_id=conversation_id,
        )
        
        return jsonify({
            'conversation_id': conversation_id,
            'enabled_document_ids': doc_ids,
            'count': len(doc_ids),
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting document selection: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/documents/user-defaults', methods=['PUT'])
@require_client_id
def set_user_document_default():
    """
    Set user's default enabled/disabled state for a document.
    
    Request body:
        document_id: int
        enabled: bool
    
    Returns:
        Updated selection
    """
    try:
        data = request.get_json() or {}
        document_id = data.get('document_id')
        enabled = data.get('enabled')
        
        if document_id is None:
            return jsonify({'error': 'document_id is required'}), 400
        if enabled is None:
            return jsonify({'error': 'enabled is required'}), 400
        
        services = get_services()
        services.document_selection_service.set_user_default(
            user_id=g.client_id,
            document_id=document_id,
            enabled=enabled,
        )
        
        return jsonify({
            'success': True,
            'document_id': document_id,
            'enabled': enabled,
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting user document default: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/documents/conversation-override', methods=['PUT'])
@require_client_id
def set_conversation_document_override():
    """
    Set conversation-specific document override.
    
    Request body:
        conversation_id: int
        document_id: int
        enabled: bool
    
    Returns:
        Updated selection
    """
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id')
        document_id = data.get('document_id')
        enabled = data.get('enabled')
        
        if conversation_id is None:
            return jsonify({'error': 'conversation_id is required'}), 400
        if document_id is None:
            return jsonify({'error': 'document_id is required'}), 400
        if enabled is None:
            return jsonify({'error': 'enabled is required'}), 400
        
        services = get_services()
        services.document_selection_service.set_conversation_override(
            conversation_id=conversation_id,
            document_id=document_id,
            enabled=enabled,
        )
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'document_id': document_id,
            'enabled': enabled,
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting conversation override: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/documents/conversation-override', methods=['DELETE'])
@require_client_id
def clear_conversation_document_override():
    """
    Clear conversation-specific document override (fall back to user default).
    
    Request body:
        conversation_id: int
        document_id: int
    
    Returns:
        Success confirmation
    """
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id')
        document_id = data.get('document_id')
        
        if conversation_id is None:
            return jsonify({'error': 'conversation_id is required'}), 400
        if document_id is None:
            return jsonify({'error': 'document_id is required'}), 400
        
        services = get_services()
        services.document_selection_service.clear_conversation_override(
            conversation_id=conversation_id,
            document_id=document_id,
        )
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'document_id': document_id,
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing conversation override: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Analytics Endpoints
# =============================================================================

@api.route('/analytics/model-usage', methods=['GET'])
def get_model_usage_stats():
    """
    Get model usage statistics.
    
    Query params:
        start_date: ISO date (optional)
        end_date: ISO date (optional)
        service: str (optional, e.g., 'chat')
    
    Returns:
        Model usage statistics
    """
    try:
        start_date = None
        end_date = None
        
        if request.args.get('start_date'):
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        if request.args.get('end_date'):
            end_date = datetime.fromisoformat(request.args.get('end_date'))
        
        service = request.args.get('service')
        
        services = get_services()
        stats = services.conversation_service.get_model_usage_stats(
            start_date=start_date,
            end_date=end_date,
            archi_service=service,
        )
        
        return jsonify({
            'stats': stats,
            'filters': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'service': service,
            },
        }), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {e}'}), 400
    except Exception as e:
        logger.error(f"Error getting model usage stats: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/analytics/ab-comparisons', methods=['GET'])
def get_ab_comparison_stats():
    """
    Get A/B comparison statistics.
    
    Query params:
        model_a: str (optional)
        model_b: str (optional)
        start_date: ISO date (optional)
        end_date: ISO date (optional)
    
    Returns:
        A/B comparison statistics with win rates
    """
    try:
        start_date = None
        end_date = None
        
        if request.args.get('start_date'):
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        if request.args.get('end_date'):
            end_date = datetime.fromisoformat(request.args.get('end_date'))
        
        model_a = request.args.get('model_a')
        model_b = request.args.get('model_b')
        
        services = get_services()
        stats = services.conversation_service.get_model_comparison_stats(
            model_a=model_a,
            model_b=model_b,
            start_date=start_date,
            end_date=end_date,
        )
        
        return jsonify({
            **stats,
            'filters': {
                'model_a': model_a,
                'model_b': model_b,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
            },
        }), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {e}'}), 400
    except Exception as e:
        logger.error(f"Error getting A/B comparison stats: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Prompts Endpoints
# =============================================================================

@api.route('/agents/template', methods=['GET'])
def get_agent_spec_template():
    """
    Return a prefilled agent spec template and available tools.
    """
    try:
        agent_name = request.args.get("name") or "New Agent"
        agent_class = _get_agent_class_name_from_config()
        tools = _get_agent_tool_registry(agent_class)
        return jsonify({
            "name": agent_name,
            "agent_class": agent_class,
            "tools": tools,
            "template": _build_agent_template(agent_name, tools),
        }), 200
    except Exception as exc:
        logger.error(f"Error building agent template: {exc}")
        return jsonify({'error': str(exc)}), 500


@api.route('/agents', methods=['POST'])
@require_client_id
def save_agent_spec():
    """
    Save a new agent spec markdown file.
    """
    try:
        data = request.get_json() or {}
        filename = _sanitize_filename(data.get("filename", ""))
        content = data.get("content")

        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        if not content or not isinstance(content, str):
            return jsonify({'error': 'Content is required'}), 400

        agents_dir = Path(_get_agents_dir_from_config())
        agents_dir.mkdir(parents=True, exist_ok=True)
        target_path = agents_dir / filename

        if target_path.exists():
            return jsonify({'error': f'File already exists: {filename}'}), 409

        target_path.write_text(content)
        try:
            load_agent_spec(target_path)
        except AgentSpecError as exc:
            target_path.unlink(missing_ok=True)
            return jsonify({'error': f'Invalid agent spec: {exc}'}), 400

        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(target_path),
        }), 200
    except Exception as exc:
        logger.error(f"Error saving agent spec: {exc}")
        return jsonify({'error': str(exc)}), 500

@api.route('/prompts', methods=['GET'])
def list_prompts():
    """
    List all available prompts by type.
    
    Returns:
        Dict mapping prompt type to list of prompt names
    """
    try:
        services = get_services()
        static_config = services.config_service.get_static_config()
        
        if not static_config:
            return jsonify({'error': 'Static config not initialized'}), 500
        
        from src.utils.prompt_service import PromptService
        prompt_service = PromptService(static_config.prompts_path)
        
        return jsonify({
            'prompts': prompt_service.list_all_prompts(),
            'prompts_path': static_config.prompts_path,
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/prompts/<prompt_type>', methods=['GET'])
def list_prompts_by_type(prompt_type: str):
    """
    List prompts of a specific type.
    
    Args:
        prompt_type: Type of prompt ('condense', 'chat', 'system')
    
    Returns:
        List of prompt names
    """
    try:
        services = get_services()
        static_config = services.config_service.get_static_config()
        
        if not static_config:
            return jsonify({'error': 'Static config not initialized'}), 500
        
        from src.utils.prompt_service import PromptService
        prompt_service = PromptService(static_config.prompts_path)
        
        if prompt_type not in PromptService.VALID_TYPES:
            return jsonify({
                'error': f'Invalid prompt type: {prompt_type}',
                'valid_types': PromptService.VALID_TYPES,
            }), 400
        
        return jsonify({
            'type': prompt_type,
            'prompts': prompt_service.list_prompts(prompt_type),
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing prompts by type: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/prompts/<prompt_type>/<name>', methods=['GET'])
def get_prompt_content(prompt_type: str, name: str):
    """
    Get the content of a specific prompt.
    
    Args:
        prompt_type: Type of prompt ('condense', 'chat', 'system')
        name: Name of the prompt (without extension)
    
    Returns:
        Prompt content
    """
    try:
        services = get_services()
        static_config = services.config_service.get_static_config()
        
        if not static_config:
            return jsonify({'error': 'Static config not initialized'}), 500
        
        from src.utils.prompt_service import PromptService, PromptNotFoundError
        prompt_service = PromptService(static_config.prompts_path)
        
        try:
            content = prompt_service.get(prompt_type, name)
            prompt = prompt_service.get_prompt(prompt_type, name)
            
            return jsonify({
                'type': prompt_type,
                'name': name,
                'content': content,
                'file_path': prompt.file_path if prompt else None,
            }), 200
            
        except PromptNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/prompts/reload', methods=['POST'])
@require_client_id
def reload_prompts():
    """
    Reload all prompts from disk. Admin only.
    
    Returns:
        Number of prompts loaded
    """
    try:
        services = get_services()
        
        # Check if user is admin
        if not services.config_service.is_admin(g.client_id):
            return jsonify({'error': 'Admin access required'}), 403
        
        static_config = services.config_service.get_static_config()
        
        if not static_config:
            return jsonify({'error': 'Static config not initialized'}), 500
        
        from src.utils.prompt_service import PromptService
        prompt_service = PromptService(static_config.prompts_path)
        count = prompt_service.reload()
        
        return jsonify({
            'success': True,
            'prompts_loaded': count,
            'prompts_path': static_config.prompts_path,
        }), 200
        
    except Exception as e:
        logger.error(f"Error reloading prompts: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@api.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status with database connectivity
    """
    try:
        services = get_services()
        
        # Test database connection
        with services.connection_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.utcnow().isoformat(),
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'database': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat(),
        }), 503


@api.route('/info', methods=['GET'])
def get_api_info():
    """
    Get API version and capabilities info.
    
    Returns:
        API information
    """
    return jsonify({
        'version': '2.0.0',
        'schema_version': '2.0.0',
        'features': [
            'user_management',
            'byok_api_keys',
            'dynamic_config',
            'document_selection_3tier',
            'model_tracking',
            'ab_comparisons',
            'analytics',
        ],
        'endpoints': {
            'users': '/api/users/*',
            'config': '/api/config/*',
            'documents': '/api/documents/*',
            'analytics': '/api/analytics/*',
        },
    }), 200


def register_api(app):
    """
    Register the API blueprint with a Flask app.
    
    Usage:
        from src.interfaces.chat_app.api import register_api
        register_api(app)
    """
    app.register_blueprint(api)
    logger.info("Registered API blueprint at /api")
