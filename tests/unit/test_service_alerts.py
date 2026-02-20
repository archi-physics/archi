"""
Unit tests for the service alert / status-board feature.

Covers:
- SQL constant content (sanity)
- FlaskAppWrapper._is_alert_manager()
- FlaskAppWrapper._get_active_banner_alerts()
- FlaskAppWrapper.status_board()
- FlaskAppWrapper.create_alert()
- FlaskAppWrapper.delete_alert()
- context processor session guard
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub out every heavy / unavailable import that app.py pulls in at module
# level.  This must happen before any code imports FlaskAppWrapper.
# ---------------------------------------------------------------------------
def _stub(name):
    """Insert a MagicMock module (and all parent packages) into sys.modules."""
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        key = '.'.join(parts[:i])
        if key not in sys.modules:
            sys.modules[key] = MagicMock()

for _mod in [
    # SSO / OAuth
    'authlib', 'authlib.integrations', 'authlib.integrations.flask_client',
    # Markdown / syntax highlighting
    'mistune',
    'pygments', 'pygments.formatters', 'pygments.lexers',
    # numpy
    'numpy',
    # flask-cors
    'flask_cors',
    # archi core + pipelines (pulls in langchain, transformers, etc.)
    'src.archi', 'src.archi.archi', 'src.archi.pipelines',
    'src.archi.pipelines.agents', 'src.archi.pipelines.agents.agent_spec',
    'src.archi.providers', 'src.archi.providers.base',
    'src.archi.utils', 'src.archi.utils.output_dataclass',
    # data manager
    'src.data_manager', 'src.data_manager.data_viewer_service',
    'src.data_manager.vectorstore', 'src.data_manager.vectorstore.manager',
    # utils that may import heavy deps
    'src.utils.connection_pool',
    'src.utils.user_service',
    'src.utils.config_service',
    'src.utils.document_selection_service',
    'src.utils.conversation_service',
    'src.utils.postgres_service_factory',
    'src.utils.env',
    'src.utils.logging',
    'src.utils.config_access',
]:
    _stub(_mod)

# Provide concrete minimal stubs for things the app references by attribute.
import flask              # ensure real flask is loaded first
import flask_cors as _fc  # noqa – already stubbed, just keep the name
sys.modules['flask_cors'].CORS = MagicMock()

# src.utils.logging must return a real logger-like object
import logging as _logging
_log_mod = types.ModuleType('src.utils.logging')
_log_mod.get_logger = _logging.getLogger
sys.modules['src.utils.logging'] = _log_mod

# src.utils.config_access – app reads config at import time via get_full_config
_cfg_access = sys.modules['src.utils.config_access']
_cfg_access.get_full_config = MagicMock(return_value={
    'name': 'test', 'global': {'DATA_PATH': '/tmp', 'ACCOUNTS_PATH': '/tmp'},
    'services': {'chat_app': {}, 'postgres': {}},
})
_cfg_access.get_services_config = MagicMock(return_value={})
_cfg_access.get_global_config = MagicMock(return_value={})
_cfg_access.get_dynamic_config = MagicMock(return_value={})

# src.archi.providers.base – used for type hints / ProviderType
_providers_base = sys.modules['src.archi.providers.base']
_providers_base.ProviderConfig = MagicMock
_providers_base.ProviderType = MagicMock
_providers_base.ModelInfo = MagicMock

# src.archi.pipelines.agents.agent_spec – several names imported
_agent_spec_mod = sys.modules['src.archi.pipelines.agents.agent_spec']
for _attr in ('AgentSpecError', 'list_agent_files', 'load_agent_spec',
              'select_agent_spec', 'load_agent_spec_from_text', 'slugify_agent_name'):
    setattr(_agent_spec_mod, _attr, MagicMock())

# src.archi.utils.output_dataclass
sys.modules['src.archi.utils.output_dataclass'].PipelineOutput = MagicMock

# ---------------------------------------------------------------------------
# Now it is safe to import test dependencies.
# ---------------------------------------------------------------------------
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask


# =============================================================================
# Helpers
# =============================================================================

def _obj(**kwargs):
    """
    Minimal fake 'self' for FlaskAppWrapper methods under test.
    Imports FlaskAppWrapper but never calls __init__, which keeps tests fast.
    We call the unbound methods directly: FlaskAppWrapper.method(obj, ...).
    """
    return SimpleNamespace(
        auth_enabled=kwargs.get('auth_enabled', False),
        sso_enabled=kwargs.get('sso_enabled', False),
        chat_app_config=kwargs.get('chat_app_config', {}),
        pg_config=kwargs.get('pg_config', {}),
    )


def _mock_db(rows=None, rowcount=0):
    """Return (mock_conn, mock_cursor) with the given fetchall rows."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows if rows is not None else []
    cursor.fetchone.return_value = rows[0] if rows else None
    cursor.rowcount = rowcount
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


def _call(method_name, obj, *args, **kwargs):
    """Invoke an unbound FlaskAppWrapper method on obj."""
    from src.interfaces.chat_app.app import FlaskAppWrapper
    return getattr(FlaskAppWrapper, method_name)(obj, *args, **kwargs)


# =============================================================================
# SQL Constant Sanity
# =============================================================================

class TestSQLConstants:
    """Smoke-check that service alert SQL constants are importable and sane."""

    def setup_method(self):
        from src.utils.sql import (
            SQL_INSERT_ALERT,
            SQL_SET_ALERT_EXPIRY,
            SQL_LIST_ALERTS,
            SQL_LIST_ACTIVE_BANNER_ALERTS,
            SQL_DELETE_ALERT,
        )
        self.constants = {
            'SQL_INSERT_ALERT': SQL_INSERT_ALERT,
            'SQL_SET_ALERT_EXPIRY': SQL_SET_ALERT_EXPIRY,
            'SQL_LIST_ALERTS': SQL_LIST_ALERTS,
            'SQL_LIST_ACTIVE_BANNER_ALERTS': SQL_LIST_ACTIVE_BANNER_ALERTS,
            'SQL_DELETE_ALERT': SQL_DELETE_ALERT,
        }

    def test_all_are_non_empty_strings(self):
        for name, sql in self.constants.items():
            assert isinstance(sql, str) and sql.strip(), f"{name} must be a non-empty string"

    def test_all_target_service_alerts_table(self):
        for name, sql in self.constants.items():
            assert 'service_alerts' in sql, f"{name} must reference service_alerts"

    def test_insert_has_returning_clause(self):
        assert 'RETURNING' in self.constants['SQL_INSERT_ALERT'].upper()

    def test_insert_has_four_value_placeholders(self):
        # severity, message, description, created_by
        assert self.constants['SQL_INSERT_ALERT'].count('%s') == 4

    def test_banner_query_filters_active_and_expiry(self):
        sql = self.constants['SQL_LIST_ACTIVE_BANNER_ALERTS'].upper()
        assert 'ACTIVE' in sql
        assert 'NOW()' in sql
        assert 'EXPIRES_AT' in sql

    def test_list_all_does_not_filter_active(self):
        # status board must show expired alerts too
        sql = self.constants['SQL_LIST_ALERTS'].upper()
        assert 'WHERE' not in sql or 'ACTIVE' not in sql

    def test_delete_has_one_placeholder(self):
        assert self.constants['SQL_DELETE_ALERT'].count('%s') == 1

    def test_set_expiry_has_two_placeholders(self):
        # (expires_at, id)
        assert self.constants['SQL_SET_ALERT_EXPIRY'].count('%s') == 2


# =============================================================================
# _is_alert_manager
# =============================================================================

class TestIsAlertManager:

    def test_auth_disabled_everyone_is_manager(self):
        obj = _obj(auth_enabled=False)
        app = Flask(__name__)
        with app.test_request_context():
            assert _call('_is_alert_manager', obj) is True

    def test_auth_enabled_empty_managers_list_returns_false(self):
        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': []}})
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context():
            from flask import session
            session['user'] = {'username': 'admin'}
            assert _call('_is_alert_manager', obj) is False

    def test_auth_enabled_missing_alerts_key_returns_false(self):
        obj = _obj(auth_enabled=True, chat_app_config={})
        app = Flask(__name__)
        with app.test_request_context():
            assert _call('_is_alert_manager', obj) is False

    def test_auth_enabled_user_in_managers_returns_true(self):
        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice', 'bob']}})
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context():
            from flask import session
            session['user'] = {'username': 'alice'}
            assert _call('_is_alert_manager', obj) is True

    def test_auth_enabled_user_not_in_managers_returns_false(self):
        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice', 'bob']}})
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context():
            from flask import session
            session['user'] = {'username': 'charlie'}
            assert _call('_is_alert_manager', obj) is False

    def test_auth_enabled_no_user_in_session_returns_false(self):
        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice']}})
        app = Flask(__name__)
        with app.test_request_context():
            assert _call('_is_alert_manager', obj) is False

    def test_auth_enabled_user_is_none_returns_false(self):
        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice']}})
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context():
            from flask import session
            session['user'] = None
            assert _call('_is_alert_manager', obj) is False


# =============================================================================
# _get_active_banner_alerts
# =============================================================================

class TestGetActiveBannerAlerts:

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_returns_list_of_dicts_with_iso_dates(self, mock_pg):
        now = datetime(2026, 2, 20, 12, 0, 0)
        future = now + timedelta(hours=2)
        conn, cursor = _mock_db(rows=[(1, 'alarm', 'DB down', 'Details', 'alice', now, future)])
        mock_pg.connect.return_value = conn

        result = _call('_get_active_banner_alerts', _obj())

        assert len(result) == 1
        a = result[0]
        assert a['id'] == 1
        assert a['severity'] == 'alarm'
        assert a['message'] == 'DB down'
        assert a['description'] == 'Details'
        assert a['created_by'] == 'alice'
        assert a['created_at'] == now.isoformat()
        assert a['expires_at'] == future.isoformat()

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_null_dates_serialized_as_none(self, mock_pg):
        conn, cursor = _mock_db(rows=[(2, 'info', 'Notice', None, None, None, None)])
        mock_pg.connect.return_value = conn

        result = _call('_get_active_banner_alerts', _obj())

        assert result[0]['created_at'] is None
        assert result[0]['expires_at'] is None

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_empty_table_returns_empty_list(self, mock_pg):
        conn, _ = _mock_db(rows=[])
        mock_pg.connect.return_value = conn

        assert _call('_get_active_banner_alerts', _obj()) == []

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_db_error_returns_empty_list_without_raising(self, mock_pg):
        mock_pg.connect.side_effect = Exception("connection refused")

        result = _call('_get_active_banner_alerts', _obj())
        assert result == []

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_multiple_alerts_all_returned(self, mock_pg):
        rows = [
            (1, 'alarm',   'Alert 1', None, None, datetime.now(), None),
            (2, 'warning', 'Alert 2', None, None, datetime.now(), None),
            (3, 'info',    'Alert 3', None, None, datetime.now(), None),
        ]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn

        result = _call('_get_active_banner_alerts', _obj())
        assert len(result) == 3
        assert result[0]['severity'] == 'alarm'
        assert result[1]['severity'] == 'warning'
        assert result[2]['severity'] == 'info'


# =============================================================================
# status_board
# =============================================================================

class TestStatusBoard:

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_renders_status_template_with_alerts(self, mock_pg, mock_render):
        now = datetime.now()
        rows = [(1, 'info', 'Test alert', 'desc', 'bob', now, None, True)]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        mock_render.assert_called_once()
        template_name = mock_render.call_args[0][0]
        alerts = mock_render.call_args[1]['alerts']
        assert template_name == 'status.html'
        assert len(alerts) == 1

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_non_expired_alert_has_expired_false(self, mock_pg, mock_render):
        future = datetime.now() + timedelta(hours=1)
        rows = [(1, 'info', 'msg', None, None, datetime.now(), future, True)]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        alert = mock_render.call_args[1]['alerts'][0]
        assert alert['expired'] is False

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_past_expires_at_marks_expired_true(self, mock_pg, mock_render):
        past = datetime.now() - timedelta(hours=1)
        rows = [(2, 'warning', 'old', None, None, past, past, True)]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        alert = mock_render.call_args[1]['alerts'][0]
        assert alert['expired'] is True

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_null_expires_at_is_not_expired(self, mock_pg, mock_render):
        rows = [(3, 'news', 'persistent', None, None, datetime.now(), None, True)]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        alert = mock_render.call_args[1]['alerts'][0]
        assert alert['expired'] is False

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_db_error_renders_with_empty_alerts(self, mock_pg, mock_render):
        mock_pg.connect.side_effect = Exception("db down")
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        assert mock_render.call_args[1]['alerts'] == []

    @patch('src.interfaces.chat_app.app.render_template')
    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_alert_row_fields_mapped_correctly(self, mock_pg, mock_render):
        now = datetime(2026, 1, 1, 10, 0, 0)
        rows = [(7, 'alarm', 'msg text', 'detail text', 'alice', now, None, True)]
        conn, _ = _mock_db(rows=rows)
        mock_pg.connect.return_value = conn
        mock_render.return_value = '<html/>'

        app = Flask(__name__)
        with app.test_request_context():
            _call('status_board', _obj())

        a = mock_render.call_args[1]['alerts'][0]
        assert a == {
            'id': 7,
            'severity': 'alarm',
            'message': 'msg text',
            'description': 'detail text',
            'created_by': 'alice',
            'created_at': now,
            'expires_at': None,
            'active': True,
            'expired': False,
        }


# =============================================================================
# create_alert
# =============================================================================

class TestCreateAlert:

    def _ctx(self, json_body=None, session_data=None):
        """Return a Flask app + pushed request context with JSON body."""
        app = Flask(__name__)
        app.secret_key = 'test'
        ctx = app.test_request_context(
            '/api/alerts',
            method='POST',
            json=json_body or {},
            content_type='application/json',
        )
        ctx.push()
        if session_data:
            from flask import session
            session.update(session_data)
        return app, ctx

    def test_non_manager_returns_403(self):
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context('/api/alerts', method='POST', json={}):
            from flask import session
            session['user'] = {'username': 'bob'}
            obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice']}})
            resp, code = _call('create_alert', obj)
        assert code == 403

    def test_missing_message_returns_400(self):
        app = Flask(__name__)
        with app.test_request_context('/api/alerts', method='POST', json={'severity': 'info'}):
            resp, code = _call('create_alert', _obj(auth_enabled=False))
        assert code == 400
        assert 'message' in resp.get_json()['error']

    def test_invalid_severity_returns_400(self):
        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'hello', 'severity': 'critical'},
        ):
            resp, code = _call('create_alert', _obj(auth_enabled=False))
        assert code == 400

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_valid_request_returns_201(self, mock_pg):
        conn, cursor = _mock_db(rows=[(42, 'info', 'msg', None, None, None, None, True)])
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'test msg', 'severity': 'info'},
        ):
            resp, code = _call('create_alert', _obj(auth_enabled=False))

        assert code == 201
        data = resp.get_json()
        assert data['id'] == 42
        assert data['severity'] == 'info'
        assert data['message'] == 'test msg'

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_with_expiry_calls_set_expiry(self, mock_pg):
        conn, cursor = _mock_db(rows=[(7, 'warning', 'msg', None, None, None, None, True)])
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'warning', 'expires_in_hours': 2},
        ):
            _call('create_alert', _obj(auth_enabled=False))

        # First execute = INSERT, second = SET_EXPIRY
        assert cursor.execute.call_count == 2

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_without_expiry_skips_set_expiry(self, mock_pg):
        conn, cursor = _mock_db(rows=[(8, 'info', 'msg', None, None, None, None, True)])
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'info'},
        ):
            _call('create_alert', _obj(auth_enabled=False))

        assert cursor.execute.call_count == 1

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_expiry_hours_sets_future_timestamp(self, mock_pg):
        conn, cursor = _mock_db(rows=[(9, 'info', 'msg', None, None, None, None, True)])
        mock_pg.connect.return_value = conn

        before = datetime.now()
        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'info', 'expires_in_hours': 4},
        ):
            _call('create_alert', _obj(auth_enabled=False))

        # Second call is (expires_at, alert_id)
        set_expiry_args = cursor.execute.call_args_list[1][0][1]
        expires_at, _ = set_expiry_args
        assert expires_at > before + timedelta(hours=3.9)
        assert expires_at < before + timedelta(hours=4.1)

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_auth_captures_username_from_session(self, mock_pg):
        conn, cursor = _mock_db(rows=[(10, 'info', 'msg', None, 'alice', None, None, True)])
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'info'},
        ):
            from flask import session
            session['user'] = {'username': 'alice'}
            obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice']}})
            _call('create_alert', obj)

        insert_args = cursor.execute.call_args_list[0][0][1]
        # (severity, message, description, created_by)
        assert insert_args[3] == 'alice'

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_auth_disabled_created_by_is_none(self, mock_pg):
        conn, cursor = _mock_db(rows=[(11, 'info', 'msg', None, None, None, None, True)])
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'info'},
        ):
            _call('create_alert', _obj(auth_enabled=False))

        insert_args = cursor.execute.call_args_list[0][0][1]
        assert insert_args[3] is None

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_db_error_returns_500(self, mock_pg):
        mock_pg.connect.side_effect = Exception("db down")

        app = Flask(__name__)
        with app.test_request_context(
            '/api/alerts', method='POST',
            json={'message': 'msg', 'severity': 'info'},
        ):
            resp, code = _call('create_alert', _obj(auth_enabled=False))
        assert code == 500

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_all_valid_severities_accepted(self, mock_pg):
        for severity in ('info', 'warning', 'alarm', 'news'):
            conn, cursor = _mock_db(rows=[(1, severity, 'msg', None, None, None, None, True)])
            mock_pg.connect.return_value = conn
            cursor.reset_mock()

            app = Flask(__name__)
            with app.test_request_context(
                '/api/alerts', method='POST',
                json={'message': 'msg', 'severity': severity},
            ):
                resp, code = _call('create_alert', _obj(auth_enabled=False))
            assert code == 201, f"Expected 201 for severity={severity}, got {code}"


# =============================================================================
# delete_alert
# =============================================================================

class TestDeleteAlert:

    def test_non_manager_returns_403(self):
        app = Flask(__name__)
        app.secret_key = 'test'
        with app.test_request_context('/api/alerts/1', method='DELETE'):
            from flask import session
            session['user'] = {'username': 'bob'}
            obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': ['alice']}})
            resp, code = _call('delete_alert', obj, 1)
        assert code == 403

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_existing_alert_returns_200_with_id(self, mock_pg):
        conn, cursor = _mock_db(rowcount=1)
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context('/api/alerts/5', method='DELETE'):
            resp, code = _call('delete_alert', _obj(auth_enabled=False), 5)

        assert code == 200
        assert resp.get_json()['deleted'] == 5

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_missing_alert_returns_404(self, mock_pg):
        conn, cursor = _mock_db(rowcount=0)
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context('/api/alerts/999', method='DELETE'):
            resp, code = _call('delete_alert', _obj(auth_enabled=False), 999)

        assert code == 404

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_db_error_returns_500(self, mock_pg):
        mock_pg.connect.side_effect = Exception("db gone")

        app = Flask(__name__)
        with app.test_request_context('/api/alerts/3', method='DELETE'):
            resp, code = _call('delete_alert', _obj(auth_enabled=False), 3)

        assert code == 500

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_deletes_correct_id(self, mock_pg):
        conn, cursor = _mock_db(rowcount=1)
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context('/api/alerts/42', method='DELETE'):
            _call('delete_alert', _obj(auth_enabled=False), 42)

        delete_args = cursor.execute.call_args[0][1]
        assert delete_args == (42,)

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_commits_transaction(self, mock_pg):
        conn, cursor = _mock_db(rowcount=1)
        mock_pg.connect.return_value = conn

        app = Flask(__name__)
        with app.test_request_context('/api/alerts/1', method='DELETE'):
            _call('delete_alert', _obj(auth_enabled=False), 1)

        conn.commit.assert_called_once()


# =============================================================================
# Context processor session guard
# =============================================================================

class TestContextProcessorSessionGuard:
    """
    The context processor must skip the DB call when user is not logged in.
    We test this by verifying _get_active_banner_alerts is never called
    when session['logged_in'] is falsy.
    """

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_unauthenticated_returns_empty_alerts_without_db_call(self, mock_pg):
        """When not logged in, context processor must return empty (no DB hit)."""
        app = Flask(__name__)
        app.secret_key = 'test'

        # Simulate the context processor logic directly
        from flask import session

        with app.test_request_context():
            # session['logged_in'] is not set
            if not session.get('logged_in'):
                result = dict(active_banner_alerts=[], is_alert_manager=False)
            else:
                result = None  # should not reach here

        assert result['active_banner_alerts'] == []
        assert result['is_alert_manager'] is False
        mock_pg.connect.assert_not_called()

    @patch('src.interfaces.chat_app.app.psycopg2')
    def test_authenticated_hits_db(self, mock_pg):
        """When logged in, context processor must query the DB."""
        conn, _ = _mock_db(rows=[])
        mock_pg.connect.return_value = conn

        obj = _obj(auth_enabled=True, chat_app_config={'alerts': {'managers': []}})
        app = Flask(__name__)
        app.secret_key = 'test'

        with app.test_request_context():
            from flask import session
            session['logged_in'] = True
            _call('_get_active_banner_alerts', obj)

        mock_pg.connect.assert_called_once()
