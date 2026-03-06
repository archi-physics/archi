"""
Service Alerts - Service Status Board endpoints.

Provides REST API and page endpoints for:
- Viewing the Service Status Board (GET /ssb/status)
- Creating service alerts (POST /api/ssb/alerts)
- Deleting service alerts (DELETE /api/ssb/alerts/<id>)
- Querying active banner alerts (used by the context processor in app.py)
"""
from datetime import datetime, timedelta

import psycopg2
from flask import Blueprint, jsonify, render_template, request, session

from src.utils.logging import get_logger
from src.utils.rbac.permission_enum import Permission
from src.utils.rbac.permissions import has_permission
from src.utils.sql import (
    SQL_DELETE_ALERT,
    SQL_INSERT_ALERT,
    SQL_LIST_ACTIVE_BANNER_ALERTS,
    SQL_LIST_ALERTS,
    SQL_SET_ALERT_EXPIRY,
)

logger = get_logger(__name__)

ssb = Blueprint('ssb', __name__)

# ---------------------------------------------------------------------------
# Module-level references injected at registration time
# ---------------------------------------------------------------------------
_pg_config: dict = {}
_auth_enabled: bool = False
_chat_app_config: dict = {}


# ---------------------------------------------------------------------------
# Helpers (public — called by the context processor in app.py)
# ---------------------------------------------------------------------------

def is_alert_manager() -> bool:
    """Return True if the current session user may manage service alerts.

    Rules (evaluated in order):
    1. Auth disabled -> everyone may manage.
    2. Auth enabled -> user is a manager if EITHER:
       a. Their username is in the ``alerts.managers`` list, OR
       b. Their session roles grant the ``alerts:manage`` permission.
    3. Auth enabled, no username match, no permission -> denied (safe default).
    """
    if not _auth_enabled:
        return True
    managers = (
        _chat_app_config
        .get('alerts', {})
        .get('managers')
    ) or []
    username = (session.get('user') or {}).get('username', '')
    if username and username in managers:
        return True
    if has_permission(Permission.Alerts.MANAGE):
        return True
    if not managers:
        logger.warning(
            "Alert managers list is not configured or empty and user lacks "
            "alerts:manage permission — no users will have permissions to "
            "manage alerts"
        )
    return False


def get_active_banner_alerts() -> list:
    """Return alerts that should appear in the page banner (non-expired, active)."""
    try:
        conn = psycopg2.connect(**_pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_LIST_ACTIVE_BANNER_ALERTS)
            rows = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'severity': row[1],
                    'message': row[2],
                    'description': row[3],
                    'created_by': row[4],
                    'created_at': row[5].isoformat() if row[5] else None,
                    'expires_at': row[6].isoformat() if row[6] else None,
                }
                for row in rows
            ]
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        logger.warning("Failed to fetch banner alerts: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Routes  (URLs match the original app.py layout exactly)
# ---------------------------------------------------------------------------

@ssb.route('/ssb/status')
def status_board():
    """Render the service status board page."""
    try:
        conn = psycopg2.connect(**_pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_LIST_ALERTS)
            rows = cursor.fetchall()
            alerts = [
                {
                    'id': row[0],
                    'severity': row[1],
                    'message': row[2],
                    'description': row[3],
                    'created_by': row[4],
                    'created_at': row[5],
                    'expires_at': row[6],
                    'active': row[7],
                    'expired': bool(row[6] and row[6] < datetime.now()),
                }
                for row in rows
            ]
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        logger.error("Failed to load status board alerts: %s", exc)
        alerts = []
    print(
        f"Loaded {len(alerts)} alerts for status board, "
        f"current user: '{(session.get('user') or {}).get('username', '')}' "
        f"is_alert_manager: {is_alert_manager()}"
    )
    return render_template(
        'status.html',
        alerts=alerts,
        is_alert_manager=is_alert_manager(),
    )


@ssb.route('/api/ssb/alerts', methods=['POST'])
def create_alert():
    """Create a new service alert. Restricted to alert managers."""
    if not is_alert_manager():
        return jsonify({'error': 'Forbidden'}), 403

    payload = request.get_json(silent=True) or {}
    severity = payload.get('severity', 'info')
    message = payload.get('message', '').strip()
    description = payload.get('description', '').strip() or None
    expires_in_hours = payload.get('expires_in_hours')
    expires_at_str = payload.get('expires_at')

    if not message:
        return jsonify({'error': 'message is required'}), 400
    if severity not in ('info', 'warning', 'alarm', 'news'):
        return jsonify({'error': 'invalid severity'}), 400

    created_by = None
    if _auth_enabled:
        created_by = (session.get('user') or {}).get('username') or None

    try:
        conn = psycopg2.connect(**_pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_INSERT_ALERT, (severity, message, description, created_by))
            row = cursor.fetchone()
            alert_id = row[0]

            expires_at = None
            if expires_at_str is not None:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                except ValueError:
                    return jsonify({
                        'error': 'expires_at must be a valid ISO-8601 datetime '
                                 '(e.g. 2026-03-01T18:00:00)'
                    }), 400
                if expires_at <= datetime.now():
                    return jsonify({'error': 'expires_at must be a future date/time'}), 400
            elif expires_in_hours is not None:
                if float(expires_in_hours) <= 0:
                    return jsonify({'error': 'expires_in_hours must be a positive number'}), 400
                expires_at = datetime.now() + timedelta(hours=float(expires_in_hours))

            if expires_at is not None:
                cursor.execute(SQL_SET_ALERT_EXPIRY, (expires_at, alert_id))

            conn.commit()
            logger.info(
                "Service alert %d created by %s: [%s] %s",
                alert_id, created_by, severity, message,
            )
            return jsonify({'id': alert_id, 'severity': severity, 'message': message}), 201
        finally:
            cursor.close()
            conn.close()
    except Exception as exc:
        logger.error("Failed to create alert: %s", exc)
        return jsonify({'error': 'internal error'}), 500


@ssb.route('/api/ssb/alerts/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id: int):
    """Delete a service alert by ID. Restricted to alert managers."""
    if not is_alert_manager():
        return jsonify({'error': 'Forbidden'}), 403

    try:
        conn = psycopg2.connect(**_pg_config)
        cursor = conn.cursor()
        try:
            cursor.execute(SQL_DELETE_ALERT, (alert_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        if not deleted:
            return jsonify({'error': 'not found'}), 404
        logger.info("Service alert %d deleted", alert_id)
        return jsonify({'deleted': alert_id}), 200
    except Exception as exc:
        logger.error("Failed to delete alert %d: %s", alert_id, exc)
        return jsonify({'error': 'internal error'}), 500


# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------

def register_service_alerts(app, *, pg_config, auth_enabled, chat_app_config,
                            require_auth):
    """Register the SSB blueprint with a Flask app.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    pg_config : dict
        PostgreSQL connection parameters.
    auth_enabled : bool
        Whether authentication is enabled.
    chat_app_config : dict
        The ``services.chat_app`` section of the application config.
    require_auth : callable
        The ``require_auth`` decorator from FlaskAppWrapper, applied to all
        blueprint routes via ``before_request``.
    """
    global _pg_config, _auth_enabled, _chat_app_config
    _pg_config = pg_config
    _auth_enabled = auth_enabled
    _chat_app_config = chat_app_config

    @ssb.before_request
    def _check_auth():
        """Apply the same auth gate used by the rest of the app."""
        # require_auth is a decorator that wraps a view; we use a no-op probe
        # function so the auth logic (session check, SSO redirect, 401) fires
        # and we can intercept a non-passthrough result.
        sentinel = object()

        @require_auth
        def _probe():
            return sentinel

        result = _probe()
        if result is not sentinel:
            return result  # redirect / 401 from require_auth

    app.register_blueprint(ssb)
    logger.info("Registered Service Alerts blueprint at /ssb")
