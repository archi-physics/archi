"""Regression tests for optional RBAC arguments and Flask audit endpoints."""

import json
import logging

import pytest
from flask import Flask, request

from src.utils.rbac import decorators, jwt_parser, registry


@pytest.fixture
def rbac_registry(monkeypatch):
    instance = registry.RBACRegistry(
        {
            "default_role": "reader",
            "roles": {
                "reader": {"permissions": ["documents:view"]},
                "editor": {
                    "inherits": ["reader"],
                    "permissions": ["documents:edit"],
                },
            },
        }
    )
    monkeypatch.setattr(registry, "_registry", instance)
    return instance


@pytest.mark.parametrize("kwargs", [{}, {"visited": None}, {"visited": set()}])
def test_resolve_permissions_with_optional_visited(rbac_registry, kwargs):
    assert rbac_registry._resolve_permissions("editor", **kwargs) == {
        "documents:view",
        "documents:edit",
    }
    assert rbac_registry._resolve_permissions("reader") == {"documents:view"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"original_roles": None},
        {"original_roles": []},
        {"original_roles": ["other"]},
    ],
)
def test_assign_default_role_with_optional_original_roles(
    rbac_registry, caplog, kwargs
):
    with caplog.at_level(logging.DEBUG, logger="rbac.audit"):
        assert jwt_parser.assign_default_role("user@example.com", **kwargs) == [
            "reader"
        ]

    entry = next(
        json.loads(record.message.removeprefix("AUDIT: "))
        for record in caplog.records
        if record.name == "rbac.audit" and record.message.startswith("AUDIT: ")
    )
    assert entry["roles"] == ["reader"]
    assert entry["source"] == "default"
    assert entry["is_default"] is True


@pytest.mark.parametrize(
    "missing_endpoint", [False, True], ids=["route", "404-handler"]
)
@pytest.mark.parametrize(
    "decorate,roles,denial_status",
    [
        pytest.param(decorators.require_authenticated, None, 401, id="auth-anonymous"),
        pytest.param(
            decorators.require_permission("documents:edit"),
            None,
            401,
            id="all-anonymous",
        ),
        pytest.param(
            decorators.require_permission("documents:edit"),
            ["reader"],
            403,
            id="all-denied",
        ),
        pytest.param(
            decorators.require_permission("documents:edit"),
            ["editor"],
            None,
            id="all-granted",
        ),
        pytest.param(
            decorators.require_any_permission(["documents:edit"]),
            None,
            401,
            id="any-anonymous",
        ),
        pytest.param(
            decorators.require_any_permission(["documents:edit"]),
            ["reader"],
            403,
            id="any-denied",
        ),
        pytest.param(
            decorators.require_any_permission(["documents:edit"]),
            ["editor"],
            None,
            id="any-granted",
        ),
        pytest.param(decorators.check_sso_required(), None, 401, id="sso-anonymous"),
    ],
)
def test_permission_audit_endpoint(
    rbac_registry, caplog, missing_endpoint, decorate, roles, denial_status
):
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="test-secret")
    app.add_url_rule("/login", "login", lambda: "login")
    view_calls = []

    @decorate
    def protected(error=None):
        view_calls.append(request.endpoint)
        return "allowed", 404 if error is not None else 200

    app.add_url_rule("/api/protected", "protected", protected)
    app.register_error_handler(404, protected)
    client = app.test_client()
    if roles is not None:
        with client.session_transaction() as session:
            session["logged_in"] = True
            session["roles"] = roles
            session["user"] = {"email": "user@example.com"}

    with caplog.at_level(logging.DEBUG, logger="rbac.audit"):
        response = client.get("/api/missing" if missing_endpoint else "/api/protected")

    assert response.status_code == (denial_status or (404 if missing_endpoint else 200))
    assert view_calls == (
        [] if denial_status else [None if missing_endpoint else "protected"]
    )
    endpoint = "<unknown>" if missing_endpoint else "protected"
    result = "DENIED" if denial_status else "GRANTED"
    audit_records = [record for record in caplog.records if record.name == "rbac.audit"]
    assert f" | {result} | {endpoint} | " in audit_records[0].message
    if denial_status:
        assert response.get_json()["status"] == denial_status
        entry = json.loads(audit_records[1].message.removeprefix("AUDIT: "))
        assert entry["endpoint"] == endpoint
        assert entry["result"] == result
        assert entry["roles"] == (roles or [])
