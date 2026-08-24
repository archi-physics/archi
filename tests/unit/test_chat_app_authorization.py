from unittest.mock import Mock, call, patch

from flask import Flask, session

from src.interfaces.chat_app import app as chat_app_module


class TestAuthorizeRequest:
    def test_returns_none_when_permission_is_granted(self):
        wrapper = object.__new__(chat_app_module.FlaskAppWrapper)
        wrapper.auth_enabled = True

        app = Flask(__name__)
        app.secret_key = "test-secret"
        with app.test_request_context("/api/protected"):
            session["logged_in"] = True
            session["roles"] = ["viewer"]

            with patch.object(chat_app_module, "has_permission", return_value=True):
                result = wrapper.authorize_request("evaluations:view")

        assert result is None

    def test_returns_forbidden_when_permission_is_denied(self):
        wrapper = object.__new__(chat_app_module.FlaskAppWrapper)
        wrapper.auth_enabled = True

        app = Flask(__name__)
        app.secret_key = "test-secret"
        with app.test_request_context("/api/protected"):
            session["logged_in"] = True
            session["roles"] = ["viewer"]
            session["user"] = {"email": "viewer@example.com"}

            with patch.object(chat_app_module, "has_permission", return_value=False):
                with patch("src.utils.rbac.audit.log_permission_check"):
                    response, status = wrapper.authorize_request("evaluations:manage")

        assert status == 403
        assert response.get_json() == {
            "error": "Forbidden",
            "message": "Permission denied: requires evaluations:manage",
            "required_permission": "evaluations:manage",
        }


class TestRequirePermission:
    def test_delegates_to_authorize_request_before_calling_the_view(self):
        wrapper = object.__new__(chat_app_module.FlaskAppWrapper)
        denial = ("Forbidden", 403)
        wrapper.authorize_request = Mock(side_effect=[None, denial])
        view = Mock(return_value="allowed")
        decorated_view = wrapper.require_perm("evaluations:view")(view)

        assert decorated_view() == "allowed"
        assert decorated_view() == denial
        assert view.call_count == 1
        assert wrapper.authorize_request.call_args_list == [
            call("evaluations:view"),
            call("evaluations:view"),
        ]
