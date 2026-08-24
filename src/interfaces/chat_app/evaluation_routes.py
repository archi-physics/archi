# isort: skip_file
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from flask import (  # isort: skip
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    session,
)

from src.evaluation.qa.catalog import MAX_IMPORT_BYTES
from src.evaluation.qa.jobs import JobConflictError
from src.utils.logging import get_logger
from src.utils.rbac.permission_enum import Permission
from src.utils.rbac.permissions import has_permission

logger = get_logger(__name__)

evaluations_bp = Blueprint("evaluations", __name__)
_STATE_KEY = "EVALUATIONS_BLUEPRINT_STATE"


class HistoryRange(str, Enum):
    DAYS_7 = "7d"
    DAYS_30 = "30d"
    DAYS_90 = "90d"
    DAYS_365 = "365d"

    @property
    def days(self) -> int:
        return {
            HistoryRange.DAYS_7: 7,
            HistoryRange.DAYS_30: 30,
            HistoryRange.DAYS_90: 90,
            HistoryRange.DAYS_365: 365,
        }[self]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _history_cutoff(value: str) -> datetime:
    try:
        history_range = HistoryRange(value)
    except ValueError as exc:
        raise ValueError("range must be one of: 7d, 30d, 90d, 365d") from exc
    return _utc_now().astimezone(timezone.utc) - timedelta(days=history_range.days)


def _state():
    return current_app.config[_STATE_KEY]


def _service():
    return _state()["service"]


def _can_manage() -> bool:
    return not session.get("logged_in") or has_permission(Permission.Evaluations.MANAGE)


def _can_run() -> bool:
    return not session.get("logged_in") or has_permission(Permission.Evaluations.RUN)


def _approval_actor() -> str:
    if not session.get("logged_in"):
        return "local-operator"
    user = session.get("user") or {}
    if not isinstance(user, dict):
        raise ValueError("authenticated approval identity is unavailable")
    for field in ("email", "username", "sub", "id"):
        value = user.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("authenticated approval identity is unavailable")


@evaluations_bp.before_request
def _authorize():
    if request.path.startswith("/api/evaluations/atom-drafts"):
        permission = Permission.Evaluations.MANAGE
    elif request.method == "POST" and (
        request.path == "/api/evaluations/runs"
        or (
            request.path.startswith("/api/evaluations/jobs/")
            and request.path.endswith(("/cancel", "/continue"))
        )
        or (
            request.path.startswith("/api/evaluations/runs/")
            and request.path.endswith("/retry-failed")
        )
    ):
        permission = Permission.Evaluations.RUN
    elif request.method == "GET":
        permission = Permission.Evaluations.VIEW
    else:
        permission = Permission.Evaluations.MANAGE
    return _state()["authorize_request"](permission)


def _validate_json_body_in_request():
    if request.content_length is not None and request.content_length > MAX_IMPORT_BYTES:
        raise ValueError("request body exceeds the 25 MB limit")
    blob = request.stream.read(MAX_IMPORT_BYTES + 1)
    if len(blob) > MAX_IMPORT_BYTES:
        raise ValueError("request body exceeds the 25 MB limit")
    try:
        value = current_app.json.loads(blob)
    except (TypeError, ValueError) as exc:
        raise ValueError("request body must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _bounded_json_response(value, *, status=200):
    blob = current_app.json.dumps(value).encode("utf-8")
    if len(blob) > MAX_IMPORT_BYTES:
        raise ValueError("structured response exceeds the 25 MB limit")
    return Response(blob, status=status, mimetype="application/json")


def _read_upload(upload, kind):
    blob = upload.read(MAX_IMPORT_BYTES + 1)
    if len(blob) > MAX_IMPORT_BYTES:
        raise ValueError(f"{kind} upload exceeds the 25 MB limit")
    return blob


def _error(exc):
    if isinstance(exc, LookupError):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, JobConflictError):
        return jsonify({"error": str(exc)}), 409
    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 400
    logger.exception("Evaluation console request failed")
    return jsonify({"error": "Internal server error"}), 500


@evaluations_bp.route("/evaluations")
def evaluations_page():
    return render_template("evaluations.html")


@evaluations_bp.route("/api/evaluations/catalog")
def catalog():
    service = _service()
    can_run = _can_run()
    can_manage = _can_manage()
    return jsonify(
        {
            "datasets": service.list_datasets(),
            "profiles": service.list_profiles(),
            "agents": service.list_agents(),
            "jobs": service.list_jobs(include_run=can_run),
            "permissions": {"can_run": can_run, "can_manage": can_manage},
        }
    )


@evaluations_bp.route("/api/evaluations/datasets", methods=["GET", "POST"])
def datasets():
    service = _service()
    if request.method == "GET":
        return jsonify({"datasets": service.list_datasets()})
    try:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ValueError("dataset file is required")
        metadata, created = service.import_dataset(
            request.form.get("name") or upload.filename,
            upload.filename,
            _read_upload(upload, "dataset"),
        )
        return jsonify({"dataset": metadata, "created": created}), (
            201 if created else 200
        )
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/datasets/<dataset_id>")
def dataset_detail(dataset_id):
    try:
        return jsonify({"dataset": _service().get_dataset(dataset_id)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/datasets/<dataset_id>/generate-atoms", methods=["POST"]
)
def generate_atoms(dataset_id):
    try:
        body = _validate_json_body_in_request()
        static_only = body.get("static_only", False)
        if not isinstance(static_only, bool):
            raise ValueError("static_only must be a boolean")
        generate_options = {"static_only": True} if static_only else {}
        job = _service().start_atom_generation(
            dataset_id,
            body.get("profile_id") or "builtin",
            **generate_options,
        )
        return jsonify({"job": job}), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/datasets/<dataset_id>/refresh-live", methods=["POST"]
)
def refresh_live(dataset_id):
    try:
        body = _validate_json_body_in_request()
        job = _service().start_live_refresh(
            dataset_id, body.get("profile_id") or "builtin"
        )
        return jsonify({"job": job}), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/datasets/<dataset_id>/review-atoms", methods=["POST"]
)
def review_atoms(dataset_id):
    try:
        draft = _service().create_atom_review_draft(dataset_id)
        return _bounded_json_response({"draft": draft}, status=201)
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/atom-drafts/<draft_id>")
def atom_draft(draft_id):
    try:
        return _bounded_json_response({"draft": _service().get_atom_draft(draft_id)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/atom-drafts/<draft_id>/retry-failed",
    methods=["POST"],
)
def retry_failed_atoms(draft_id):
    try:
        job = _service().start_atom_retry(draft_id)
        return jsonify({"job": job}), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/atom-drafts/<draft_id>/static-only",
    methods=["POST"],
)
def make_atom_draft_static_only(draft_id):
    try:
        draft = _service().make_atom_draft_static_only(draft_id)
        return _bounded_json_response({"draft": draft})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/atom-drafts/<draft_id>/save", methods=["POST"])
def save_atom_draft(draft_id):
    try:
        body = _validate_json_body_in_request()
        dataset = _service().save_reviewed_dataset(
            draft_id,
            body.get("name"),
            body.get("reviewed_items"),
            approval_actor=_approval_actor(),
        )
        return jsonify({"dataset": dataset}), 201
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/profiles", methods=["GET", "POST"])
def profiles():
    service = _service()
    if request.method == "GET":
        return jsonify({"profiles": service.list_profiles()})
    try:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ValueError("profile file is required")
        metadata, created = service.import_profile(
            request.form.get("name") or upload.filename,
            upload.filename,
            _read_upload(upload, "profile"),
        )
        return jsonify({"profile": metadata, "created": created}), (
            201 if created else 200
        )
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/profiles/<profile_id>")
def profile_detail(profile_id):
    try:
        return jsonify({"profile": _service().get_profile(profile_id)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/agents/<agent_filename>")
def agent_detail(agent_filename):
    try:
        return jsonify({"agent": _service().get_agent_snapshot(agent_filename)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/runs", methods=["GET", "POST"])
def runs():
    service = _service()
    if request.method == "GET":
        try:
            cutoff = _history_cutoff(
                request.args.get("range", HistoryRange.DAYS_90.value)
            )
            return jsonify({"runs": service.list_runs(cutoff=cutoff)})
        except Exception as exc:
            return _error(exc)
    try:
        body = _validate_json_body_in_request()
        job = service.start_evaluation(
            name=body.get("name"),
            dataset_id=body.get("dataset_id"),
            profile_id=body.get("profile_id") or "builtin",
            agent_spec=body.get("agent_spec"),
            attempts=body.get("attempts", 1),
            run_workers=body.get("run_workers", 1),
            score_workers=body.get("score_workers", 1),
        )
        return jsonify({"job": job}), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/runs/<history_id>")
def run_detail(history_id):
    try:
        return _bounded_json_response({"run": _service().get_run(history_id)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/runs/<history_id>/retry-failed",
    methods=["POST"],
)
def retry_failed_evaluation(history_id):
    try:
        job = _service().start_evaluation_retry(history_id)
        return jsonify({"job": job}), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/runs/<history_id>/report")
def run_report(history_id):
    try:
        return Response(
            _service().get_report(history_id),
            mimetype="text/markdown",
        )
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/jobs/<job_id>")
def job_detail(job_id):
    try:
        service = _service()
        job = service.get_job_detail(
            job_id,
            include_run=_can_run(),
            include_hidden=_can_manage(),
        )
        return _bounded_json_response({"job": job})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/jobs/<job_id>/cancel",
    methods=["POST"],
)
def cancel_job(job_id):
    try:
        return jsonify(_service().cancel_evaluation(job_id))
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/jobs/<job_id>/refresh-live",
    methods=["POST"],
)
def refresh_attention_job(job_id):
    try:
        body = _validate_json_body_in_request()
        result = _service().refresh_attention_evaluation(
            job_id, body.get("profile_id") or "builtin"
        )
        return jsonify(result), 202
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/jobs/<job_id>/continue",
    methods=["POST"],
)
def continue_job(job_id):
    try:
        return jsonify({"job": _service().continue_evaluation(job_id)}), 202
    except Exception as exc:
        return _error(exc)


def register_evaluations(app, *, authorize_request, service):
    app.config[_STATE_KEY] = {
        "authorize_request": authorize_request,
        "service": service,
    }
    app.register_blueprint(evaluations_bp)
    logger.info("Registered evaluation console at /evaluations")
