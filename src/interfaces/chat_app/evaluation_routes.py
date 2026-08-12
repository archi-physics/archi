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
)

from src.evaluation.qa.catalog import MAX_IMPORT_BYTES
from src.evaluation.qa.jobs import JobConflictError
from src.utils.logging import get_logger
from src.utils.rbac.permission_enum import Permission

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


@evaluations_bp.before_request
def _authorize():
    if request.path.startswith("/api/evaluations/atom-drafts"):
        permission = Permission.Evaluations.MANAGE
    elif request.method == "POST" and (
        request.path == "/api/evaluations/runs"
        or (
            request.path.startswith("/api/evaluations/jobs/")
            and request.path.endswith("/cancel")
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
    sentinel = object()

    @_state()["require_perm"](permission)
    def probe():
        return sentinel

    result = probe()
    if result is not sentinel:
        return result


def _json_body():
    value = request.get_json(silent=True)
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


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
    return jsonify(
        {
            "datasets": service.catalog.list_datasets(),
            "profiles": service.catalog.list_profiles(),
            "agents": service.list_agents(),
            "jobs": service.list_jobs(),
        }
    )


@evaluations_bp.route("/api/evaluations/datasets", methods=["GET", "POST"])
def datasets():
    service = _service()
    if request.method == "GET":
        return jsonify({"datasets": service.catalog.list_datasets()})
    try:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ValueError("dataset file is required")
        metadata, created = service.catalog.import_dataset(
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
        return jsonify({"dataset": _service().catalog.get_dataset(dataset_id)})
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route(
    "/api/evaluations/datasets/<dataset_id>/generate-atoms", methods=["POST"]
)
def generate_atoms(dataset_id):
    try:
        body = _json_body()
        job = _service().start_atom_generation(
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
        draft = _service().catalog.create_atom_review_draft(dataset_id)
        return jsonify({"draft": draft}), 201
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/atom-drafts/<draft_id>")
def atom_draft(draft_id):
    try:
        return jsonify({"draft": _service().catalog.get_atom_draft(draft_id)})
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


@evaluations_bp.route("/api/evaluations/atom-drafts/<draft_id>/save", methods=["POST"])
def save_atom_draft(draft_id):
    try:
        body = _json_body()
        dataset = _service().catalog.save_reviewed_dataset(
            draft_id, body.get("name"), body.get("reviewed_items")
        )
        return jsonify({"dataset": dataset}), 201
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/profiles", methods=["GET", "POST"])
def profiles():
    service = _service()
    if request.method == "GET":
        return jsonify({"profiles": service.catalog.list_profiles()})
    try:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ValueError("profile file is required")
        metadata, created = service.catalog.import_profile(
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
        return jsonify({"profile": _service().catalog.get_profile(profile_id)})
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
            return jsonify({"runs": service.history.list_runs(cutoff=cutoff)})
        except Exception as exc:
            return _error(exc)
    try:
        body = _json_body()
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
        return jsonify({"run": _service().history.get_run(history_id)})
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
            _service().history.get_report(history_id),
            mimetype="text/markdown",
        )
    except Exception as exc:
        return _error(exc)


@evaluations_bp.route("/api/evaluations/jobs/<job_id>")
def job_detail(job_id):
    try:
        return jsonify({"job": _service().get_job(job_id)})
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


def register_evaluations(app, *, require_perm, service):
    app.config[_STATE_KEY] = {
        "require_perm": require_perm,
        "service": service,
    }
    app.register_blueprint(evaluations_bp)
    logger.info("Registered evaluation console at /evaluations")
