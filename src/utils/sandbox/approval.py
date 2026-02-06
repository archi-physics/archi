"""
Sandbox approval mechanism for human-in-the-loop code execution.

This module provides functionality to pause sandbox execution and wait for
user approval before running code. It supports both auto-approve and
manual approval modes.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    
    PENDING = "pending"
    """Waiting for user decision."""
    
    APPROVED = "approved"
    """User approved the execution."""
    
    REJECTED = "rejected"
    """User rejected the execution."""
    
    EXPIRED = "expired"
    """Approval request timed out."""
    
    CANCELLED = "cancelled"
    """Request was cancelled (e.g., stream aborted)."""


@dataclass
class ApprovalRequest:
    """Represents a pending sandbox approval request."""
    
    approval_id: str
    """Unique identifier for this approval request."""
    
    trace_id: str
    """The trace ID of the agent run that requested approval."""
    
    conversation_id: int
    """The conversation this request belongs to."""
    
    code: str
    """The code to be executed."""
    
    language: str
    """Programming language (python, bash, sh)."""
    
    image: str
    """Docker image to use for execution."""
    
    tool_call_id: str
    """The tool call ID from the agent."""
    
    status: ApprovalStatus = ApprovalStatus.PENDING
    """Current status of the request."""
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the request was created."""
    
    resolved_at: Optional[datetime] = None
    """When the request was approved/rejected/expired."""
    
    resolved_by: Optional[str] = None
    """User who resolved the request (if any)."""
    
    timeout_seconds: float = 300.0
    """How long to wait for approval before expiring."""
    
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.status != ApprovalStatus.PENDING:
            return False
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "approval_id": self.approval_id,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "code": self.code,
            "language": self.language,
            "image": self.image,
            "tool_call_id": self.tool_call_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "timeout_seconds": self.timeout_seconds,
        }


# Global registry of pending approval requests
_approval_requests: Dict[str, ApprovalRequest] = {}
_approval_lock = threading.Lock()

# Callbacks for notifying approval status changes
_approval_callbacks: Dict[str, List[Callable[[ApprovalRequest], None]]] = {}


def create_approval_request(
    *,
    trace_id: str,
    conversation_id: int,
    code: str,
    language: str,
    image: str,
    tool_call_id: str,
    timeout_seconds: float = 300.0,
) -> ApprovalRequest:
    """
    Create a new approval request for sandbox code execution.
    
    Args:
        trace_id: The trace ID of the agent run.
        conversation_id: The conversation ID.
        code: The code to execute.
        language: Programming language.
        image: Docker image.
        tool_call_id: The tool call ID.
        timeout_seconds: How long to wait for approval.
    
    Returns:
        The created ApprovalRequest.
    """
    approval_id = str(uuid.uuid4())
    
    request = ApprovalRequest(
        approval_id=approval_id,
        trace_id=trace_id,
        conversation_id=conversation_id,
        code=code,
        language=language,
        image=image,
        tool_call_id=tool_call_id,
        timeout_seconds=timeout_seconds,
    )
    
    with _approval_lock:
        _approval_requests[approval_id] = request
    
    logger.info(
        "Created approval request: id=%s, trace=%s, conversation=%d, language=%s",
        approval_id, trace_id, conversation_id, language
    )
    
    return request


def get_approval_request(approval_id: str) -> Optional[ApprovalRequest]:
    """Get an approval request by ID."""
    with _approval_lock:
        return _approval_requests.get(approval_id)


def get_pending_approvals_for_trace(trace_id: str) -> List[ApprovalRequest]:
    """Get all pending approval requests for a trace."""
    with _approval_lock:
        return [
            req for req in _approval_requests.values()
            if req.trace_id == trace_id and req.status == ApprovalStatus.PENDING
        ]


def get_pending_approvals_for_conversation(conversation_id: int) -> List[ApprovalRequest]:
    """Get all pending approval requests for a conversation."""
    with _approval_lock:
        return [
            req for req in _approval_requests.values()
            if req.conversation_id == conversation_id 
            and req.status == ApprovalStatus.PENDING
        ]


def resolve_approval(
    approval_id: str,
    approved: bool,
    resolved_by: Optional[str] = None,
) -> Optional[ApprovalRequest]:
    """
    Resolve an approval request (approve or reject).
    
    Args:
        approval_id: The approval request ID.
        approved: True to approve, False to reject.
        resolved_by: Optional user identifier.
    
    Returns:
        The updated ApprovalRequest, or None if not found.
    """
    with _approval_lock:
        request = _approval_requests.get(approval_id)
        if not request:
            logger.warning("Approval request not found: %s", approval_id)
            return None
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(
                "Approval request %s already resolved: %s", 
                approval_id, request.status.value
            )
            return request
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            request.resolved_at = datetime.now(timezone.utc)
            logger.info("Approval request %s expired", approval_id)
        else:
            request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            request.resolved_at = datetime.now(timezone.utc)
            request.resolved_by = resolved_by
            logger.info(
                "Approval request %s %s by %s",
                approval_id,
                "approved" if approved else "rejected",
                resolved_by or "unknown"
            )
        
        # Notify callbacks
        callbacks = _approval_callbacks.get(approval_id, [])
    
    # Call outside lock to avoid deadlocks
    for callback in callbacks:
        try:
            callback(request)
        except Exception as e:
            logger.error("Approval callback error: %s", e, exc_info=True)
    
    return request


def cancel_approvals_for_trace(trace_id: str) -> List[ApprovalRequest]:
    """
    Cancel all pending approvals for a trace (e.g., when stream is aborted).
    
    Args:
        trace_id: The trace ID to cancel approvals for.
    
    Returns:
        List of cancelled approval requests.
    """
    cancelled = []
    
    with _approval_lock:
        for request in _approval_requests.values():
            if request.trace_id == trace_id and request.status == ApprovalStatus.PENDING:
                request.status = ApprovalStatus.CANCELLED
                request.resolved_at = datetime.now(timezone.utc)
                cancelled.append(request)
    
    if cancelled:
        logger.info("Cancelled %d approval requests for trace %s", len(cancelled), trace_id)
    
    return cancelled


def wait_for_approval(
    approval_id: str,
    timeout: Optional[float] = None,
    poll_interval: float = 0.5,
) -> ApprovalRequest:
    """
    Wait for an approval request to be resolved.
    
    This is a blocking call that polls the approval status.
    
    Args:
        approval_id: The approval request ID.
        timeout: Maximum time to wait (uses request timeout if None).
        poll_interval: How often to check status.
    
    Returns:
        The resolved ApprovalRequest.
    
    Raises:
        ValueError: If approval request not found.
    """
    request = get_approval_request(approval_id)
    if not request:
        raise ValueError(f"Approval request not found: {approval_id}")
    
    effective_timeout = timeout if timeout is not None else request.timeout_seconds
    start_time = time.time()
    
    while True:
        # Check current status
        with _approval_lock:
            request = _approval_requests.get(approval_id)
        
        if not request:
            raise ValueError(f"Approval request disappeared: {approval_id}")
        
        # Check if resolved
        if request.status != ApprovalStatus.PENDING:
            return request
        
        # Check if expired
        if request.is_expired():
            resolve_approval(approval_id, approved=False)
            with _approval_lock:
                return _approval_requests.get(approval_id, request)
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= effective_timeout:
            resolve_approval(approval_id, approved=False)
            with _approval_lock:
                return _approval_requests.get(approval_id, request)
        
        # Wait before next poll
        time.sleep(poll_interval)


def register_approval_callback(
    approval_id: str,
    callback: Callable[[ApprovalRequest], None],
) -> None:
    """Register a callback to be notified when an approval is resolved."""
    with _approval_lock:
        if approval_id not in _approval_callbacks:
            _approval_callbacks[approval_id] = []
        _approval_callbacks[approval_id].append(callback)


def cleanup_old_requests(max_age_seconds: float = 3600) -> int:
    """
    Remove old resolved requests to prevent memory leaks.
    
    Args:
        max_age_seconds: Remove requests older than this.
    
    Returns:
        Number of requests cleaned up.
    """
    now = datetime.now(timezone.utc)
    to_remove = []
    
    with _approval_lock:
        for approval_id, request in _approval_requests.items():
            if request.status != ApprovalStatus.PENDING:
                age = (now - request.created_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(approval_id)
        
        for approval_id in to_remove:
            del _approval_requests[approval_id]
            _approval_callbacks.pop(approval_id, None)
    
    if to_remove:
        logger.debug("Cleaned up %d old approval requests", len(to_remove))
    
    return len(to_remove)
