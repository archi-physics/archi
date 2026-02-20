"""SQL queries used by archi"""

# =============================================================================
# Conversation Queries
# =============================================================================

SQL_INSERT_CONVO = """
INSERT INTO conversations (
    archi_service, conversation_id, sender, content, link, context, ts,
    model_used, pipeline_used
)
VALUES %s
RETURNING message_id;
"""

SQL_INSERT_FEEDBACK = """
INSERT INTO feedback (
    mid, feedback_ts, feedback, feedback_msg, incorrect, unhelpful, inappropriate
)
VALUES (%s, %s, %s, %s, %s, %s, %s);
"""

SQL_DELETE_REACTION_FEEDBACK = """
DELETE FROM feedback
WHERE mid = %s AND feedback IN ('like', 'dislike');
"""

SQL_GET_REACTION_FEEDBACK = """
SELECT feedback FROM feedback
WHERE mid = %s AND feedback IN ('like', 'dislike')
ORDER BY feedback_ts DESC
LIMIT 1;
"""

SQL_QUERY_CONVO = """
SELECT sender, content
FROM conversations
WHERE conversation_id = %s
ORDER BY message_id ASC;
"""

SQL_QUERY_CONVO_WITH_FEEDBACK = """
SELECT c.sender,
       c.content,
       c.message_id,
       lf.feedback,
       COALESCE(cf.comment_count, 0) AS comment_count
FROM conversations c
LEFT JOIN (
    SELECT DISTINCT ON (mid)
        mid,
        feedback,
        feedback_ts
    FROM feedback
    WHERE feedback IN ('like', 'dislike')
    ORDER BY mid, feedback_ts DESC
) lf ON lf.mid = c.message_id
LEFT JOIN (
    SELECT mid,
           COUNT(*) AS comment_count
    FROM feedback
    WHERE feedback = 'comment'
    GROUP BY mid
) cf ON cf.mid = c.message_id
WHERE c.conversation_id = %s
ORDER BY c.message_id ASC;
"""

SQL_INSERT_TIMING = """
INSERT INTO timing (
    mid,
    client_sent_msg_ts,
    server_received_msg_ts,
    lock_acquisition_ts,
    vectorstore_update_ts,
    query_convo_history_ts,
    chain_finished_ts,
    archi_message_ts,
    insert_convo_ts,
    finish_call_ts,
    server_response_msg_ts,
    msg_duration
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# =============================================================================
# Conversation Metadata Queries
# =============================================================================

SQL_CREATE_CONVERSATION = """
INSERT INTO conversation_metadata (
    title, created_at, last_message_at, client_id, archi_version
)
VALUES (%s, %s, %s, %s, %s)
RETURNING conversation_id;
"""

SQL_UPSERT_CONVERSATION_METADATA = """
INSERT INTO conversation_metadata (
    conversation_id, title, created_at, last_message_at, client_id, archi_version
)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (conversation_id) DO UPDATE
SET last_message_at = EXCLUDED.last_message_at;
"""

SQL_UPDATE_CONVERSATION_TIMESTAMP = """
UPDATE conversation_metadata
SET last_message_at = %s
WHERE conversation_id = %s AND client_id = %s;
"""

SQL_LIST_CONVERSATIONS = """
SELECT conversation_id, title, created_at, last_message_at
FROM conversation_metadata
WHERE client_id = %s
ORDER BY last_message_at DESC
LIMIT %s;
"""

SQL_GET_CONVERSATION_METADATA = """
SELECT conversation_id, title, created_at, last_message_at
FROM conversation_metadata
WHERE conversation_id = %s AND client_id = %s;
"""

SQL_DELETE_CONVERSATION = """
DELETE FROM conversation_metadata
WHERE conversation_id = %s AND client_id = %s;
"""

# =============================================================================
# Tool Calls Queries
# =============================================================================

SQL_INSERT_TOOL_CALLS = """
INSERT INTO agent_tool_calls (
    conversation_id, message_id, step_number, tool_name, tool_args, tool_result, ts
)
VALUES %s;
"""

SQL_QUERY_TOOL_CALLS = """
SELECT step_number, tool_name, tool_args, tool_result, ts
FROM agent_tool_calls
WHERE message_id = %s
ORDER BY step_number ASC;
"""

# =============================================================================
# A/B Comparison Queries
# =============================================================================

SQL_INSERT_AB_COMPARISON = """
INSERT INTO ab_comparisons (
    conversation_id, user_prompt_mid, response_a_mid, response_b_mid, 
    model_a, pipeline_a, model_b, pipeline_b, is_config_a_first
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING comparison_id;
"""

SQL_UPDATE_AB_PREFERENCE = """
UPDATE ab_comparisons
SET preference = %s, preference_ts = %s
WHERE comparison_id = %s;
"""

SQL_GET_AB_COMPARISON = """
SELECT comparison_id, conversation_id, user_prompt_mid, response_a_mid, response_b_mid,
       model_a, pipeline_a, model_b, pipeline_b, 
       is_config_a_first, preference, preference_ts, created_at
FROM ab_comparisons
WHERE comparison_id = %s;
"""

SQL_GET_PENDING_AB_COMPARISON = """
SELECT comparison_id, conversation_id, user_prompt_mid, response_a_mid, response_b_mid,
       model_a, pipeline_a, model_b, pipeline_b,
       is_config_a_first, preference, preference_ts, created_at
FROM ab_comparisons
WHERE conversation_id = %s AND preference IS NULL
ORDER BY created_at DESC
LIMIT 1;
"""

SQL_DELETE_AB_COMPARISON = """
DELETE FROM ab_comparisons
WHERE comparison_id = %s;
"""

SQL_GET_AB_COMPARISONS_BY_CONVERSATION = """
SELECT comparison_id, conversation_id, user_prompt_mid, response_a_mid, response_b_mid,
       model_a, pipeline_a, model_b, pipeline_b,
       is_config_a_first, preference, preference_ts, created_at
FROM ab_comparisons
WHERE conversation_id = %s
ORDER BY created_at ASC;
"""

# =============================================================================
# Agent Trace Queries
# =============================================================================

SQL_CREATE_AGENT_TRACE = """
INSERT INTO agent_traces (
    trace_id, conversation_id, message_id, user_message_id, 
    config_id, pipeline_name, events, started_at, status
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

SQL_UPDATE_AGENT_TRACE = """
UPDATE agent_traces
SET events = %s,
    completed_at = %s,
    status = %s,
    message_id = COALESCE(%s, message_id),
    total_tool_calls = %s,
    total_duration_ms = %s,
    cancelled_by = %s,
    cancellation_reason = %s
WHERE trace_id = %s;
"""

SQL_GET_AGENT_TRACE = """
SELECT trace_id, conversation_id, message_id, user_message_id,
       config_id, pipeline_name, events, started_at, completed_at,
       status, total_tool_calls, total_tokens_used, total_duration_ms,
       cancelled_by, cancellation_reason, created_at
FROM agent_traces
WHERE trace_id = %s;
"""

SQL_GET_TRACE_BY_MESSAGE = """
SELECT trace_id, conversation_id, message_id, user_message_id,
       config_id, pipeline_name, events, started_at, completed_at,
       status, total_tool_calls, total_tokens_used, total_duration_ms,
       cancelled_by, cancellation_reason, created_at
FROM agent_traces
WHERE message_id = %s;
"""

SQL_GET_ACTIVE_TRACE = """
SELECT trace_id, conversation_id, message_id, user_message_id,
       config_id, pipeline_name, events, started_at, status
FROM agent_traces
WHERE conversation_id = %s AND status = 'running'
ORDER BY started_at DESC
LIMIT 1;
"""

SQL_CANCEL_ACTIVE_TRACES = """
UPDATE agent_traces
SET status = 'cancelled',
    completed_at = %s,
    cancelled_by = %s,
    cancellation_reason = %s
WHERE conversation_id = %s AND status = 'running';
"""
