-- create tables
CREATE TABLE IF NOT EXISTS configs (
    config_id SERIAL,
    config TEXT NOT NULL,
    config_name TEXT NOT NULL,
    PRIMARY KEY (config_id)
);
CREATE TABLE IF NOT EXISTS a2rchi_settings (
    settings_id SERIAL,
    a2rchi JSONB NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (settings_id)
);
CREATE TABLE IF NOT EXISTS a2rchi_model_options (
    name TEXT PRIMARY KEY,
    config JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS a2rchi_pipeline_options (
    name TEXT PRIMARY KEY,
    config JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS a2rchi_agent_options (
    name TEXT PRIMARY KEY,
    config JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS a2rchi_tool_options (
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (agent_name, tool_name)
);
CREATE TABLE IF NOT EXISTS a2rchi_mcp_server_options (
    name TEXT PRIMARY KEY,
    config JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id SERIAL,
    client_id TEXT,
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMP NOT NULL DEFAULT NOW(),
    a2rchi_version VARCHAR(50),
    PRIMARY KEY (conversation_id)
);
CREATE TABLE IF NOT EXISTS conversations (
    a2rchi_service TEXT NOT NULL,
    conversation_id INTEGER NOT NULL,
    message_id SERIAL,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    link TEXT NOT NULL,
    context TEXT NOT NULL,
    ts TIMESTAMP NOT NULL,
    conf_id INTEGER NOT NULL,
    PRIMARY KEY (message_id),
    FOREIGN KEY (conf_id) REFERENCES configs(config_id),
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS feedback (
    mid INTEGER NOT NULL,
    feedback_ts TIMESTAMP NOT NULL,
    feedback TEXT NOT NULL,
    feedback_msg TEXT,
    incorrect BOOLEAN,
    unhelpful BOOLEAN,
    inappropriate BOOLEAN,
    PRIMARY KEY (mid, feedback_ts),
    FOREIGN KEY (mid) REFERENCES conversations(message_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS timing (
    mid INTEGER NOT NULL,
    client_sent_msg_ts TIMESTAMP NOT NULL,
    server_received_msg_ts TIMESTAMP NOT NULL,
    lock_acquisition_ts TIMESTAMP NOT NULL,
    vectorstore_update_ts TIMESTAMP NOT NULL,
    query_convo_history_ts TIMESTAMP NOT NULL,
    chain_finished_ts TIMESTAMP NOT NULL,
    a2rchi_message_ts TIMESTAMP NOT NULL,
    insert_convo_ts TIMESTAMP NOT NULL,
    finish_call_ts TIMESTAMP NOT NULL,
    server_response_msg_ts TIMESTAMP NOT NULL,
    msg_duration INTERVAL SECOND NOT NULL,
    PRIMARY KEY (mid),
    FOREIGN KEY (mid) REFERENCES conversations(message_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id SERIAL,
    conversation_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_args JSONB,
    tool_result TEXT,
    ts TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id),
    FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES conversations(message_id) ON DELETE CASCADE
);

-- seed configs/settings
{% if configs %}
{% for cfg in configs %}
INSERT INTO configs (config, config_name)
VALUES ('{{ cfg.payload }}', '{{ cfg.name }}');
{% endfor %}
{% endif %}

{% if a2rchi_settings_json %}
INSERT INTO a2rchi_settings (a2rchi)
VALUES ('{{ a2rchi_settings_json }}'::jsonb);
{% endif %}

{% if a2rchi_model_options %}
INSERT INTO a2rchi_model_options (name, config)
VALUES
{% for opt in a2rchi_model_options %}
  ('{{ opt.name }}', '{{ opt.config }}'::jsonb){% if not loop.last %},{% endif %}
{% endfor %}
;
{% endif %}

{% if a2rchi_pipeline_options %}
INSERT INTO a2rchi_pipeline_options (name, config, enabled)
VALUES
{% for opt in a2rchi_pipeline_options %}
  ('{{ opt.name }}', '{{ opt.config }}'::jsonb, {{ 'true' if opt.enabled else 'false' }}){% if not loop.last %},{% endif %}
{% endfor %}
;
{% endif %}

{% if a2rchi_agent_options %}
INSERT INTO a2rchi_agent_options (name, config, enabled)
VALUES
{% for opt in a2rchi_agent_options %}
  ('{{ opt.name }}', '{{ opt.config }}'::jsonb, {{ 'true' if opt.enabled else 'false' }}){% if not loop.last %},{% endif %}
{% endfor %}
;
{% endif %}

{% if a2rchi_tool_options %}
INSERT INTO a2rchi_tool_options (agent_name, tool_name, enabled)
VALUES
{% for opt in a2rchi_tool_options %}
  ('{{ opt.agent_name }}', '{{ opt.tool_name }}', {{ 'true' if opt.enabled else 'false' }}){% if not loop.last %},{% endif %}
{% endfor %}
;
{% endif %}

{% if a2rchi_mcp_server_options %}
INSERT INTO a2rchi_mcp_server_options (name, config, enabled)
VALUES
{% for opt in a2rchi_mcp_server_options %}
  ('{{ opt.name }}', '{{ opt.config }}'::jsonb, {{ 'true' if opt.enabled else 'false' }}){% if not loop.last %},{% endif %}
{% endfor %}
;
{% endif %}

-- create grafana user if it does not exist
{% if use_grafana -%}
DO
$do$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana') THEN
        CREATE USER grafana WITH PASSWORD '{{ grafana_pg_password }}';
        GRANT USAGE ON SCHEMA public TO grafana;
        GRANT SELECT ON public.timing TO grafana;
        GRANT SELECT ON public.conversations TO grafana;
        GRANT SELECT ON public.conversation_metadata TO grafana;
        GRANT SELECT ON public.feedback TO grafana;
        GRANT SELECT ON public.configs TO grafana;
        GRANT SELECT ON public.agent_tool_calls TO grafana;
    END IF;
END
$do$;
{% endif %}
