-- create grafana user if it does not exist
DO
$do$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana') THEN
        CREATE USER grafana WITH PASSWORD '__GRAFANA_PG_PASSWORD__';
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
