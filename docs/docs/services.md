# Services

Archi supports several **services** — containerized applications that interact with the AI pipelines. Services are enabled at deploy time with the `--services` flag.

```bash
archi create [...] --services chatbot,uploader,grafana
```

List all available services with:

```bash
archi list-services
```

---

## Chat Interface

The primary user-facing service. Provides a web-based chat application for interacting with Archi's AI agents.

**Default port:** `7861`

### Key Features

- Streaming responses with tool-call visualization
- Agent selector dropdown for switching between agents
- Built-in [Data Viewer](data_sources.md#data-viewer) at `/data`
- Settings panel for model/provider selection
- [BYOK](models_providers.md#bring-your-own-key-byok) support
- Conversation history
- [Service Status Board & Alert Banners](#service-status-board--alert-banners)

### Configuration

```yaml
services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
    default_provider: local
    default_model: llama3.2
    trained_on: "Course documentation"
    hostname: "example.mit.edu"
    port: 7861
    external_port: 7861
```

### Running

```bash
archi create [...] --services chatbot
```

---

## Service Status Board & Alert Banners

The Service Status Board (SSB) is a built-in feature of the Chat Interface that lets designated operators communicate service health, planned downtime, known issues, and general announcements directly to all users — without external tooling.

### How It Works

**Alert banners** appear as colour-coded strips at the top of every page in the chat app. Up to 5 active alerts are displayed at once. Each banner can be individually dismissed by the user client-side. A **details** link redirects to the full status board.

The **Status Board** at `/ssb/status` provides:

- **Active Alerts** — non-expired alerts with severity badges, creator, and timestamp
- **Expired Alerts** — historical record shown at reduced opacity
- **Post New Alert form** — visible only to configured alert managers

### Severity Levels

| Severity | Colour | Intended Use |
|----------|--------|--------------|
| `alarm` | Red | Service outage or critical failure |
| `warning` | Amber | Degraded performance, elevated error rate |
| `news` | Blue | Release notes, planned maintenance |
| `info` | Slate | General informational notices |

### Creating and Deleting Alerts

Navigate to **Status** in the main chat header (or go to `/ssb/status` directly). The **Post New Alert** form is shown to users who have alert manager access. Fill in:

- **Message** (required) — short text shown in the banner
- **Severity** (required) — one of `alarm`, `warning`, `news`, `info`
- **Description** (optional) — longer explanation shown only on the status page
- **Expires at** (optional) — datetime after which the alert is hidden from banners; expired alerts remain visible in the status board history

To delete an alert, click the **Delete** button on its card on the status board. Deletion is permanent.

Alerts can also be created via the REST API:

```bash
curl -X POST http://localhost:7861/api/ssb/alerts \
  -H 'Content-Type: application/json' \
  -d '{
    "severity": "warning",
    "message": "Embedding pipeline running — responses may be slower than usual",
    "description": "Optional longer explanation shown on the status board.",
    "expires_in_hours": 4
  }'
```

Or with an explicit expiry timestamp:

```bash
curl -X POST http://localhost:7861/api/ssb/alerts \
  -H 'Content-Type: application/json' \
  -d '{
    "severity": "alarm",
    "message": "Model backend unavailable",
    "expires_at": "2026-02-21T18:00:00"
  }'
```

### API Endpoints

| Method | Route | Auth Required | Description |
|--------|-------|---------------|-------------|
| `GET` | `/ssb/status` | Any authenticated user | Render the status board page |
| `POST` | `/api/ssb/alerts` | Alert managers only | Create a new alert |
| `DELETE` | `/api/ssb/alerts/<id>` | Alert managers only | Delete an alert by ID |

### Access Control

Alert managers are configured via `services.chat_app.alerts.managers` (username list) or the `alerts:manage` RBAC permission. The rules are:

1. **Auth disabled** → everyone may create and delete alerts.
2. **Auth enabled** → a user is an alert manager if **either**:
    - their username is in the `alerts.managers` list, **or**
    - their session roles grant the `alerts:manage` permission.
3. **Auth enabled, no username match, no `alerts:manage` permission** → nobody may manage (safe default; a warning is logged).

All users can always *view* alerts and the status board regardless of access level.

```yaml
# Username-based access (backwards compatible):
services:
  chat_app:
    alerts:
      managers:
        - alice
        - bob

# Role-based access (can be combined with the above):
services:
  chat_app:
    auth:
      auth_roles:
        roles:
          ops-team:
            permissions:
              - alerts:manage
```

See [Configuration → `services.chat_app.alerts`](configuration.md#serviceschat_appalerts) for the full reference.

---

## Document Upload

Document upload is exposed in the chat UI and backed by the **Data Manager** service. Documents can be uploaded via the web interface or by copying files directly into the data directory.

See [Data Sources — Adding Documents Manually](data_sources.md#adding-documents-manually) for setup instructions.

---

## Data Manager

A background service that handles data ingestion, vectorstore management, and scheduled re-scraping. It is automatically started with most deployments.

**Default port:** `7871`

### Features

- Orchestrates all data collectors (links, git, JIRA, Redmine)
- Manages the vectorstore (chunking, embedding, indexing)
- Provides a scheduling system for periodic re-ingestion
- Exposes API endpoints for ingestion status and schedule management

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ingestion/status` | GET | Current ingestion progress |
| `/api/reload-schedules` | POST | Trigger schedule reload from database |
| `/api/schedules` | GET | Current schedule status |

### Configuration

```yaml
services:
  data_manager:
    port: 7871
    external_port: 7871
```

---

## Piazza Interface

Reads posts from a Piazza forum and posts draft responses to a specified Slack channel.

### Setup

1. Go to [Slack Apps](https://api.slack.com/apps) and sign in to your workspace.
2. Click **Create New App** → **From scratch**. Name the app and select the workspace.
3. Go to **Incoming Webhooks** under Features and toggle it on.
4. Click **Add New Webhook** and select the target channel.
5. Copy the **Webhook URL** to your secrets file.

### Configuration

Get the Piazza network ID from the class homepage URL (e.g., `https://piazza.com/class/m0g3v0ahsqm2lg` → `m0g3v0ahsqm2lg`).

```yaml
services:
  piazza:
    agent_class: QAPipeline
    provider: local
    model: llama3.2
    network_id: <your Piazza network ID>
  chat_app:
    trained_on: "Your class materials"
```

### Secrets

```bash
PIAZZA_EMAIL=...
PIAZZA_PASSWORD=...
SLACK_WEBHOOK=...
```

### Running

```bash
archi create [...] --services chatbot,piazza
```

---

## Redmine / Mailbox Interface

Reads new tickets in a Redmine project, drafts a response as a comment, and sends it as an email when the ticket is marked "Resolved" by an admin.

### Configuration

```yaml
services:
  redmine_mailbox:
    url: https://redmine.example.com
    project: my-project
    redmine_update_time: 10
    mailbox_update_time: 10
    answer_tag: "-- Archi -- Resolving email was sent"
```

### Secrets

```bash
IMAP_USER=...
IMAP_PW=...
REDMINE_USER=...
REDMINE_PW=...
SENDER_SERVER=...
SENDER_PORT=587
SENDER_REPLYTO=...
SENDER_USER=...
SENDER_PW=...
```

### Running

```bash
archi create [...] --services chatbot,redmine-mailer
```

---

## Jira Ticket Responder Service

Polls configured Jira projects for recently updated tickets in the configured eligible statuses, answers each issue trigger once, and posts the answer as a Jira comment for operators to approve. Answers can be public or role-restricted. The service can also answer Jira comments that explicitly mention the responder account when `respond_to_mentions` is enabled.

### Configuration

```yaml
services:
  jira_ticket_responder:
    url: https://its.cern.ch/jira/
    projects:
      - CMSTZ
      - CMSDM
    visible_to_role: Developers # Optional; omit to make answers public.
    poll_interval_minutes: 1  # Optional; defaults to 1.
    lookback_days: 7          # Optional; defaults to 7.
    respond_to_mentions: false # Optional; defaults to false.
    mention_allowed_roles:     # Optional; empty or omitted allows all roles.
      - Developers
    eligible_statuses:        # Optional; defaults to ["Open", "In Progress"].
      - Open
      - In Progress
```

The `jira_ticket_responder` service uses `services.jira_ticket_responder` only. Do not add `enabled`; process enablement is controlled by `--services jira_ticket_responder`.

### Behavior

- Each poll searches configured projects and `eligible_statuses` with a rolling Jira JQL window of `updated >= "-<lookback_days>d"`, so tickets updated while the service was down are still considered while they remain in the configured lookback window.
- The service stores Jira answer state in `jira_responder_triggers`. Issue answers use trigger key `issue:<ISSUE_KEY>`. Mention answers use trigger key `comment:<COMMENT_ID>`.
- When `respond_to_mentions` is `true`, the service fetches the 50 newest comments for each candidate issue, detects Jira Server/Data Center wiki mentions like `[~cmsai]` using the authenticated Jira service account `name`, and ignores comments authored by that service account.
- When `mention_allowed_roles` is non-empty, a mention is answered only when its author belongs to at least one configured Jira project role. Direct role membership and membership through a Jira group both count. Unknown roles and role lookup failures deny mention responses for the affected issue. Empty or omitted `mention_allowed_roles` allows every non-service-account author.
- `visible_to_role` controls Jira comment visibility independently of mention authorization. When it is a non-empty role name, answers are restricted to that one Jira role. When it is empty or omitted, answers are public to everyone who can view the issue. Jira supports only one role in a comment visibility restriction.
- Mention scanning does not apply a separate comment timestamp cutoff. If an issue is selected because it was recently updated, an older mention can be answered if it is still among the 50 newest comments and has no trigger row.
- Issue triggers and mention-comment triggers are independent. If both are eligible in the same poll, both can produce Jira comments.
- A trigger in `answering` is retried once after 60 seconds. If that retried trigger is still `answering` after 600 seconds, it is marked `failed`. `answered` and `failed` triggers are skipped by later polls.
- Jira comments include the Archi answer and, when Archi returns them, capped Jira wiki-rendered `{panel}` sections for reasoning trace and tool calls. The service uses standard Jira wiki panels and `{noformat}` blocks, not collapsible expand macros.
- The Jira comment is posted before conversation persistence. A trigger is marked `answered` only after Jira accepts the comment. If persistence fails after posting, the Jira comment remains and the trigger records `last_error`.
- In the rare case where Jira accepts a comment but the local `answered` transition cannot be confirmed, the trigger is marked `failed` with `response_comment_id` when available. Operators should not assume every `failed` row means no Jira comment was posted.
- Run only one Jira responder instance per Jira project and service account. The database state protects normal restarts and repeated polls; separate deployments using separate databases do not coordinate with each other.

### State Table

Fresh deployments create the state table from `init.sql`. Existing deployments create it during Jira responder startup with idempotent DDL. This is schema creation only; it does not backfill trigger rows for Jira comments that were already posted by earlier test or deployment runs. Operators can run the same SQL manually before starting the service if preferred:

```sql
CREATE TABLE IF NOT EXISTS jira_responder_triggers (
    trigger_key TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('issue','mention_comment')),
    issue_key TEXT NOT NULL,
    trigger_comment_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('answering','answered','failed')),
    retry_used BOOLEAN NOT NULL DEFAULT FALSE,
    last_error TEXT,
    conversation_id INTEGER REFERENCES conversation_metadata(conversation_id) ON DELETE SET NULL,
    response_comment_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_issue
    ON jira_responder_triggers(issue_key);

CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_status
    ON jira_responder_triggers(status, updated_at);
```

### Secrets

```bash
JIRA_TICKET_RESPONDER_PAT=...
PG_PASSWORD=...
# Add the API key required by the resolved Archi provider, such as OPENAI_API_KEY.
```

`JIRA_PAT` is used by the Jira data source for read-only ingestion. `JIRA_TICKET_RESPONDER_PAT` is used by the ticket responder service to browse issues and add comments. Use distinct Jira accounts for least privilege, and keep the ticket responder token tied to a dedicated account because the responder uses that account identity for comment posting and mention self-filtering.

Include any provider key required by the resolved Archi provider in the `.env` passed to `archi create` so it is copied into the deployment. Provider key validation is handled by Archi during agent startup.

### Running

```bash
archi create [...] --services chatbot,jira_ticket_responder
```

---

## Mattermost Interface

Reads posts from a Mattermost forum and posts draft responses to a specified channel.

### Configuration

```yaml
services:
  mattermost:
    update_time: 60
```

### Secrets

```bash
MATTERMOST_WEBHOOK=...
MATTERMOST_PAK=...
MATTERMOST_CHANNEL_ID_READ=...
MATTERMOST_CHANNEL_ID_WRITE=...
```

### Running

```bash
archi create [...] --services chatbot,mattermost
```

---

## Grafana Monitoring

Monitor system performance and LLM usage with a pre-configured Grafana dashboard.

**Default port:** `3000`

> **Note:** If redeploying with an existing name (without removing volumes), the PostgreSQL Grafana user may not have been created. Deploy a fresh instance to avoid issues.

### Configuration

```yaml
services:
  grafana:
    external_port: 3000
```

### Secrets

```bash
PG_PASSWORD=<your_database_password>
GRAFANA_PG_PASSWORD=<grafana_db_password>
```

### Running

```bash
archi create [...] --services chatbot,grafana
```

After deployment, access Grafana at `your-hostname:3000`. The default login is `admin`/`admin` — you'll be prompted to change the password on first login. Navigate to **Menu → Dashboards → Archi → Archi Usage** for the main dashboard.

> **Tip:** For the "Recent Conversation Messages" panel, click the three dots → **Edit** → find "Override 4" → enable **Cell value inspect** to expand long text entries. Click **Apply** to save.

---

## Grader Interface

An automated grading service for handwritten assignments with a web interface.

> **Note:** This service is experimental and not yet fully generalized.

### Requirements

The following files are needed:

- **`users.csv`**: Two columns — `MIT email` and `Unique code`
- **`solution_with_rubric_*.txt`**: One file per problem, named with the problem number. Begins with the problem name and a line of dashes.
- **`admin_password.txt`**: Admin code for resetting student attempts (passed as a secret).

### Configuration

```yaml
services:
  grader_app:
    provider: local
    model: llama3.2
    prompts:
      grading:
        final_grade_prompt: final_grade.prompt
      image_processing:
        image_processing_prompt: image_processing.prompt
    num_problems: 1
    local_rubric_dir: ~/grading/my_rubrics
    local_users_csv_dir: ~/grading/logins
  chat_app:
    trained_on: "rubrics, class info, etc."
```

### Secrets

```bash
ADMIN_PASSWORD=your_password
```

### Running

```bash
archi create [...] --services grader
```
