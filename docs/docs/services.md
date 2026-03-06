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

Alert managers are configured via `services.chat_app.alerts.managers`. The rules are:

1. **Auth disabled** → everyone may create and delete alerts.
2. **Auth enabled, managers list present** → only the listed usernames may manage.
3. **Auth enabled, managers list absent or empty** → nobody may manage (safe default; a warning is logged).

All users can always *view* alerts and the status board regardless of access level.

```yaml
services:
  chat_app:
    alerts:
      managers:
        - alice
        - bob
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
