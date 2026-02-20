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
