# User Guide

## Overview

Archi supports various **data sources** as easy ways to ingest your data into the vector store databased used for document retrieval. These include:

- **Links lists (even behind SSO)**: automatically scrape and ingest documents from a list of URLs
- **Git scraping**: git mkdocs repositories
- **Ticketing systems**: JIRA, Redmine, Piazza
- **Local documents**

Additionally, Archi supports various **interfaces/services**, which are applications that interact with the RAG system. These include:

- **Chat interface**: a web-based chat application
- **Piazza integration**: read posts from Piazza and post draft responses to a Slack channel
- **Cleo/Redmine integration**: read emails and create tickets in Redmine
- **Mattermost integration**: read posts from Mattermost and post draft responses to a Mattermost channel
- **Grafana monitoring dashboard**: monitor system and LLM performance metrics
- **Document uploader**: web interface for uploading and managing documents
- **Grader**: automated grading service for assignments with web interface

Both data sources and interfaces/services are enabled via flags to the `archi create` command,
```bash
archi create [...] --services=chatbot,piazza,... --sources jira,redmine,...
```
The parameters of the services and sources are configured via the configuration file. See below for more details.

We support various **pipelines** (agent classes and classic pipelines) which are pre-defined sequences of operations that process user inputs and generate responses.
Each service may support a given pipeline. Agent prompts and tool subsets are defined in agent markdown files, while the agent class and default model are configured per service.
See the `Services` and `Pipelines` sections below for more details.
We support various **models** for both embeddings and LLMs, which can be run locally or accessed via APIs.
See the `Models` section below for more details.

Finally, we support various **retrievers** and **embedding techniques** for document retrieval.
These are configured via the configuration file.
See the `Vector Store` section below for more details.

### Agent tools (search + retrieval)

The chat agent can use a few built-in tools to locate evidence. These are internal capabilities of the chat service:

- **Metadata search**: find files by name/path/source metadata. Use free-text for partial matches, or exact filters with `key:value`.
  Example: `mz_dilepton.py` or `relative_path:full/path/to/mz_dilepton.py`.
- **Content search (grep)**: line-level search inside file contents; supports regex and context lines.
  Example: `timeout error` with `before=2` and `after=2`.
- **Document fetch**: pull full text for a specific file by hash (truncated with `max_chars`).
- **Vectorstore search**: semantic retrieval of relevant passages when you don't know exact keywords.

These tools are meant to be used together: search first, then fetch only the most relevant documents.

### Optional command line options

In addition to the required `--name`, `--config/--config-dir`, `--env-file`, and `--services` arguments, the `archi create` command accepts several useful flags:

1. **`--podman`**: Run the deployment with Podman instead of Docker.
2. **`--sources` / `-src`**: Enable additional ingestion sources (`git`, `sso`, `jira`, `redmine`, ...). Provide a comma-separated list.
3. **`--gpu-ids`**: Mount specific GPUs (`--gpu-ids all` or `--gpu-ids 0,1`). The legacy `--gpu` flag still works but maps to `all`.
4. **`--tag`**: Override the local image tag (defaults to `2000`). Handy when building multiple configurations side-by-side.
5. **`--agents` / `-a`**: Path to a directory of agent markdown files. At least one `*.md` file is required.

### Agent specs

Agent specs live in a directory you pass via `--agents`. Each `*.md` file must include:

- A top-level `# Name` heading
- A `## Tools` section with a bullet list of tool names
- A `## Prompt` section containing the inline system prompt

The service selects the first agent spec in lexicographic order.
The `tools` list enables a subset of tools defined by the agent class.

In the chat UI, use the agent dropdown in the header to switch the active agent, edit or delete existing agents, or add
a new one with the plus button. The selection is persisted and takes effect on the next request.
5. **`--hostmode`**: Use host networking for all services.
6. **`--verbosity` / `-v`**: Control CLI logging level (0 = quiet, 4 = debug).
7. **`--force`** / **`--dry-run`**: Force recreation of an existing deployment and/or show what would happen without actually deploying.

You can inspect the available services and sources, together with descriptions, using `archi list-services`.
The CLI checks that host ports are free before deploying; if a port is already in use, adjust `services.*.external_port` (or `services.*.port` in `--hostmode`) and retry.

> **GPU helpers**
>
> GPU access requires the NVIDIA drivers plus the NVIDIA Container Toolkit. After installing the toolkit, generate CDI entries (for Podman) with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` and confirm with `nvidia-ctk cdi list`. Docker users should run `sudo nvidia-ctk runtime configure --runtime=docker`.

---

## Data Sources

These are the different ways to ingest data into the vector store used for document retrieval.

### Web Link Lists

A web link list is a simple text file containing a list of URLs, one per line.
Archi will fetch the content from each URL and add it to the vector store, using the `Scraper` class.

#### Configuration

You can define which lists of links Archi will ingest in the configuration file as follows:
```yaml
data_manager:
  sources:
    links:
      input_lists:  # REQUIRED
        - miscellanea.list  # list of websites with relevant info
        - [...other lists...]
```

Each list should be a simple text file containing one URL per line, e.g.,
```
https://example.com/page1
https://example.com/page2
[...]
```

In the case that some of the links are behind a Single Sign-On (SSO) system, enable the SSO source in your configuration and specify the collector class:
```yaml
data_manager:
  sources:
    sso:
      enabled: true
    links:
      selenium_scraper:
        enabled: true
        selenium_class: CERNSSOScraper  # or whichever class is appropriate
        selenium_class_map:
          CERNSSOScraper:
            kwargs:
              headless: true
              max_depth: 2
```
Then, run `archi create ... --sources sso` to activate the SSO collector.

Note: source configuration is persisted to PostgreSQL `static_config` at deployment time and used at runtime.

You can customise the HTTP scraper behaviour (for example, to avoid SSL verification warnings):
```yaml
data_manager:
  sources:
    links:
      scraper:
        reset_data: true
        verify_urls: false
        enable_warnings: false
```

#### Secrets

If you are using SSO, depending on the class, you may need to provide your login credentials in a secrets file as follows:
```bash
SSO_USERNAME=username
SSO_PASSWORD=password
```
Then, make sure that the links you provide in the `.list` file(s) start with `sso-`, e.g.,
```
sso-https://example.com/protected/page
```

#### Running

Link scraping is automatically enabled in Archi, you don't need to add any arguments to the `create` command unless the links are sso protected.

---

### Git scraping

In some cases, the RAG input may be documentations based on MKDocs git repositories.
Instead of scraping these sites as regular HTML sites you can obtain the relevant content using the `GitScraper` class.

#### Configuration

To configure it, enable the git source in the configuration file:
```yaml
data_manager:
  sources:
    git:
      enabled: true
```
In the input lists, make sure to prepend `git-` to the URL of the **repositories** you are interested in scraping.
```
git-https://github.com/example/mkdocs/documentation.git
```

#### Secrets

You will need to provide a git username and token in the secrets file,
```bash
GIT_USERNAME=your_username
GIT_TOKEN=your_token
```

#### Running

Enable the git source during deployment with `--sources git`.

---

### JIRA

The JIRA integration allows Archi to fetch issues and comments from specified JIRA projects and add them to the vector store, using the `JiraClient` class.

#### Configuration

Select which projects to scrape in the configuration file:
```yaml
data_manager:
  sources:
    jira:
      url: https://jira.example.com
      projects:
        - PROJECT_KEY
      anonymize_data: true
      cutoff_date: "2023-01-01"
```

You can further customise anonymisation via the global anonymiser settings.
```yaml
data_manager:
  utils:
    anonymizer:
      nlp_model: en_core_web_sm
      excluded_words:
        - Example
      greeting_patterns:
        - '^(hi|hello|hey|greetings|dear)\b'
      signoff_patterns:
        - '\b(regards|sincerely|best regards|cheers|thank you)\b'
      email_pattern: '[\w\.-]+@[\w\.-]+\.\w+'
      username_pattern: '\[~[^\]]+\]'
```

The anonymizer will remove names, emails, usernames, greetings, signoffs, and any other words you specify from the fetched data.
This is useful if you want to avoid having personal information in the vector store.
The optional `cutoff_date` can be used to skip tickets created before a specified ISO-8601 date.

#### Secrets

A personal access token (PAT) is required to authenticate and authorize with JIRA.
Add `JIRA_PAT=<token>` to your `.env` file before deploying with `--sources jira`.

#### Running

Enable the source at deploy time with:
```bash
archi create [...] --services=chatbot --sources jira
```

---

### Adding Documents and the Uploader Interface

#### Adding Documents

There are two main ways to add documents to Archi's vector database. They are:

- Manually adding files while the service is running via the uploader GUI
- Directly copying files into the container

These methods are outlined below.

#### Manual Uploader

In order to upload documents while Archi is running via an easily accessible GUI, enable the uploader service when creating the deployment:
```bash
archi create [...] --services=chatbot,uploader
```
The exact port may vary based on configuration (default external port is `5003`).
A quick `podman ps` or `docker ps` will show which port is exposed.

In order to access the manager, you must first create an admin account. Grab the container ID with `podman ps`/`docker ps` and then enter the container:
```
docker exec -it <CONTAINER-ID> bash
```
Run the bundled helper:
```
python -u src/bin/service_create_account.py
```
from the `/root/Archi` directory inside the container. This script will guide you through creating an account; never reuse sensitive passwords here.

Once you have created an account, visit the outgoing port of the data manager docker service and then log in.
The GUI will then allow you to upload documents while Archi is still running. Note that it may take a few minutes for all the documents to upload.

#### Directly copying files to the container

The documents used for RAG live in the chat container at `/root/data/<directory>/<files>`. Thus, in a pinch, you can `docker/podman cp` a file at this directory level, e.g., `podman/docker cp myfile.pdf <container name or ID>:/root/data/<new_dir>/`. If you need to make a new directory in the container, you can do `podman exec -it <container name or ID> mkdir /root/data/<new_dir>`.

#### Data Viewer

The chat interface includes a built-in Data Viewer for browsing and managing ingested documents. Access it at `/data` on your chat app (e.g., `http://localhost:7861/data`).

**Features:**

- **Browse documents**: View all ingested documents with metadata (source, file type, chunk count)
- **Search and filter**: Filter documents by name or source type
- **View content**: Click on a document to see its full content and individual chunks
- **Enable/disable documents**: Toggle whether specific documents are included in RAG retrieval
- **Bulk operations**: Enable or disable multiple documents at once

**Document States:**

| State | Description |
|-------|-------------|
| Enabled | Document chunks are included in retrieval (default) |
| Disabled | Document is excluded from retrieval but remains in the database |

Disabling documents is useful for:
- Temporarily excluding outdated content
- Testing retrieval with specific document subsets
- Hiding sensitive documents from certain users

---

### Redmine

Use the Redmine source to ingest solved tickets (question/answer pairs) into the vector store.

#### Configuration

```yaml
data_manager:
  sources:
    redmine:
      url: https://redmine.example.com
      project: my-project
      anonymize_data: true
```

#### Secrets

Add the following to your `.env` file:
```bash
REDMINE_USER=...
REDMINE_PW=...
```

#### Running

Enable the source at deploy time with:
```bash
archi create [...] --services=chatbot --sources redmine
```

> To automate email replies, also enable the `redmine-mailer` service (see the Services section below).

---

## Interfaces/Services

These are the different apps that Archi supports, which allow you to interact with the AI pipelines.

### Piazza Interface

Set up Archi to read posts from your Piazza forum and post draft responses to a specified Slack channel. To do this, a Piazza login (email and password) is required, plus the network ID of your Piazza channel, and lastly, a Webhook for the slack channel Archi will post to. See below for a step-by-step description of this.

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps) and sign in to workspace where you will eventually want Archi to post to (note doing this in a business workspace like the MIT one will require approval of the app/bot).
2. Click 'Create New App', and then 'From scratch'. Name your app and again select the correct workspace. Then hit 'Create App'
3. Now you have your app, and there are a few things to configure before you can launch Archi:
4. Go to Incoming Webhooks under Features, and toggle it on.
5. Click 'Add New Webhook', and select the channel you want Archi to post to.
6. Now, copy the 'Webhook URL' and paste it into the secrets file, and handle it like any other secret!

#### Configuration

Beyond standard required configuration fields, the network ID of the Piazza channel is required (see below for an example config). You can get the network ID by simply navigating to the class homepage, and grabbing the sequence that follows 'https://piazza.com/class/'. For example, the 8.01 Fall 2024 homepage is: 'https://piazza.com/class/m0g3v0ahsqm2lg'. The network ID is thus 'm0g3v0ahsqm2lg'.

Example minimal config for the Piazza interface:

```yaml
name: bare_minimum_configuration #REQUIRED

data_manager:
  sources:
    links:
      input_lists:
        - class_info.list # class info links

services:
  piazza:
    agent_class: QAPipeline
    provider: local
    model: llama3.2
    network_id: <your Piazza network ID here> # REQUIRED
  chat_app:
    trained_on: "Your class materials" # REQUIRED
```

#### Secrets

The necessary secrets for deploying the Piazza service are the following:

```bash
PIAZZA_EMAIL=...
PIAZZA_PASSWORD=...
SLACK_WEBHOOK=...
```

The Slack webhook secret is described above. The Piazza email and password should be those of one of the class instructors. Remember to put this information in files named following what is written above.

#### Running

To run the Piazza service, simply add the piazza flag. For example:

```bash
archi create [...] --services=chatbot,piazza
```

---

### Redmine/Mailbox Interface

Archi will read all new tickets in a Redmine project, and draft a response as a comment to the ticket.
Once the ticket is updated to the "Resolved" status by an admin, Archi will send the response as an email to the user who opened the ticket.
The admin can modify Archi's response before sending it out.

#### Configuration

```yaml
services:
  redmine_mailbox:
    url: https://redmine.example.com
    project: my-project
    redmine_update_time: 10
    mailbox_update_time: 10
    answer_tag: "-- Archi -- Resolving email was sent"
```

#### Secrets

Add the following secrets to your `.env` file:
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

#### Running

```bash
archi create [...] --services=chatbot,redmine-mailer
```

---

### Mattermost Interface

Set up Archi to read posts from your Mattermost forum and post draft responses to a specified Mattermost channel.

#### Configuration

```yaml
services:
  mattermost:
    update_time: 60
```

#### Secrets

You need to specify a webhook, access token, and channel identifiers:
```bash
MATTERMOST_WEBHOOK=...
MATTERMOST_PAK=...
MATTERMOST_CHANNEL_ID_READ=...
MATTERMOST_CHANNEL_ID_WRITE=...
```

#### Running

To run the Mattermost service, include it when selecting services. For example:
```bash
archi create [...] --services=chatbot,mattermost
```

---

### Grafana Interface

Monitor the performance of your Archi instance with the Grafana interface. This service provides a web-based dashboard to visualize various metrics related to system performance, LLM usage, and more.

> Note, if you are deploying a version of Archi you have already used (i.e., you haven't removed the images/volumes for a given `--name`), the postgres will have already been created without the Grafana user created, and it will not work, so make sure to deploy a fresh instance.

#### Configuration

```yaml
services:
  grafana:
    external_port: 3000
```

#### Secrets

Grafana shares the Postgres database with other services, so you need both the database password and a Grafana-specific password:
```bash
PG_PASSWORD=<your_database_password>
GRAFANA_PG_PASSWORD=<grafana_db_password>
```

#### Running

Deploy Grafana alongside your other services:
```bash
archi create [...] --services=chatbot,grafana
```
and you should see something like this
```
CONTAINER ID  IMAGE                                     COMMAND               CREATED        STATUS                  PORTS                             NAMES
87f1c7289d29  docker.io/library/postgres:17             postgres              9 minutes ago  Up 9 minutes (healthy)  5432/tcp                          postgres-gtesting2
40130e8e23de  docker.io/library/grafana-gtesting2:2000                        9 minutes ago  Up 9 minutes            0.0.0.0:3000->3000/tcp, 3000/tcp  grafana-gtesting2
d6ce8a149439  localhost/chat-gtesting2:2000             python -u archi/...  9 minutes ago  Up 9 minutes            0.0.0.0:7861->7861/tcp            chat-gtesting2
```
where the grafana interface is accessible at `your-hostname:3000`. To change the external port from `3000`, you can do this in the config at `services.grafana.external_port`. The default login and password are both "admin", which you will be prompted to change should you want to after first logging in. Navigate to the Archi dashboard from the home page by going to the menu > Dashboards > Archi > Archi Usage. Note, `your-hostname` here is the just name of the machine. Grafana uses its default configuration which is `localhost` but unlike the chat interface, there are no APIs where we template with a selected hostname, so the container networking handles this nicely.

> Pro tip: once at the web interface, for the "Recent Conversation Messages (Clean Text + Link)" panel, click the three little dots in the top right hand corner of the panel, click "Edit", and on the right, go to e.g., "Override 4" (should have Fields with name: clean text, also Override 7 for context column) and override property "Cell options > Cell value inspect". This will allow you to expand the text boxes with messages longer than can fit. Make sure you click apply to keep the changes.

> Pro tip 2: If you want to download all of the information from any panel as a CSV, go to the same three dots and click "Inspect", and you should see the option.

---

### Grader Interface

Interface to launch a website which for a provided solution and rubric (and a couple of other things detailed below), will grade scanned images of a handwritten solution for the specified problem(s).

> Nota bene: this is not yet fully generalized and "service" ready, but instead for testing grading pipelines and a base off of which to build a potential grading app.

#### Requirements

To launch the service the following files are required:

- `users.csv`. This file is .csv file that contains two columns: "MIT email" and "Unique code", e.g.:

```
MIT email,Unique code
username@mit.edu,222
```

For now, the system requires the emails to be in the MIT domain, namely, contain "@mit.edu". TODO: make this an argument that is passed (e.g., school/email domain)

- `solution_with_rubric_*.txt`. These are .txt files that contain the problem solution followed by the rubric. The naming of the files should follow exactly, where the `*` is the problem number. There should be one of these files for every problem you want the app to be able to grade. The top of the file should be the problem name with a line of dashes ("-") below, e.g.:

```
Anti-Helmholtz Coils
---------------------------------------------------
```

These files should live in a directory which you will pass to the config, and Archi will handle the rest.

- `admin_password.txt`. This file will be passed as a secret and be the admin code to login in to the page where you can reset attempts for students.

#### Secrets

The only grading specific secret is the admin password, which like shown above, should be put in the following file

```bash
ADMIN_PASSWORD=your_password
```

Then it behaves like any other secret.

#### Configuration

The required fields in the configuration file are different from the rest of the Archi services. Below is an example:

```yaml
name: grading_test # REQUIRED

services:
  chat_app:
    trained_on: "rubrics, class info, etc." # REQUIRED
  grader_app:
    provider: local
    model: llama3.2
    prompts:
      grading:
        final_grade_prompt: final_grade.prompt
      image_processing:
        image_processing_prompt: image_processing.prompt
    num_problems: 1 # REQUIRED
    local_rubric_dir: ~/grading/my_rubrics # REQUIRED
    local_users_csv_dir: ~/grading/logins # REQUIRED

data_manager:
  [...]
```

1. `name` -- The name of your configuration (required).
2. `services.grader_app.provider` -- Provider for grading/image pipelines (e.g., `local`, `openai`).
3. `services.grader_app.model` -- Default model id for grading/image pipelines.
4. `services.grader_app.prompts.grading.final_grade_prompt` -- Path to the grading prompt file for evaluating student solutions.
5. `services.grader_app.prompts.image_processing.image_processing_prompt` -- Path to the prompt file for image processing.
6. `services.chat_app.trained_on` -- A brief description of the data or materials Archi is trained on (required).
7. `services.grader_app.num_problems` -- Number of problems the grading service should expect (must match the number of rubric files).
8. `services.grader_app.local_rubric_dir` -- Directory containing the `solution_with_rubric_*.txt` files.
9. `services.grader_app.local_users_csv_dir` -- Directory containing the `users.csv` file.

#### Running

```bash
archi create [...] --services=grader
```

---

## Models

Models are either:

1. Hosted locally, either via VLLM or HuggingFace transformers.
2. Accessed via an API, e.g., OpenAI, Anthropic, etc.
3. Accessed via an Ollama server instance.

### Local Models

To use a local model, specify one of the local model classes in `models.py`:

- `HuggingFaceOpenLLM`
- `HuggingFaceImageLLM`
- `VLLM`

### Models via APIs

We support the following model classes in `models.py` for models accessed via APIs:

- `OpenAILLM`
- `OpenRouterLLM`
- `AnthropicLLM`

#### OpenRouter

OpenRouter uses the OpenAI-compatible API. No config entry is required for API-key providers; if
`OPENROUTER_API_KEY` is set, the provider appears in the chat UI. To make OpenRouter the default,
set `services.chat_app.provider` and `services.chat_app.model` accordingly.

### Ollama

Configure Ollama under `services.chat_app.providers.local`, and set `services.chat_app.provider: local` with
`services.chat_app.model` to the model id you want as the default. Example:

```yaml
services:
  chat_app:
    provider: local
    model: llama3.2
    providers:
      local:
        base_url: http://localhost:11434
        mode: ollama
        models:
          - llama3.2
```

### Bring Your Own Key (BYOK)

Archi supports Bring Your Own Key (BYOK), allowing users to provide their own API keys for LLM providers at runtime. This enables:

- **Cost attribution**: Users pay for their own API usage
- **Provider flexibility**: Switch between providers without admin intervention
- **Privacy**: Use personal accounts for sensitive queries

#### Key Hierarchy

API keys are resolved in the following order (highest priority first):

1. **Environment Variables**: Admin-configured keys (e.g., `OPENAI_API_KEY`)
2. **Docker Secrets**: Keys mounted at `/run/secrets/`
3. **Session Storage**: User-provided keys via the Settings UI

!!! note
    Environment variable keys always take precedence. If an admin configures a key via environment variable, users cannot override it with their own key.

#### Using BYOK in the Chat Interface

1. Open the **Settings** modal (gear icon)
2. Expand the **API Keys** section
3. For each provider you want to use:
   - Enter your API key in the input field
   - Click **Save** to store it in your session
4. Select your preferred **Provider** and **Model** from the dropdowns
5. Start chatting!

**Status Indicators:**

| Icon | Meaning |
|------|---------|
| ✓ Env | Key configured via environment variable (cannot be changed) |
| ✓ Session | Key configured via your session |
| ○ | No key configured |

#### Supported Providers

| Provider | Environment Variable | API Key Format |
|----------|---------------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `sk-...` |
| Anthropic | `ANTHROPIC_API_KEY` | `sk-ant-...` |
| Google Gemini | `GOOGLE_API_KEY` | `AIza...` |
| OpenRouter | `OPENROUTER_API_KEY` | `sk-or-...` |

#### Security Considerations

- **Keys are never logged** - API keys are redacted from all log output
- **Keys are never echoed** - The UI only shows masked placeholders
- **Session-scoped** - Keys are cleared when you log out or your session expires
- **HTTPS recommended** - For production deployments, always use HTTPS to protect keys in transit

#### API Endpoints

For programmatic access, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/providers/keys` | GET | Get status of all provider keys |
| `/api/providers/keys/set` | POST | Set a session API key (validates before storing) |
| `/api/providers/keys/clear` | POST | Clear a session API key |

---

## Vector Store

The vector store is a database that stores document embeddings, enabling semantic and/or lexical search over your knowledge base. Archi uses PostgreSQL with pgvector as the default vector store backend to index and retrieve relevant documents based on similarity to user queries.

### Backend Selection

Archi uses PostgreSQL with the pgvector extension as its vector store backend. This provides production-grade vector similarity search integrated with your existing PostgreSQL database.

Configure vector store settings in your configuration file:

```yaml
services:
  vectorstore:
    backend: postgres  # PostgreSQL with pgvector (only supported backend)
```

### Configuration

Vector store settings are configured under the `data_manager` section:

```yaml
data_manager:
  collection_name: default_collection
  embedding_name: OpenAIEmbeddings
  chunk_size: 1000
  chunk_overlap: 0
  reset_collection: true
  num_documents_to_retrieve: 5
  distance_metric: cosine
```

#### Core Settings

- **`collection_name`**: Name of the vector store collection. Default: `default_collection`
- **`chunk_size`**: Maximum size of text chunks (in characters) when splitting documents. Default: `1000`
- **`chunk_overlap`**: Number of overlapping characters between consecutive chunks. Default: `0`
- **`reset_collection`**: If `true`, deletes and recreates the collection on startup. Default: `true`
- **`num_documents_to_retrieve`**: Number of relevant document chunks to retrieve for each query. Default: `5`

#### Distance Metrics

The `distance_metric` determines how similarity is calculated between embeddings:

- **`cosine`**: Cosine similarity (default) - measures the angle between vectors
- **`l2`**: Euclidean distance - measures straight-line distance
- **`ip`**: Inner product - measures dot product similarity

```yaml
data_manager:
  distance_metric: cosine  # Options: cosine, l2, ip
```

### Embedding Models

Embeddings convert text into numerical vectors. Archi supports multiple embedding providers:

#### OpenAI Embeddings

```yaml
data_manager:
  embedding_name: OpenAIEmbeddings
  embedding_class_map:
    OpenAIEmbeddings:
      class: OpenAIEmbeddings
      kwargs:
        model: text-embedding-3-small
      similarity_score_reference: 10
```

#### HuggingFace Embeddings

```yaml
data_manager:
  embedding_name: HuggingFaceEmbeddings
  embedding_class_map:
    HuggingFaceEmbeddings:
      class: HuggingFaceEmbeddings
      kwargs:
        model_name: sentence-transformers/all-MiniLM-L6-v2
        model_kwargs:
          device: cpu
        encode_kwargs:
          normalize_embeddings: true
      similarity_score_reference: 10
      query_embedding_instructions: null
```

### Supported Document Formats

The vector store can process the following file types:

- **Text files**: `.txt`, `.C`
- **Markdown**: `.md`
- **Python**: `.py`
- **HTML**: `.html`
- **PDF**: `.pdf`

Documents are automatically loaded with the appropriate parser based on file extension.

### Document Synchronization

Archi automatically synchronizes your data directory with the vector store:

1. **Adding documents**: New files in the data directory are automatically chunked, embedded, and added to the collection
2. **Removing documents**: Files deleted from the data directory are removed from the collection
3. **Source tracking**: Each ingested artifact is recorded in the Postgres catalog (`resources` table) with its resource hash and relative file path

### Hybrid Search

Combine semantic search with keyword-based BM25 search for improved retrieval:

```yaml
data_manager:
  use_hybrid_search: true
  bm25_weight: 0.6
  semantic_weight: 0.4
```

- **`use_hybrid_search`**: Enable hybrid search combining BM25 and semantic similarity. Default: `true`
- **`bm25_weight`**: Weight for BM25 keyword scores (base config default: `0.6`).
- **`semantic_weight`**: Weight for semantic similarity scores (base config default: `0.4`).
- **BM25 tuning**: Parameters like `k1` and `b` are set when the PostgreSQL BM25 index is created and are no longer configurable via this file.

### Stemming

By specifying the stemming option within your configuration, stemming functionality for the documents in Archi will be enabled. By doing so, documents inserted into the retrieval pipeline, as well as the query that is matched with them, will be stemmed and simplified for faster and more accurate lookup.

```yaml
data_manager:
  stemming:
    enabled: true
```

When enabled, both documents and queries are processed using the Porter Stemmer algorithm to reduce words to their root forms (e.g., "running" → "run"), improving matching accuracy.

### PostgreSQL Backend (Default)

Archi uses PostgreSQL with pgvector for vector storage by default. The PostgreSQL service is automatically started when you deploy with the chatbot service.

```yaml
services:
  postgres:
    host: postgres
    port: 5432
    database: archi
  vectorstore:
    backend: postgres
```

Required secrets for PostgreSQL:
```bash
PG_PASSWORD=your_secure_password
```

---

## Benchmarking

Archi has benchmarking functionality provided by the `evaluate` CLI command. We currently support two modes:

1. `SOURCES`: given a user question and a list of correct sources, check if the retrieved documents contain any of the correct sources.
2. `RAGAS`: use the Ragas RAG evaluator module to return numerical values judging by 4 of their provided metrics the quality of the answer: `answer_relevancy`, `faithfulness`, `context precision`, and `context relevancy`.

### Preparing the queries file

Provide your list of questions, answers, and relevant sources in JSON format as follows:

```json
[
  {
    "question": "",
    "sources": [...],
    "answer": ""
    // (optional)
    "sources_match_field": [...]
  },
  ...
]
```

Explanation of fields:
- `question`: The question to be answered by the Archi instance.
- `sources`: A list of sources (e.g., URLs, ticket IDs) that contain the answer. They are identified via the `sources_match_field`, which must be one of the metadata fields of the documents in your vector store.
- `answer`: The expected answer to the question, used for evaluation.
- `sources_match_field` (optional): A list of metadata fields to match the sources against (e.g., `url`, `ticket_id`). If not provided, defaults to what is in the configuration file under `data_manager:services:benchmarking:mode_settings:sources:default_match_field`.

Example: (see also `examples/benchmarking/queries.json`)
```json 
[
  {
    "question": "Does Jorian Benke work with the PPC and what topic will she work on?",
    "sources": ["https://ppc.mit.edu/blog/2025/07/14/welcome-our-first-ever-in-house-masters-student/", "CMSPROD-42"],
    "answer": "Yes, Jorian works with the PPC and her topic is the study of Lorentz invariance.",
    "source_match_field": ["url", "ticket_id"]
  },
  ...
]
```
N.B.: one could also provide the `url` for the JIRA ticket: it is just a choice that you must make, and detail in `source_match_field`. i.e., the following will evaluate equivalently as the above example:
```json 
[
  {
    "question": "Does Jorian Benke work with the PPC and what topic will she work on?",
    "sources": ["https://ppc.mit.edu/blog/2025/07/14/welcome-our-first-ever-in-house-masters-student/", "https://its.cern.ch/jira/browse/CMSPROD-42"],
    "answer": "Yes, Jorian works with the PPC and her topic is the study of Lorentz invariance.",
    "source_match_field": ["url", "url"]
  },
  ...
]
```

### Configuration

You can evaluate one or more configurations by specifying the `evaluate` command with the `-cd` flag pointing to the directory containing your configuration file(s). You can also specify individual files with the `-c` flag. This can be useful if you're interested in comparing different hyperparameter settings.

We support two modes, which you can specify in the configuration file under `services:benchmarking:modes`. You can choose either or both of `RAGAS` and `SOURCES`.

The RAGAS mode will use the Ragas RAG evaluator module to return numerical values judging by 4 of their provided metrics: `answer_relevancy`, `faithfulness`, `context precision`, and `context relevancy`. More information about these metrics can be found on the [Ragas website](https://docs.ragas.io/en/stable/concepts/metrics/). 

The SOURCES mode will check if the retrieved documents contain any of the correct sources. The matching is done by comparing a given metadata field for any source. The default is `file_name`, as per the configuration file (`data_manager:services:benchmarking:mode_settings:sources:default_match_field`). You can override this on a per-query basis by specifying the `sources_match_field` field in the queries file, as described above.

The configuration file should look like the following:

```yaml
services:
  benchmarking:
    queries_path: examples/benchmarking/queries.json
    out_dir: bench_out
    modes:
      - "RAGAS"
      - "SOURCES"
    mode_settings:
      sources:
        default_match_field: ["file_name"] # default field to match sources against, can be overridden in the queries file
      ragas_settings:
        provider: <provider name> # can be one of OpenAI, HuggingFace, Ollama, and Anthropic
        evaluation_model_settings:
          model_name: <model name> # ensure this lines up with the langchain API name for your chosen model and provider
          base_url: <url> # address to your running Ollama server should you have chosen the Ollama provider
        embedding_model: <embedding provider> # OpenAI or HuggingFace
```

Finally, before you run the command ensure `out_dir`, the output directory, both exists on your system and that the path is correctly specified so that results can show up inside of it.

### Running

To run the benchmarking script simply run the following:

``` bash
archi evaluate -n <name> -e <env_file> -cd <configs_directory> <optionally use  -c <file1>,<file2>, ...> <OPTIONS>
```

Example:
```bash
archi evaluate -n benchmark -c examples/benchmarking/benchmark_configs/example_conf.yaml --gpu-ids all
```

### Additional options

You might also want to adjust the `timeout` setting, which is the upper limit on how long the Ragas evaluation takes on a single QA pair, or the `batch_size`, which determines how many QA pairs to evaluate at once, which you might want to adjust, e.g., based on hardware constraints, as Ragas doesn't pay great attention to that. The corresponding configuration options are similarly set for the benchmarking services, as follows:

```yaml
services:
  benchmarking:
    timeout: <time in seconds> # default is 180
    batch_size: <desired batch size> # no default setting, set by Ragas...
```

### Results

The output of the benchmarking will be saved in the `out_dir` specified in the configuration file. The results will be saved in a timestamped subdirectory, e.g., `bench_out/2042-10-01_12-00-00/`.

To later examine your data, check out `scripts/benchmarking/`, which contains some plotting functions and an ipynotebook with some basic usage examples. This is useful to play around with the results of the benchmarking, we will soon also have instead dedicated scripts to produce the plots of interest.

---

## Other

Some useful additional features supported by the framework.

---
## Configuration Management

Archi uses a three-tier configuration system that allows flexibility at different levels:

### Configuration Hierarchy

1. **Static Configuration** (deploy-time, immutable)
   - Set during deployment from config.yaml
   - Includes: deployment name, embedding model, available pipelines/models
   - Cannot be changed at runtime

2. **Dynamic Configuration** (admin-controlled)
   - Runtime-modifiable settings
   - Includes: default model, temperature, retrieval parameters, active prompts
   - Only admins can modify via API

3. **User Preferences** (per-user overrides)
   - Individual users can override certain settings
   - Includes: preferred model, temperature, prompt selections
   - Takes precedence over dynamic config for that user

### Effective Configuration

When a request is made, Archi resolves the effective value for each setting:

```
User Preference (if set) → Dynamic Config (admin default) → Static Default
```

For example, if an admin sets `temperature: 0.7` but a user prefers `0.5`, 
that user's requests will use `0.5`.

### Admin-Only Settings

The following settings require admin privileges to modify:

- `active_pipeline` - Default pipeline
- `active_model` - Default model
- `temperature`, `max_tokens`, `top_p`, `top_k` - Generation parameters
- `num_documents_to_retrieve` - Retrieval settings
- `active_condense_prompt`, `active_chat_prompt`, `active_system_prompt` - Default prompts
- `verbosity` - Logging level

### User-Overridable Settings

Users can personalize these settings via the preferences API:

- `preferred_model` - Model selection
- `preferred_temperature` - Generation temperature
- `preferred_max_tokens` - Max response length
- `preferred_num_documents` - Number of documents to retrieve
- `preferred_condense_prompt`, `preferred_chat_prompt`, `preferred_system_prompt` - Prompt selections
- `theme` - UI theme

---

## Prompt Customization

Archi supports customizable prompts organized by type. Prompts are stored as files in your deployment directory for easy editing and version control.

### Prompt Location

After deployment, prompts are located at:
```
~/.archi/<deployment-name>/data/prompts/
```

Default prompt templates are provided in the repository at `examples/defaults/prompts/` for reference.

### Prompt Types

- **condense**: Prompts for condensing conversation history
- **chat**: Main response generation prompts  
- **system**: System-level instructions

### Prompt Directory Structure

```
data/prompts/
├── condense/
│   ├── default.prompt
│   └── concise.prompt
├── chat/
│   ├── default.prompt
│   ├── formal.prompt
│   └── technical.prompt
└── system/
    ├── default.prompt
    └── helpful.prompt
```

### Creating Custom Prompts

1. Navigate to your deployment's prompt directory: `~/.archi/<deployment-name>/data/prompts/`
2. Create or edit a `.prompt` file in the appropriate subdirectory
3. Use standard prompt template syntax with placeholders:
   - `{retriever_output}` - Retrieved documents
   - `{question}` - User question
   - `{history}` - Conversation history

Example `data/prompts/chat/technical.prompt`:
```
You are a technical assistant specializing in software development.
Use precise technical terminology and provide code examples when appropriate.

Context: {context}

Question: {question}
```

### Activating Prompts

**Admin (deployment-wide default):**
```
PATCH /api/config/dynamic
{"active_chat_prompt": "technical"}
```

**User (personal preference):**
```
PATCH /api/users/me/preferences
{"preferred_chat_prompt": "formal"}
```

### Reloading Prompts

After adding or modifying prompt files, reload the cache:
```
POST /api/prompts/reload
```

This is admin-only and updates the in-memory prompt cache without restarting services.

---

## Admin Guide

### Becoming an Admin

Admin status is set in the PostgreSQL `users` table:

```sql
UPDATE users SET is_admin = true WHERE email = 'admin@example.com';
```

### Admin Responsibilities

1. **Configuration Management**
   - Set deployment-wide defaults via `/api/config/dynamic`
   - Monitor configuration changes via `/api/config/audit`

2. **Prompt Management**
   - Add/edit prompt files in `prompts/` directory
   - Reload prompts via API after changes

3. **User Management**
   - Grant admin privileges as needed
   - Review user activity and preferences

### Audit Logging

All admin configuration changes are logged:

```
GET /api/config/audit?limit=50
```

Returns:
```json
{
  "entries": [
    {
      "user_id": "admin_user",
      "changed_at": "2025-01-20T15:30:00Z",
      "config_type": "dynamic",
      "field_name": "temperature",
      "old_value": "0.7",
      "new_value": "0.8"
    }
  ]
}
```

### Best Practices

1. **Test configuration changes** in a staging environment first
2. **Document changes** - use meaningful commit messages for prompt file updates
3. **Monitor audit log** - review for unexpected changes
4. **Backup prompts** - version control your custom prompts

---
