# User Guide

This guide covers the core concepts and features of Archi. Each topic has its own dedicated page for detailed reference.

## Overview

Archi is a retrieval-based assistant framework with four core parts:

- **Data sources**: Where knowledge comes from (links, git repos, JIRA/Redmine, uploaded files)
- **Vector store + retrievers**: Where ingested content is indexed and searched semantically/lexically
- **Agents + tools**: The reasoning layer that decides what to do and can call tools (search, fetch, MCP, etc.)
- **Services**: The apps users interact with (`chatbot`, `data_manager`, integrations, dashboards)

Why both a vector store and tools?

- The **vector store** is best for relevance-based retrieval across the indexed knowledge base.
- **Tools** let the agent do targeted operations (metadata lookup, full-document fetch, external system calls) that pure embedding search cannot do reliably.

Services are enabled at deployment via flags to `archi create`:

```bash
archi create [...] --services chatbot
```

Pipelines (agent classes) define runtime behavior. Agent specs define prompt + enabled tool subset. Models, embeddings, and retriever settings are configured in YAML.

## Data Sources

Data sources define what gets ingested into Archi's knowledge base for retrieval.
Archi supports several data ingestion methods:

- **Web link lists** (including SSO-protected pages)
- **Git scraping** for MkDocs-based repositories
- **JIRA** and **Redmine** ticketing systems
- **Manual document upload** via the Uploader service or direct file copy
- **Local documents**

Sources are configured under `data_manager.sources` in your config file.

**[Read more →](data_sources.md)**

---

## Services

Archi provides these deployable services:

| Service | Description | Default Port |
|---------|-------------|-------------|
| `chatbot` | Web-based chat interface | 7861 |
| `data_manager` | Data ingestion and vectorstore management | 7871 |
| `piazza` | Piazza forum integration with Slack | — |
| `redmine-mailer` | Redmine ticket responses via email | — |
| `mattermost` | Mattermost channel integration | — |
| `grafana` | Monitoring dashboard | 3000 |
| `grader` | Automated grading service | 7862 |

**[Read more →](services.md)**

---

## Agents & Tools

Agents are defined by **agent specs** — Markdown files with YAML frontmatter specifying name, tools, and system prompt. The agent specs directory is configured via `services.chat_app.agents_dir`.

**[Read more →](agents_tools.md)**

---

## Models & Providers

Archi supports five LLM provider types:

| Provider | Models |
|----------|--------|
| OpenAI | GPT-4o, GPT-4, etc. |
| Anthropic | Claude 4, Claude 3.5 Sonnet, etc. |
| Google Gemini | Gemini 2.0 Flash, Gemini 1.5 Pro, etc. |
| OpenRouter | Access to 100+ models via a unified API |
| Local (Ollama/vLLM) | Any open-source model |

Users can also provide their own API keys at runtime via **Bring Your Own Key (BYOK)**.

**[Read more →](models_providers.md)**

---

## Configuration Management

Archi uses a three-tier configuration system:

1. **Static Configuration** (deploy-time, immutable): deployment name, embedding model, available pipelines
2. **Dynamic Configuration** (admin-controlled, runtime-modifiable): default model, temperature, retrieval parameters
3. **User Preferences** (per-user overrides): preferred model, temperature, prompt selections

Settings are resolved as: User Preference → Dynamic Config → Static Default.

See the [Configuration Reference](configuration.md) for the full YAML schema and the [API Reference](api_reference.md) for the configuration API.

---

<<<<<<< HEAD
## Secrets

Secrets are stored in a `.env` file passed via `--env-file`. Required secrets depend on your deployment:

| Secret | Required For |
|--------|-------------|
| `PG_PASSWORD` | All deployments |
| `OPENAI_API_KEY` | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google Gemini provider |
| `OPENROUTER_API_KEY` | OpenRouter provider |
| `HUGGINGFACEHUB_API_TOKEN` | Private HuggingFace models |
| `GIT_USERNAME` / `GIT_TOKEN` | Git source |
| `JIRA_PAT` | JIRA source |
| `REDMINE_USER` / `REDMINE_PW` | Redmine source |

See [Data Sources](data_sources.md) and [Services](services.md) for service-specific secrets.
=======
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

archi:
  [... archi config ...]

services:
  piazza:
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

archi:
  pipelines:
    - GradingPipeline
  pipeline_map:
    GradingPipeline:
      prompts:
        required:
          final_grade_prompt: final_grade.prompt
      models:
        required:
          final_grade_model: OllamaInterface
    ImageProcessingPipeline:
      prompts:
        required:
          image_processing_prompt: image_processing.prompt
      models:
        required:
          image_processing_model: OllamaInterface

services:
  chat_app:
    trained_on: "rubrics, class info, etc." # REQUIRED
  grader_app:
    num_problems: 1 # REQUIRED
    local_rubric_dir: ~/grading/my_rubrics # REQUIRED
    local_users_csv_dir: ~/grading/logins # REQUIRED

data_manager:
  [...]
```

1. `name` -- The name of your configuration (required).
2. `archi.pipelines` -- List of pipelines to use (e.g., `GradingPipeline`, `ImageProcessingPipeline`).
3. `archi.pipeline_map` -- Mapping of pipelines to their required prompts and models.
4. `archi.pipeline_map.GradingPipeline.prompts.required.final_grade_prompt` -- Path to the grading prompt file for evaluating student solutions.
5. `archi.pipeline_map.GradingPipeline.models.required.final_grade_model` -- Model class for grading (e.g., `OllamaInterface`, `HuggingFaceOpenLLM`).
6. `archi.pipeline_map.ImageProcessingPipeline.prompts.required.image_processing_prompt` -- Path to the prompt file for image processing.
7. `archi.pipeline_map.ImageProcessingPipeline.models.required.image_processing_model` -- Model class for image processing (e.g., `OllamaInterface`, `HuggingFaceImageLLM`).
8. `services.chat_app.trained_on` -- A brief description of the data or materials Archi is trained on (required).
9. `services.grader_app.num_problems` -- Number of problems the grading service should expect (must match the number of rubric files).
10. `services.grader_app.local_rubric_dir` -- Directory containing the `solution_with_rubric_*.txt` files.
11. `services.grader_app.local_users_csv_dir` -- Directory containing the `users.csv` file.

For ReAct-style agents (e.g., `CMSCompOpsAgent`), you may optionally set `archi.pipeline_map.<Agent>.recursion_limit` (default `100`) to control the LangGraph recursion cap; when the limit is hit, the agent returns a final wrap-up response using the collected context.

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

### vLLM

For high-throughput GPU inference with tool-calling support, Archi can deploy a [vLLM](https://docs.vllm.ai/) server as a sidecar container. Reference models with the `vllm/` prefix in your config:

```yaml
archi:
  pipeline_map:
    CMSCompOpsAgent:
      models:
        required:
          agent_model: vllm/Qwen/Qwen3-8B
```

Deploy with `--services chatbot,vllm-server --gpu-ids all`. See the [vLLM Provider](vllm.md) page for full configuration, architecture details, and troubleshooting.

### Models via APIs

We support the following model classes in `models.py` for models accessed via APIs:

- `OpenAILLM`
- `OpenRouterLLM`
- `AnthropicLLM`

#### OpenRouter

OpenRouter uses the OpenAI-compatible API. Configure it by setting `OpenRouterLLM` in your config and providing
`OPENROUTER_API_KEY`. Optional attribution headers can be set via `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME`.

```yaml
archi:
  model_class_map:
    OpenRouterLLM:
      class: OpenRouterLLM
      kwargs:
        model_name: openai/gpt-4o-mini
        temperature: 0.7
```

### Ollama

In order to use an Ollama server instance for the chatbot, it is possible to specify `OllamaInterface` for the model name. To then correctly use models on the Ollama server, in the keyword args, specify both the url of the server and the name of a model hosted on the server.

```yaml
archi:
  model_class_map:
    OllamaInterface:
      kwargs:
        base_model: "gemma3" # example
        url: "url-for-server"

```

In this case, the `gemma3` model is hosted on the Ollama server at `url-for-server`. You can check which models are hosted on your server by going to `url-for-server/models`.

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
>>>>>>> 9b860a3f (docs: add vllm provider documentation)

---

## Benchmarking

Archi has benchmarking functionality via the `archi evaluate` CLI command:

- **SOURCES mode**: Checks if retrieved documents contain the correct sources
- **RAGAS mode**: Uses the Ragas evaluator for answer relevancy, faithfulness, context precision, and context relevancy

**[Read more →](benchmarking.md)**

---

## Admin Guide

### Becoming an Admin

Set admin status in PostgreSQL:

```sql
UPDATE users SET is_admin = true WHERE email = 'admin@example.com';
```

### Admin Capabilities

- Set deployment-wide defaults via the dynamic configuration API
- Manage prompts (add, edit, reload via API)
- View the configuration audit log
- Grant admin privileges to other users

### Audit Logging

All admin configuration changes are logged and queryable:

```
GET /api/config/audit?limit=50
```

See the [API Reference](api_reference.md#configuration) for full endpoint documentation.
