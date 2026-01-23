# User Guide

## Overview

A2RCHI supports various **data sources** as easy ways to ingest your data into the vector store databased used for document retrieval. These include:

- **Links lists (even behind SSO)**: automatically scrape and ingest documents from a list of URLs
- **Git scraping**: git mkdocs repositories
- **Ticketing systems**: JIRA, Redmine, Piazza
- **Local documents**

Additionally, A2RCHI supports various **interfaces/services**, which are applications that interact with the RAG system. These include:

- **Chat interface**: a web-based chat application
- **Piazza integration**: read posts from Piazza and post draft responses to a Slack channel
- **Cleo/Redmine integration**: read emails and create tickets in Redmine
- **Mattermost integration**: read posts from Mattermost and post draft responses to a Mattermost channel
- **Grafana monitoring dashboard**: monitor system and LLM performance metrics
- **Document uploader**: web interface for uploading and managing documents
- **Grader**: automated grading service for assignments with web interface

Both data sources and interfaces/services are enabled via flags to the `a2rchi create` command,
```bash
a2rchi create [...] --services=chatbot,piazza,... --sources jira,redmine,...
```
The parameters of the services and sources are configured via the configuration file. See below for more details.

We support various **pipelines** which are pre-defined sequences of operations that process user inputs and generate responses.
Each service may support a given pipeline.
See the `Services` and `Pipelines` sections below for more details.
For each pipeline, you can use different models, retrievers, and prompts for different steps of the pipeline.
We support various **models** for both embeddings and LLMs, which can be run locally or accessed via APIs.
See the `Models` section below for more details.
Both pipelines and models are configured via the configuration file.

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

In addition to the required `--name`, `--config/--config-dir`, `--env-file`, and `--services` arguments, the `a2rchi create` command accepts several useful flags:

1. **`--podman`**: Run the deployment with Podman instead of Docker.
2. **`--sources` / `-src`**: Enable additional ingestion sources (`git`, `sso`, `jira`, `redmine`, ...). Provide a comma-separated list.
3. **`--gpu-ids`**: Mount specific GPUs (`--gpu-ids all` or `--gpu-ids 0,1`). The legacy `--gpu` flag still works but maps to `all`.
4. **`--tag`**: Override the local image tag (defaults to `2000`). Handy when building multiple configurations side-by-side.
5. **`--hostmode`**: Use host networking for all services.
6. **`--verbosity` / `-v`**: Control CLI logging level (0 = quiet, 4 = debug).
7. **`--force`** / **`--dry-run`**: Force recreation of an existing deployment and/or show what would happen without actually deploying.

You can inspect the available services and sources, together with descriptions, using `a2rchi list-services`.
The CLI checks that host ports are free before deploying; if a port is already in use, adjust `services.*.external_port` (or `services.*.port` in `--hostmode`) and retry.

> **GPU helpers**
>
> GPU access requires the NVIDIA drivers plus the NVIDIA Container Toolkit. After installing the toolkit, generate CDI entries (for Podman) with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` and confirm with `nvidia-ctk cdi list`. Docker users should run `sudo nvidia-ctk runtime configure --runtime=docker`.

---

## Data Sources

These are the different ways to ingest data into the vector store used for document retrieval.

### Web Link Lists

A web link list is a simple text file containing a list of URLs, one per line.
A2RCHI will fetch the content from each URL and add it to the vector store, using the `Scraper` class.

#### Configuration

You can define which lists of links A2RCHI will ingest in the configuration file as follows:
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
      sso_class: CERNSSOScraper  # or whichever class is appropriate
      sso_class_map:
        CERNSSOScraper:
          kwargs:
            headless: true
            max_depth: 2
```
Then, run `a2rchi create ... --sources sso` to activate the SSO collector.

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

Link scraping is automatically enabled in A2RCHI, you don't need to add any arguments to the `create` command unless the links are sso protected.

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

The JIRA integration allows A2RCHI to fetch issues and comments from specified JIRA projects and add them to the vector store, using the `JiraClient` class.

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
a2rchi create [...] --services=chatbot --sources jira
```

---

### Adding Documents and the Uploader Interface

#### Adding Documents

There are two main ways to add documents to A2RCHI's vector database. They are:

- Manually adding files while the service is running via the uploader GUI
- Directly copying files into the container

These methods are outlined below.

#### Manual Uploader

In order to upload documents while A2RCHI is running via an easily accessible GUI, enable the uploader service when creating the deployment:
```bash
a2rchi create [...] --services=chatbot,uploader
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
from the `/root/A2RCHI` directory inside the container. This script will guide you through creating an account; never reuse sensitive passwords here.

Once you have created an account, visit the outgoing port of the data manager docker service and then log in.
The GUI will then allow you to upload documents while A2RCHI is still running. Note that it may take a few minutes for all the documents to upload.

#### Directly copying files to the container

The documents used for RAG live in the chat container at `/root/data/<directory>/<files>`. Thus, in a pinch, you can `docker/podman cp` a file at this directory level, e.g., `podman/docker cp myfile.pdf <container name or ID>:/root/data/<new_dir>/`. If you need to make a new directory in the container, you can do `podman exec -it <container name or ID> mkdir /root/data/<new_dir>`.

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
a2rchi create [...] --services=chatbot --sources redmine
```

> To automate email replies, also enable the `redmine-mailer` service (see the Services section below).

---

## Interfaces/Services

These are the different apps that A2RCHI supports, which allow you to interact with the AI pipelines.

### Piazza Interface

Set up A2RCHI to read posts from your Piazza forum and post draft responses to a specified Slack channel. To do this, a Piazza login (email and password) is required, plus the network ID of your Piazza channel, and lastly, a Webhook for the slack channel A2RCHI will post to. See below for a step-by-step description of this.

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps) and sign in to workspace where you will eventually want A2RCHI to post to (note doing this in a business workspace like the MIT one will require approval of the app/bot).
2. Click 'Create New App', and then 'From scratch'. Name your app and again select the correct workspace. Then hit 'Create App'
3. Now you have your app, and there are a few things to configure before you can launch A2RCHI:
4. Go to Incoming Webhooks under Features, and toggle it on.
5. Click 'Add New Webhook', and select the channel you want A2RCHI to post to.
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

a2rchi:
  [... a2rchi config ...]

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
a2rchi create [...] --services=chatbot,piazza
```

---

### Redmine/Mailbox Interface

A2RCHI will read all new tickets in a Redmine project, and draft a response as a comment to the ticket.
Once the ticket is updated to the "Resolved" status by an admin, A2RCHI will send the response as an email to the user who opened the ticket.
The admin can modify A2RCHI's response before sending it out.

#### Configuration

```yaml
services:
  redmine_mailbox:
    url: https://redmine.example.com
    project: my-project
    redmine_update_time: 10
    mailbox_update_time: 10
    answer_tag: "-- A2RCHI -- Resolving email was sent"
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
a2rchi create [...] --services=chatbot,redmine-mailer
```

---

### Mattermost Interface

Set up A2RCHI to read posts from your Mattermost forum and post draft responses to a specified Mattermost channel.

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
a2rchi create [...] --services=chatbot,mattermost
```

---

### Grafana Interface

Monitor the performance of your A2RCHI instance with the Grafana interface. This service provides a web-based dashboard to visualize various metrics related to system performance, LLM usage, and more.

> Note, if you are deploying a version of A2RCHI you have already used (i.e., you haven't removed the images/volumes for a given `--name`), the postgres will have already been created without the Grafana user created, and it will not work, so make sure to deploy a fresh instance.

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
a2rchi create [...] --services=chatbot,grafana
```
and you should see something like this
```
CONTAINER ID  IMAGE                                     COMMAND               CREATED        STATUS                  PORTS                             NAMES
d27482864238  localhost/chromadb-gtesting2:2000         uvicorn chromadb....  9 minutes ago  Up 9 minutes (healthy)  0.0.0.0:8000->8000/tcp, 8000/tcp  chromadb-gtesting2
87f1c7289d29  docker.io/library/postgres:16             postgres              9 minutes ago  Up 9 minutes (healthy)  5432/tcp                          postgres-gtesting2
40130e8e23de  docker.io/library/grafana-gtesting2:2000                        9 minutes ago  Up 9 minutes            0.0.0.0:3000->3000/tcp, 3000/tcp  grafana-gtesting2
d6ce8a149439  localhost/chat-gtesting2:2000             python -u a2rchi/...  9 minutes ago  Up 9 minutes            0.0.0.0:7861->7861/tcp            chat-gtesting2
```
where the grafana interface is accessible at `your-hostname:3000`. To change the external port from `3000`, you can do this in the config at `services.grafana.external_port`. The default login and password are both "admin", which you will be prompted to change should you want to after first logging in. Navigate to the A2RCHI dashboard from the home page by going to the menu > Dashboards > A2RCHI > A2RCHI Usage. Note, `your-hostname` here is the just name of the machine. Grafana uses its default configuration which is `localhost` but unlike the chat interface, there are no APIs where we template with a selected hostname, so the container networking handles this nicely.

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

These files should live in a directory which you will pass to the config, and A2RCHI will handle the rest.

- `admin_password.txt`. This file will be passed as a secret and be the admin code to login in to the page where you can reset attempts for students.

#### Secrets

The only grading specific secret is the admin password, which like shown above, should be put in the following file

```bash
ADMIN_PASSWORD=your_password
```

Then it behaves like any other secret.

#### Configuration

The required fields in the configuration file are different from the rest of the A2RCHI services. Below is an example:

```yaml
name: grading_test # REQUIRED

a2rchi:
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
2. `a2rchi.pipelines` -- List of pipelines to use (e.g., `GradingPipeline`, `ImageProcessingPipeline`).
3. `a2rchi.pipeline_map` -- Mapping of pipelines to their required prompts and models.
4. `a2rchi.pipeline_map.GradingPipeline.prompts.required.final_grade_prompt` -- Path to the grading prompt file for evaluating student solutions.
5. `a2rchi.pipeline_map.GradingPipeline.models.required.final_grade_model` -- Model class for grading (e.g., `OllamaInterface`, `HuggingFaceOpenLLM`).
6. `a2rchi.pipeline_map.ImageProcessingPipeline.prompts.required.image_processing_prompt` -- Path to the prompt file for image processing.
7. `a2rchi.pipeline_map.ImageProcessingPipeline.models.required.image_processing_model` -- Model class for image processing (e.g., `OllamaInterface`, `HuggingFaceImageLLM`).
8. `services.chat_app.trained_on` -- A brief description of the data or materials A2RCHI is trained on (required).
9. `services.grader_app.num_problems` -- Number of problems the grading service should expect (must match the number of rubric files).
10. `services.grader_app.local_rubric_dir` -- Directory containing the `solution_with_rubric_*.txt` files.
11. `services.grader_app.local_users_csv_dir` -- Directory containing the `users.csv` file.

Note: `a2rchi` settings live in `src/a2rchi/a2rchi-default-settings.yaml` and are seeded into Postgres at deployment; runtime services read settings from Postgres only. Static config is used for deployment-time defaults and is also stored in Postgres for runtime reads (including `data_manager`). Runtime loaders use `load_runtime_config`/`load_runtime_*` for deployment config and `load_a2rchi_settings` for agent settings to keep the split explicit.

For ReAct-style agents (e.g., `CMSCompOpsAgent`), you may optionally set `a2rchi.pipeline_map.<Agent>.recursion_limit` (default `100`) to control the LangGraph recursion cap; when the limit is hit, the agent returns a final wrap-up response using the collected context.

#### Running

```bash
a2rchi create [...] --services=grader
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
- `AnthropicLLM`

### Ollama

In order to use an Ollama server instance for the chatbot, it is possible to specify `OllamaInterface` for the model name. To then correctly use models on the Ollama server, in the keyword args, specify both the url of the server and the name of a model hosted on the server.

```yaml
a2rchi:
  model_class_map:
    OllamaInterface:
      kwargs:
        base_model: "gemma3" # example
        url: "url-for-server"

```

In this case, the `gemma3` model is hosted on the Ollama server at `url-for-server`. You can check which models are hosted on your server by going to `url-for-server/models`.

---

## Vector Store

The vector store is a database that stores document embeddings, enabling semantic and/or lexical search over your knowledge base. A2RCHI uses ChromaDB as the vector store backend to index and retrieve relevant documents based on similarity to user queries.

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

- **`collection_name`**: Name of the ChromaDB collection. Default: `default_collection`
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

Embeddings convert text into numerical vectors. A2RCHI supports multiple embedding providers:

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

A2RCHI automatically synchronizes your data directory with the vector store:

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
  bm25:
    k1: 0.5
    b: 0.75
```

- **`use_hybrid_search`**: Enable hybrid search combining BM25 and semantic similarity. Default: `false`
- **`bm25_weight`**: Weight for BM25 keyword scores. Default: `0.6`
- **`semantic_weight`**: Weight for semantic similarity scores. Default: `0.4`
- **`bm25.k1`**: BM25 term frequency saturation parameter. Default: `0.5`
- **`bm25.b`**: BM25 document length normalization parameter. Default: `0.75`

### Stemming

By specifying the stemming option within your configuration, stemming functionality for the documents in A2RCHI will be enabled. By doing so, documents inserted into the retrieval pipeline, as well as the query that is matched with them, will be stemmed and simplified for faster and more accurate lookup.

```yaml
data_manager:
  stemming:
    enabled: true
```

When enabled, both documents and queries are processed using the Porter Stemmer algorithm to reduce words to their root forms (e.g., "running" → "run"), improving matching accuracy.

### ChromaDB Backend

A2RCHI supports both local and remote ChromaDB instances:

#### Local (Persistent)

```yaml
services:
  chromadb:
    local_vstore_path: /path/to/vectorstore
```

#### Remote (HTTP Client)

```yaml
services:
  chromadb:
    use_HTTP_chromadb_client: true
    chromadb_host: localhost
    chromadb_port: 8000
```

---

## Benchmarking

A2RCHI has benchmarking functionality provided by the `evaluate` CLI command. We currently support two modes:

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
- `question`: The question to be answered by the A2RCHI instance.
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
a2rchi evaluate -n <name> -e <env_file> -cd <configs_directory> <optionally use  -c <file1>,<file2>, ...> <OPTIONS>
```

Example:
```bash
a2rchi evaluate -n benchmark -c examples/benchmarking/benchmark_configs/example_conf.yaml --gpu-ids all
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

### Add ChromaDB Document Management API Endpoints

##### Debugging ChromaDB endpoints

Debugging REST API endpoints to the A2RCHI chat application for programmatic access to the ChromaDB vector database can be exposed with the following configuration change.
To enable the ChromaDB endpoints, add the following to your config file under `services.chat_app`:

```yaml
services:
  chat_app:
    # ... other config options ...
    enable_debug_chroma_endpoints: true  # Default: false
```

###### ChromaDB  Endpoints Info

####### `/api/list_docs` (GET)

Lists all documents indexed in ChromaDB with pagination support.

**Query Parameters:**
- `page`: Page number (1-based, default: 1)
- `per_page`: Documents per page (default: 50, max: 500)
- `content_length`: Content preview length (default: -1 for full content)

**Response:**
```json
{
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total_documents": 1250,
    "total_pages": 25,
    "has_next": true,
    "has_prev": false
  },
  "documents": [...]
}
```

####### `/api/search_docs` (POST)

Performs semantic search on the document collection using vector similarity.

**Request Body:**
- `query`: Search query string (required)
- `n_results`: Number of results (default: 5, max: 100)
- `content_length`: Max content length (default: -1, max: 5000)
- `include_full_content`: Include complete document content (default: false)

**Response:**
```json
{
  "query": "machine learning",
  "search_params": {...},
  "documents": [
    {
      "content": "Document content...",
      "content_length": 1200,
      "metadata": {...},
      "similarity_score": 0.85
    }
  ]
}
```

---
