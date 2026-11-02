# Data Sources

> **v2 (legacy).** This page describes the v2 application (the RAG chatbot) on the `main` branch. It is not the 3.0 documentation. v2 keeps running on `main` until the cutover; see [Moving from v2 to 3.0](../migration-from-v2.md).

Archi ingests content from a variety of **data sources** into the PostgreSQL-backed vector store used for document retrieval. Sources are configured in `data_manager.sources` in your YAML configuration.

> **Note:** The `links` source is always enabled by default — you do not need to pass it explicitly.

---

## Web Link Lists

A web link list is a simple text file containing one URL per line. Archi fetches the content from each URL and adds it to the vector store using the `Scraper` class.

### Configuration

Define which link lists to ingest in your configuration file:

```yaml
data_manager:
  sources:
    links:
      input_lists:
        - miscellanea.list
        - additional_urls.list
```

Each `.list` file contains one URL per line:

```
https://example.com/page1
https://example.com/page2
```

### Customizing the Scraper

You can tune HTTP scraping behaviour:

```yaml
data_manager:
  sources:
    links:
      scraper:
        reset_data: true
        verify_urls: false
        enable_warnings: false
```

### SSO-Protected Links

If some links are behind a Single Sign-On (SSO) system, enable the SSO source and configure the Selenium-based collector:

```yaml
data_manager:
  sources:
    sso:
      enabled: true
    links:
      selenium_scraper:
        enabled: true
        selenium_class: CERNSSOScraper
        selenium_class_map:
          CERNSSOScraper:
            kwargs:
              headless: true
              max_depth: 2
```

With `sso.enabled: true`, prefix protected URLs with `sso-`:

```
sso-https://example.com/protected/page
```

**Secrets:**

```bash
SSO_USERNAME=username
SSO_PASSWORD=password
```

### Running

Link scraping is controlled by your config (`data_manager.sources.links.enabled`).

---

## Git Scraping

Ingest content from MkDocs-based git repositories using the `GitScraper` class, which extracts Markdown content directly instead of scraping rendered HTML.

### Configuration

```yaml
data_manager:
  sources:
    git:
      enabled: true
```

In your link lists, prefix repository URLs with `git-`:

```
git-https://github.com/example/mkdocs/documentation.git
```

### Secrets

```bash
GIT_USERNAME=your_username
GIT_TOKEN=your_token
```

Once enabled in config, deploy normally with `archi create --config <config.yaml> --services <...>`.

---

## Indico

Ingest events, contributions (talks) and slide materials from an [Indico](https://getindico.io/) instance. Slides in PDF/PPTX/PPT/ODP format are converted to Markdown via [MarkItDown](https://github.com/microsoft/markitdown).

### Configuration

```yaml
data_manager:
  sources:
    indico:
      enabled: true
      base_url: https://indico.cern.ch
      use_sso: false       # set true for SSO-protected events
      slide_conversion:
        enabled: true
        formats: [pdf, pptx, ppt, odp]
```

Add event URLs to your link lists. URLs with `indico` in the hostname and `/event/` in the path are auto-detected. For Indico instances without `indico` in the hostname, use the explicit `indico-` prefix:

```
https://indico.cern.ch/event/1408515/
https://indico.stfc.ac.uk/event/1825/
indico-https://events.example.org/event/42/
```

For each event the scraper produces Markdown resources for the event metadata, each contribution (talk), and each slide deck. Multi-day events can be restricted with day-filtering options (`max_days`, `only_first_day`, `days`, `date_range`) — see `src/cli/templates/base-config.yaml` for the full set.

For SSO-protected events, set `use_sso: true` and configure the Selenium-based collector as for [SSO-Protected Links](#sso-protected-links). The same `SSO_USERNAME` / `SSO_PASSWORD` secrets are used.

Once enabled in config, deploy normally with `archi create --config <config.yaml> --services <...>`.

### Alternative: the Indico MCP sidecar (on-demand, bearer-authenticated)

The batch scraper above is offline-friendly but heavy: it walks a list of URLs ahead of time and uses Selenium-driven CERN SSO for auth. For most setups the preferred path is the **Indico MCP sidecar** ([`archi-physics/mcp-servers` → `indico/`](https://github.com/archi-physics/mcp-servers/tree/main/indico)), which:

- Authenticates with a bearer token + API key/secret (no Selenium, no browser).
- Lets the agent fetch event metadata, contributions, and attached files on demand at chat time.
- Pairs with the agent-side `ingest_indico_event` tool to chunk + embed + index downloaded attachments via a shared docker volume, putting the same kind of resources into the catalog (`source_type=web`, `metadata.scraper=indico`, `metadata.event_id=<id>`) as the batch scraper would.

The MCP path supplants the `use_sso: true` Selenium flow for new deployments; keep the batch IndicoScraper for offline ingest of large historical archives or when MCP isn't deployed. See [`indico/README.md`](https://github.com/archi-physics/mcp-servers/blob/main/indico/README.md) for configuration (one `shared_volume:` field in the `mcp_servers.indico` entry).

---

## JIRA

Fetch issues and comments from specified JIRA projects using the `JiraClient` class.

### Configuration

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

The optional `cutoff_date` skips tickets created before the specified ISO-8601 date.

### Anonymization

Customize data anonymization to remove personal information:

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

### Secrets

```bash
JIRA_PAT=<your_jira_personal_access_token>
```

Once enabled in config, deploy normally with `archi create --config <config.yaml> --services <...>`.

---

## Redmine

Ingest solved tickets (question/answer pairs) from Redmine into the vector store.

### Configuration

```yaml
data_manager:
  sources:
    redmine:
      url: https://redmine.example.com
      project: my-project
      anonymize_data: true
```

### Secrets

```bash
REDMINE_USER=...
REDMINE_PW=...
```

Once enabled in config, deploy normally with `archi create --config <config.yaml> --services <...>`.

> To automate email replies to resolved tickets, also enable the `redmine-mailer` service. See [Services](services.md).

---

## Adding Documents Manually

### Document Upload (via Chat UI)

The chatbot service includes a built-in document upload interface. When logged in to the chat UI, navigate to `/upload` to upload documents through your browser.

**First-time setup — create an admin account:**

```bash
docker exec -it <CONTAINER-ID> bash
python -u src/bin/service_create_account.py
```

Run the script from the `/root/archi` directory inside the container. After creating an account, visit the chat UI to log in and upload documents.

### Directly Copying Files

Documents used for RAG live in the container at `/root/data/<directory>/`. You can copy files directly:

```bash
docker cp myfile.pdf <container-id>:/root/data/my_docs/
```

To create a new directory inside the container:

```bash
docker exec -it <container-id> mkdir /root/data/my_docs
```

---

## Data Viewer

The chat interface includes a built-in **Data Viewer** for browsing and managing ingested documents. Access it at `/data` on your chat app (e.g., `http://localhost:7861/data`).

### Features

- **Browse documents**: View all ingested documents with metadata (source, file type, chunk count)
- **Search and filter**: Filter documents by name or source type
- **View content**: Click a document to see its full content and individual chunks
- **Enable/disable documents**: Toggle whether specific documents are included in RAG retrieval
- **Bulk operations**: Enable or disable multiple documents at once

### Document States

| State | Description |
|-------|-------------|
| Enabled | Document chunks are included in retrieval (default) |
| Disabled | Document is excluded from retrieval but remains in the database |

Disabling documents is useful for temporarily excluding outdated content, testing retrieval with specific document subsets, or hiding sensitive documents from certain users.

---

## Source Configuration Notes

- Source configuration is persisted to PostgreSQL (`static_config.sources_config`) at deployment time and used at runtime.
- The `visible` flag on a source (`sources.<name>.visible`) controls whether content from that source appears in chat citations and user-facing listings. It defaults to `true`.
- All sources can be listed with `archi list-services`.
