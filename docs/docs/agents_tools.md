# Agents & Tools

Archi uses an **agent-based architecture** where AI assistants are defined by **agent specs** — Markdown files that specify the agent's name, available tools, and system prompt.

## Agent Specs

Agent specs live in the directory configured at `services.chat_app.agents_dir`. Each `*.md` file defines one agent.

### Format

An agent spec is a Markdown file with YAML frontmatter:

```markdown
---
name: CMS Comp Ops
tools:
  - search_local_files
  - search_metadata_index
  - list_metadata_schema
  - fetch_catalog_document
  - search_vectorstore_hybrid
  - mcp
---

You are the CMS Comp Ops assistant. You help with operational questions,
troubleshooting, and documentation lookups. Use tools when needed, cite
evidence from retrieved sources, and keep responses concise and actionable.
```

**Required fields:**

- **`name`** (string): Display name for the agent in the UI dropdown.
- **`tools`** (list of strings): Tools this agent can use — a subset of tools defined by the agent class.

**Prompt body:** Everything after the frontmatter is the system prompt.

### Practical Agent Spec Examples

#### Minimal default agent

```markdown
---
name: CMS CompOps Default
tools:
  - search_local_files
  - search_metadata_index
  - list_metadata_schema
  - fetch_catalog_document
  - search_vectorstore_hybrid
---

You are a CMS CompOps assistant.

Rules:
- Use tools to find evidence before answering.
- Prefer concise, operationally actionable responses.
- Cite relevant files or tickets when possible.
- If evidence is missing, say so and suggest the next query/tool call.
```

#### MCP-enabled agent

```markdown
---
name: CMS CompOps + MCP
tools:
  - search_vectorstore_hybrid
  - fetch_catalog_document
  - mcp
---

You are a research-focused assistant.

Use vectorstore tools for internal docs first.
Use MCP tools for external/system checks when internal evidence is insufficient.
Always distinguish internal evidence from MCP-derived evidence.
```

#### MONIT-focused agent (if MONIT tools are enabled by the agent class)

```markdown
---
name: CMS CompOps MONIT
tools:
  - search_vectorstore_hybrid
  - fetch_catalog_document
  - monit_opensearch_search
  - monit_opensearch_aggregation
---

You support CompOps incident triage.

When a request is about rates, failures, or timeseries:
1. Query MONIT tools.
2. Correlate with internal docs.
3. Return likely cause, confidence, and next checks.
```

These files should live in `services.chat_app.agents_dir` and be selected by `services.chat_app.agent_class`.

### File Discovery

The service loads all `*.md` files from the agents directory. The first agent in lexicographic order is selected as the default. Users can switch agents via the dropdown in the chat UI header.

---

## Available Tools

Agent classes define a **tool registry** — a mapping of tool names to tool builders. Each agent spec selects a subset of available tools via the `tools` list.

The default agent class (`CMSCompOpsAgent`) provides these tools:

### `search_local_files`

Line-level search inside file contents. Supports regex and configurable context lines.

- **Use for:** Finding specific text, code patterns, or error messages
- **Example query:** `timeout error` with `before=2` and `after=2`

### `search_metadata_index`

Search the document catalog by file name, path, or source metadata. Use free-text for partial matches or `key:value` for exact filters.

- **Use for:** Finding files by name, path, or source
- **Example query:** `mz_dilepton.py` or `relative_path:full/path/to/file.py`

### `list_metadata_schema`

Returns the metadata schema and available filter keys. Helps the agent understand what metadata fields exist.

### `fetch_catalog_document`

Pull the full text of a specific file by its hash. Supports truncation with `max_chars`.

- **Use for:** Reading a specific document after finding it via search

### `search_vectorstore_hybrid`

Semantic and keyword (BM25) hybrid retrieval of relevant passages from the vector store.

- **Use for:** Answering questions when you don't know exact keywords

### `mcp`

Enables Model Context Protocol (MCP) tools from external servers. See [MCP Integration](#mcp-integration) below.

---

## Agent Management in the Chat UI

The chat interface provides a full agent management experience:

- **Agent dropdown**: Switch between available agents using the dropdown in the header
- **Create agents**: Click the "+" button to create a new agent with name, tools, and prompt
- **Edit agents**: Click the pencil icon on any agent to modify its configuration
- **Delete agents**: Remove agents you no longer need (with confirmation)

Changes are persisted and take effect on the next request.

---

## Pipelines

Archi supports several pipeline classes. The active pipeline is configured per service via the `agent_class` key:

```yaml
services:
  chat_app:
    agent_class: CMSCompOpsAgent
```

### `CMSCompOpsAgent`

The default pipeline. A ReAct agent with tool-use capabilities — it can search documents, fetch content, and query the vector store. Supports MCP tools and streaming responses.

### `QAPipeline`

A simpler question-answering pipeline for straightforward retrieval-augmented generation without tool use.

### Pipeline Configuration

Pipeline selection is configured per service:

```yaml
services:
  chat_app:
    agent_class: CMSCompOpsAgent
    agents_dir: examples/agents
```

The class (`CMSCompOpsAgent`, `QAPipeline`, etc.) defines available tools and runtime behavior.
The selected agent spec file defines the active prompt and tool subset.

---

## MCP Integration

Archi supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for connecting agents to external tool servers. This allows agents to use tools defined by third-party MCP servers — such as web search, code execution, or custom APIs — alongside Archi's built-in tools.

### How It Works

1. MCP servers are defined in your deployment runtime configuration
2. An agent spec includes `mcp` in its `tools` list to opt in
3. At agent initialization, Archi connects to the MCP servers and discovers available tools
4. The MCP tools are added to the agent's toolset alongside built-in tools

### Configuration

Define MCP servers in your deployment configuration:

```yaml
mcp_servers:
  my_server:
    transport: "stdio"
    command: "uvx"
    args:
      - "mcp-server-example"
  web_search:
    transport: "sse"
    url: "http://localhost:8080/sse"
```

Each server entry follows the format expected by the `langchain-mcp-adapters` library:

- **`transport`**: Communication method — `"stdio"` (subprocess) or `"sse"` (HTTP Server-Sent Events)
- **`command`** / **`args`**: For `stdio` transport, the command to launch the server
- **`url`**: For `sse` transport, the server endpoint

### Agent Spec Example

Include `mcp` in the tools list to enable MCP tools for an agent:

```markdown
---
name: Research Assistant
tools:
  - search_vectorstore_hybrid
  - fetch_catalog_document
  - mcp
---

You are a research assistant with access to external tools via MCP.
Use vectorstore search for internal documents and MCP tools for
external information retrieval.
```

### Runtime Behavior

- MCP sessions are maintained in a background event loop for the lifetime of the service
- Each MCP tool is wrapped for synchronous execution so it integrates seamlessly with the ReAct agent loop
- Tool names from MCP servers are namespaced to avoid conflicts with built-in tools

---

## Vector Store & Retrieval

The vector store powers document retrieval in Archi. It uses PostgreSQL with pgvector for production-grade vector similarity search.

### Core Settings

```yaml
data_manager:
  collection_name: default_collection
  embedding_name: OpenAIEmbeddings
  chunk_size: 1000
  chunk_overlap: 0
  reset_collection: true
  distance_metric: cosine
  retrievers:
    hybrid_retriever:
      num_documents_to_retrieve: 5
      bm25_weight: 0.6
      semantic_weight: 0.4
```

| Setting | Description | Default |
|---------|-------------|---------|
| `collection_name` | Name of the vector store collection | `default_collection` |
| `chunk_size` | Maximum characters per text chunk | `1000` |
| `chunk_overlap` | Overlapping characters between chunks | `0` |
| `reset_collection` | Wipe and recreate collection on startup | `true` |
| `retrievers.hybrid_retriever.num_documents_to_retrieve` | Top-k documents per query | `5` |
| `distance_metric` | Similarity metric: `cosine`, `l2`, or `ip` | `cosine` |

### Hybrid Search

Hybrid search (semantic + BM25 keyword retrieval) is enabled by default as a dynamic runtime setting. The weights are configurable in your YAML:

```yaml
data_manager:
  retrievers:
    hybrid_retriever:
      bm25_weight: 0.6
      semantic_weight: 0.4
```

### Stemming

Enable stemming to reduce words to root forms for improved matching:

```yaml
data_manager:
  stemming:
    enabled: true
```

### Supported Document Formats

`.txt`, `.md`, `.py`, `.c`, `.C`, `.h`, `.sh`, `.html`, `.htm`, `.pdf`, `.json`, `.yaml`, `.yml`, `.csv`, `.tsv`, `.log`, `.rst`, `.php`

### Document Synchronization

Archi automatically synchronizes documents with the vector store:

1. **New files** are chunked, embedded, and indexed
2. **Deleted files** are removed from the collection
3. **All artifacts** are tracked in the PostgreSQL `resources` catalog
