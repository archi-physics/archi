<p align="center">
  <img src="https://raw.githubusercontent.com/archi-physics/archi/main/docs/docs/_static/archi-logo.png" width="200" />
</p>

# _archi_

[![CI](https://github.com/archi-physics/archi/actions/workflows/pr-preview.yml/badge.svg)](https://github.com/archi-physics/archi/actions/workflows/pr-preview.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://archi-physics.github.io/archi/)

_archi_ is a retrieval-augmented generation framework for research and education teams who need a low-barrier to entry, configurable, private, and extensible assistant. The system was first developed at MIT for the SubMIT computing project, and now powers chat, ticketing, and course-support workflows across academia and research organizations.

## Key Capabilities

_archi_ provides:
- Customizable AI pipelines that combine data retrieval and LLMs (and more tools to come!).
- Data ingestion connectors: web links, git repositories, local files, JIRA, and more.
- Interfaces: chat app, ticketing assistant, email bot, and more.
- Support for running or interacting with local and API-based LLMs.
- Modular design that allows custom data sources, LLM backends, and deployment targets.
- Containerized services and CLI utilities for repeatable deployments.

## Documentation

The [docs](https://archi-physics.github.io/archi/) are organized as follows:

- [Install](https://archi-physics.github.io/archi/install/) — system requirements and installation.
- [Quickstart](https://archi-physics.github.io/archi/quickstart/) — deploy your first Archi instance in minutes.
- [User Guide](https://archi-physics.github.io/archi/user_guide/) — overview of all capabilities.
- [Data Sources](https://archi-physics.github.io/archi/data_sources/) — configure web links, git, JIRA, Redmine, and more.
- [Services](https://archi-physics.github.io/archi/services/) — chat, uploader, data manager, Piazza, and other interfaces.
- [Models & Providers](https://archi-physics.github.io/archi/models_providers/) — LLM providers, embeddings, and BYOK.
- [Agents & Tools](https://archi-physics.github.io/archi/agents_tools/) — agent specs, tools, MCP integration.
- [Configuration](https://archi-physics.github.io/archi/configuration/) — full YAML config schema reference.
- [CLI Reference](https://archi-physics.github.io/archi/cli_reference/) — all CLI commands and options.
- [API Reference](https://archi-physics.github.io/archi/api_reference/) — REST API endpoints.
- [Benchmarking](https://archi-physics.github.io/archi/benchmarking/) — evaluate retrieval and response quality.
- [Advanced Setup](https://archi-physics.github.io/archi/advanced_setup_deploy/) — GPU setup and production deployment.
- [Developer Guide](https://archi-physics.github.io/archi/developer_guide/) — architecture, contributing, and extension patterns.
- [Troubleshooting](https://archi-physics.github.io/archi/troubleshooting/) — common issues and fixes.

## Getting Started

Follow the [Install](https://archi-physics.github.io/archi/install/) and [Quickstart](https://archi-physics.github.io/archi/quickstart/) guide to set up prerequisites, configure data sources, and launch an instance.

## Contributing

We welcome fixes and new integrations—see the [Developer Guide](https://archi-physics.github.io/archi/developer_guide/) for coding standards, testing instructions, and contribution tips. Please open issues or pull requests on the [GitHub repository](https://github.com/archi-physics/archi).

## License and Support

_archi_ is released under the [MIT License](LICENSE). For project inquiries, contact paus@mit.edu.
