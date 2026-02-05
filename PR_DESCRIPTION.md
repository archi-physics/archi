# PR: Data Viewer & Upload Consolidation

## Summary

This PR consolidates the data viewing and upload functionality into the chat_app interface, implementing a proxy pattern where the chat service forwards data management requests to the data-manager service. This enables a unified user experience while maintaining separation of concerns.

Additionally, this PR includes:
- **A2rchi → Archi Rename**: Standardized all references from A2rchi/A2RCHI to Archi/ARCHI throughout the codebase
- **Dynamic Schedule Reload**: Cron schedules can now be updated without service restart

## Key Changes

### 🔄 Naming Standardization (A2rchi → Archi)

- Environment variable: `A2RCHI_CONFIGS_PATH` → `ARCHI_CONFIGS_PATH`
- Container paths: `/root/A2rchi/` → `/root/archi/`
- Database columns documented: `a2rchi_version` → `archi_version`, etc.
- Product name in all docs and configs

### ⏰ Dynamic Schedule Reload

- Cron schedules for auto-sync can now be changed at runtime
- No service restart required when updating schedule in config
- Scheduler checks for config changes every minute

### 🎨 New UI Pages

**Data Viewer (`/data`)** - Browse and manage indexed documents
- Document listing with source type categorization (local files, git, URLs, Jira)
- Search and filter capabilities
- Document preview with content rendering
- Chunk visualization showing how documents are split
- Enable/disable documents from search results
- Stats bar showing document counts by category

**Upload Page (`/upload`)** - Add new data sources
- **Files Tab**: Drag-and-drop file upload with queue management
- **URLs Tab**: Web scraping with crawl depth and SSO options
- **Git Repos Tab**: Clone repositories with MkDocs/code file indexing options
- **Jira Tab**: Sync Jira projects for ticket indexing
- Embedding status bar showing pending vs processed documents
- Auto-sync scheduling for recurring updates

**Database Viewer (`/admin/database`)** - Debug database contents
- Raw table inspection for documents, chunks, and catalog
- Useful for development and troubleshooting

### 🔧 Backend Architecture

**Proxy Pattern Implementation**
- Chat app proxies upload/data management requests to data-manager service
- Endpoints: `/api/upload/*`, `/api/sources/*`, `/api/documents/*`, `/api/catalog/*`
- Maintains authentication at the chat app level
- Data-manager runs on separate port (5001) for service isolation

**API Endpoints Added to Chat App**
```
GET  /api/upload/status          - Embedding status (pending vs embedded counts)
POST /api/upload/file            - Upload file (proxied to data-manager)
POST /api/upload/embed           - Trigger embedding process
GET  /api/sources/git            - List git repositories
POST /api/sources/git/clone      - Clone new repository
GET  /api/sources/urls/queue     - Get URL scrape queue
POST /api/sources/urls/add       - Add URL to scrape
GET  /api/documents              - List all documents
GET  /api/documents/<id>         - Get document details
PUT  /api/documents/<id>/toggle  - Enable/disable document
GET  /api/catalog                - Get catalog entries
```

### 📁 New Files

**Frontend Modules**
- `static/modules/data-viewer.js` - Data viewer page logic
- `static/modules/upload.js` - Upload page with multi-tab support
- `static/modules/database-viewer.js` - Database inspection tool
- `static/modules/content-renderer.js` - Markdown/code rendering
- `static/modules/file-tree.js` - File tree visualization

**Stylesheets**
- `static/data.css` - Data viewer styles
- `static/upload.css` - Upload page styles
- `static/database.css` - Database viewer styles

**Templates**
- `templates/data.html` - Data viewer page
- `templates/upload.html` - Upload page
- `templates/database.html` - Database viewer page

### 🧪 Testing

**New Playwright Tests** (226 tests total, all passing)
- `tests/ui/data-viewer.spec.ts` - Data viewer page tests (28 tests)
- `tests/ui/upload.spec.ts` - Upload page tests (27 tests)
- `tests/ui/workflows/17-file-upload.spec.ts` - File upload workflows
- `tests/ui/workflows/18-url-scraping.spec.ts` - URL scraping workflows
- `tests/ui/workflows/19-git-repos.spec.ts` - Git repo management
- `tests/ui/workflows/20-data-management.spec.ts` - Document management

**Integration Tests**
- `tests/smoke/test_integration.py` - CatalogService and DataViewerService tests
- `tests/smoke/test_ingest.py` - Document ingestion tests

### 📚 Documentation

- `docs/docs/database_schema.md` - Complete PostgreSQL schema reference
- `docs/docs/local_dev_testing.md` - Local development setup guide
- `scripts/dev/run_chat_local.sh` - Helper script for local testing

### 🐛 Bug Fixes

- Fixed `sources_config` empty dict check in data_manager.py (empty `{}` was treated as falsy)
- Added lazy-loading for redmine answer tag to avoid config loading at import time
- Fixed API mock endpoints in UI tests to match actual backend

## Configuration Changes

Added `data_manager` service config to support local development:
```yaml
services:
  data_manager:
    host: "0.0.0.0"
    port: 5001
    template_folder: "..."
    static_folder: "..."
```

## Testing Instructions

1. Start PostgreSQL:
   ```bash
   docker run -d --name archi-test-postgres -p 5439:5432 \
     -e POSTGRES_USER=archi -e POSTGRES_PASSWORD=testpassword123 \
     -e POSTGRES_DB=archi postgres:15
   ```

2. Start data-manager service:
   ```bash
   ARCHI_CONFIGS_PATH="tests/smoke/local_dev_config/" \
   PG_PASSWORD=testpassword123 \
   python src/bin/service_data_manager.py
   ```

3. Start chat service:
   ```bash
   ARCHI_CONFIGS_PATH="tests/smoke/local_dev_config/" \
   PG_PASSWORD=testpassword123 \
   python src/bin/service_chat.py
   ```

4. Run UI tests:
   ```bash
   npx playwright test tests/ui/
   ```

## Screenshots

The new pages include:
- **Data Viewer**: Category-based document browsing with preview pane
- **Upload Page**: Multi-tab interface for different data sources
- **Database Viewer**: Raw table inspection for debugging

## Breaking Changes

None - all existing functionality preserved.

## Related Issues

- Consolidates upload functionality from separate uploader_app
- Enables unified data management through chat interface
