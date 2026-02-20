# Default Prompt Templates

This directory contains default prompt templates for archi deployments.

## Directory Structure

```
prompts/
├── chat/           # Response generation prompts
│   ├── default.prompt
│   ├── formal.prompt
│   └── technical.prompt
├── condense/       # History condensation prompts
│   ├── default.prompt
│   └── concise.prompt
└── system/         # System instruction prompts
    ├── default.prompt
    └── helpful.prompt
```

## Prompt Types

### Chat Prompts (`chat/`)
Used to generate responses with retrieval context. These prompts receive:
- `{question}` - The user's question
- `{retriever_output}` - Retrieved context documents
- `{history}` - Chat history (optional)

### Condense Prompts (`condense/`)
Used to condense chat history into a standalone question. These prompts receive:
- `{question}` - The current question
- `{history}` - Previous chat history

### System Prompts (`system/`)
Used as system instructions for the LLM. These set the assistant's persona and behavior.

## Customizing Prompts

### Option 1: Edit Deployment Prompts (Recommended)
After deployment, prompts are copied to your deployment's `data/prompts/` directory:
```
~/.archi/<deployment-name>/data/prompts/
```

Edit the files there and reload via API:
```bash
curl -X POST http://localhost:7868/api/prompts/reload
```

### Option 2: Specify in Config
You can specify explicit prompt file paths in your config.yaml for classic pipelines:
```yaml
services:
  chat_app:
    agent_class: QAPipeline
    prompts:
      chat_prompt: /path/to/my/custom.prompt
      condense_prompt: /path/to/my/condense.prompt
```

## File Format

Prompt files use the `.prompt` extension. Lines starting with `#` at the beginning of the file are treated as comments:

```
# Description of what this prompt does
# Required variables: {question}, {retriever_output}

You are a helpful assistant...
```

## Creating Custom Prompts

1. Copy an existing prompt as a starting point
2. Modify the content for your use case
3. Ensure all required template variables are present
4. Test with your deployment

Template variables are filled in at runtime using Python's `.format()` method.
