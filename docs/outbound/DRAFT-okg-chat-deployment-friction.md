# Standing up a chat deployment takes three commands and three undeclared manual steps

## What this asks for

`okg chat-instance up` plus `okg chat sync` describe a chat deployment in
impressive detail — UI features, model names, the preset, its system prompt,
its suggestions, the MCP tool list. But between those two commands sit three
steps that the manifest cannot express and that okg never performs, so an
operator finishes by hand-writing configuration into the instance through the
vendor's admin API. One of those steps is not merely tedious: skipping it is
the default, and the default silently bypasses the graph.

Nothing below is specific to our deployment. Any okg chat deployment meets all
three.

## Behavior before → after

**Before.** The path from nothing to a working chat is six steps, three of
which are okg commands:

1. `okg chat-instance up`
2. create the first account and mint an admin token — *by hand*
3. point the instance at a model provider — *by hand*
4. `okg mcp-serve --transport streamable-http`
5. `okg chat sync`
6. open the site and select the deployment preset — *by hand, and the
   pre-selected default is the wrong one*

**After.** Steps 2, 3 and 6 disappear. `up` → `mcp-serve` → `sync` → the site
opens on the preset and works.

## Findings

### 1. A synced instance opens on a model that has no graph access

The vendor selects the initial model from the first entry of a
comma-separated list: the frontend does
`selectedModelId = $config?.default_models.split(',')[0]`
(`/app/build/_app/immutable/chunks/BLLL3FN7.js`, v0.11.0), fed by
`ui.default_models` (`backend/open_webui/config.py:3047`).

`chat sync` writes exactly one id into that field —
`config["DEFAULT_MODELS"] = default or ""` at `sync.py:4432` — and `default`
is `chat.models.default`. That same value is required to be the preset's
underlying model, since it is passed as `base_model_id` when the preset is
created (`sync.py:4937`, `sync.py:4961`). It therefore has to name a raw
provider model.

The consequence is that the one model the UI pre-selects is the one
configuration guaranteed to have no graph tools and no system prompt. A reader
who opens the site and asks a domain question gets an answer from the model's
general knowledge, with nothing indicating that the deployment's graph was
never consulted. The preset that `sync` just built, with its tools and its
grounding instructions, is one menu click away and nothing points at it.

There is no deployment-side workaround. `chat.models.default` is the only knob
(`chat_block.py:543-547`); `ui:` accepts only `site_name`, `logo_ref`,
`banner` and `features` (`chat_block.py:279`). Reordering `models.builtin`
does not help — that writes `MODEL_ORDER_LIST`, which is ordering, not
selection.

Because the vendor field is a list and okg already knows the preset id it just
created, the smallest fix is to name the preset first rather than collapsing
the field to the base model. A separate manifest field for the UI default
would be cleaner, but is not required to close the hazard.

*Adjacent, same root:* when the preset **is** selected but reached through the
API rather than the UI, the vendor deliberately does not attach its tools —
"API callers don't expect hidden tools; they can explicitly request tools via
`tool_ids`" (`middleware.py:2861`). Here the system prompt *does* apply, so
the model is told it has graph operators it was not given, and describes
searches it never ran. Both paths lead to ungrounded answers; only this one
also produces claims of tool use.

### 2. Nothing can say where the models come from, and sync cannot see that it is missing

`chat:` accepts model *names* — `builtin` and `default` (`chat_block.py:543`)
— but nothing about the provider behind them. `chat-instance up` injects
exactly two variables into the container, the app database URL and the session
key, from a literal dict at `instance.py:1253` with no passthrough. Neither
okg's code nor its docs mention a provider anywhere.

So the operator configures it afterwards by calling the vendor's admin API
directly. That write is invisible to okg: `chat sync` reconciles six surfaces
— `instance_ui`, `models`, `preset`, `prompts`, `deep_links`, `mcp`
(`sync.py:248-253`) — and the provider connection is in none of them. Nothing
checks that a model can answer at all. **`chat sync` therefore reports green on
an instance that cannot produce a single reply.**

The asymmetry is the argument: sync already probes the MCP endpoint and
refuses to proceed when the graph tools are unreachable (`sync.py:4092`). The
tool half of the chat is verified end to end; the model half is not checked at
all, though both are dependencies of the same conversation.

### 3. The first admin account is not bootstrapped

`chat sync` requires `OKG_CHAT_ADMIN_TOKEN` (`sync.py:229`) and there is no
command that produces one. The operator signs up — through the browser, or by
posting to the vendor's signup route, which returns a token sync accepts — and
exports it.

okg is already the party that could do this. It creates the instance, owns its
application database, and mints and manages its session-signing key
(`instance.py:233`). Creating the first account is the same class of act
against the same assets, one step further along.

## How I verified

- Vendor selection logic read out of `ghcr.io/open-webui/open-webui:v0.11.0`
  directly (`config.py`, `main.py`, and the built frontend chunk cited above).
- okg behavior read from `src/okg` at `2d528e824`; every line reference above
  was opened, not recalled.
- Findings 2 and 3 were met in practice during two end-to-end bring-ups: both
  were completed only by calling the vendor admin API by hand, and finding 1
  produced a confidently wrong answer that was mistaken for an okg bug until
  the cause was traced.
