## Tool catalog (framework-agnostic)

```python
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES, schemas_for_names

schemas = schemas_for_names(DEFAULT_RL_TOOL_NAMES)
```

## verl adapter

Spectune tools stay framework-agnostic. For verl tool-agent rollout, use the
optional adapter:

```python
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.verl import write_tools_config

write_tools_config("examples/verl/tools_config.yaml", tool_names=DEFAULT_RL_TOOL_NAMES)
```

Point verl at that YAML via `actor_rollout_ref.rollout.multi_turn.tool_config_path`.
Each entry uses `spectune.tools.verl.SpectuneTool`, which:

- loads **full** OpenAI / JSON Schema trees from `ToolManager` (avoids verl's
  pydantic models that drop `items` / `minimum` / …);
- dispatches to `ToolManager.invoke`;
- returns compact tool JSON (strips echoed `data.request` payloads);
- never raises into the worker — failures become `completion=failure` responses.

See `examples/verl/README.md`.

## Tool Status Checklist

Last run: 2026-07-24 (`SPECTUNE_ENABLE_NETWORK_TESTS=1 pytest -v tests/test_external_tools.py`)

These tests call live services and are marked `external`; default CI excludes them with `pytest -v -m "not external"`.

| Tool | Status | Reason |
|---|---|---|
| `web_search` | PASS | — |
| `code_interpreter` | PASS | — |
| `nmr_generate` | PASS | — |
| `nmr_repair` | PASS | — |
| `nmr_rerank` | PASS | — |
| `nmr_forward_predict` | PASS | — |
| `nmrexp_search` | **FAIL** | Request timed out (backend unresponsive) |
| `reaction_local_index_search` | PASS | — |
| `askcos_reaction_forward_predict` | PASS | — |
| `unimol3_reaction_forward_predict` | SKIP | `UNIMOL3_REACTION_FORWARD_PREDICT_API_URL` not set; service not yet deployed |
| `semantic_scholar_search` | **FAIL** | HTTP 429 Too Many Requests (no API key set; rate-limited) |
| `crossref_search` | PASS | — |
| `wikipedia_search` | **FAIL** | SSL handshake timeout (network blocks outbound HTTPS) |
