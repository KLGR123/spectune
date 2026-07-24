## Tool Status Checklist

Last run: 2026-07-24 (`SPECTUNE_ENABLE_NETWORK_TESTS=1 pytest -v tests/test_external_tools.py`)

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