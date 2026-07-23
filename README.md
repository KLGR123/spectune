# spectune

Train your own spectrum-interpretation agent for NMR and EI-MS through a full pipeline: dataset construction and augmentation, rollout, post-training (SFT and RL), and end-to-end evaluation.

## Install

```bash
pip install -e .                  # core package only
pip install -e ".[dev]"           # + pytest, ruff
pip install -e ".[dev,chem,mcp]"  # + rdkit (formula tools) and fastmcp (nmr_forward_predict)
```

## Credentials

```bash
export VOLCENGINE_WEBSEARCH_API_KEY=<key>   # web_search
export SANDBOX_FUSION_URL=<url>             # code_interpreter (sandbox backend)
export NMR_GENERATE_API_URL=<url>           # nmr_generate
export NMR_REPAIR_API_URL=<url>             # nmr_repair
export NMR_RANK_API_URL=<url>               # nmr_rerank
export NMR_PREDICT_MCP_URL=<url>            # nmr_forward_predict (needs `pip install spectune[mcp]`)
```

## Tests

```bash
pytest -v                                   # all tests; real-API tests skip if creds absent
pytest -v tests/test_external_tools.py      # external calls (requires credentials above)
```

## Contributing

Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests must pass the automated quality checks and receive maintainer review.