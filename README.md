# spectune

Train your own spectrum-interpretation agent for NMR and EI-MS through a full pipeline: dataset construction and augmentation, rollout, post-training (SFT and RL), and end-to-end evaluation.

## Install

```bash
pip install -e .                          # core package only
pip install -e ".[dev]"                   # + pytest, ruff
pip install -e ".[dev,chem,mcp,reaction]"  # + rdkit, fastmcp (nmr_forward_predict), pandas (chempile parquet)
```

## Credentials

Most tools work with no configuration beyond what's listed here; unset URLs/keys
make the corresponding tool report `status="unavailable"` rather than failing.

```bash
export VOLCENGINE_WEBSEARCH_API_KEY=<key>   # web_search
export SANDBOX_FUSION_URL=<url>             # code_interpreter (sandbox backend)
export NMR_GENERATE_API_URL=<url>           # nmr_generate
export NMR_REPAIR_API_URL=<url>             # nmr_repair
export NMR_RANK_API_URL=<url>               # nmr_rerank
export NMR_PREDICT_MCP_URL=<url>            # nmr_forward_predict (needs `pip install spectune[mcp]`)
export NMREXP_SEARCH_MCP_BASE_URL=<url>     # nmrexp_search

export RXN_LOCAL_INDEX_USPTO_CSV=<path>            # reaction_local_index_search (needs `pip install spectune[chem]`)
export RXN_LOCAL_INDEX_CHEMPILE_PARQUET=<path>     # reaction_local_index_search (needs `pip install spectune[chem,reaction]`)
export RXN_LOCAL_INDEX_PISTACHIO_SMI=<path>        # reaction_local_index_search (needs `pip install spectune[chem]`)
# askcos_reaction_forward_predict calls the public https://askcos.mit.edu API by default; no credentials needed.
# ASKCOS_PUBLIC_BASE_URL=<url>              # optional override, e.g. for a private ASKCOS deployment
export UNIMOL3_REACTION_FORWARD_PREDICT_API_URL=<url> # unimol3_reaction_forward_predict; unset by default
                                                       # (placeholder until a self-hosted Uni-Mol3
                                                       # forward-prediction service is deployed)

# semantic_scholar_search / crossref_search / wikipedia_search need no credentials.
export SEMANTIC_SCHOLAR_API_KEY=<key>       # optional, raises Semantic Scholar rate limits
export CROSSREF_MAILTO=<email>              # optional, joins Crossref's "polite pool"
export WIKIPEDIA_LANGUAGE=<code>            # optional, defaults to "en"
```

See `secrets.env.example`. 

```
source secrets.env
```

## Tests

```bash
cd spectune
pytest -v                                   # all tests; real-API tests skip if creds/config absent
pytest -v tests/test_external_tools.py      # external calls (requires credentials/config above)
SPECTUNE_ENABLE_NETWORK_TESTS=1 pytest -v tests/test_external_tools.py  # + credential-free public APIs
```



## Contributing

Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests must pass the automated quality checks and receive maintainer review.