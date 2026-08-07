# spectune

Train your own spectrum-interpretation agent for NMR and EI-MS through a full pipeline: dataset construction and augmentation, rollout, post-training (SFT and RL), and end-to-end evaluation.

Spectune owns data norms, sample construction, tools, and rewards; external trainers such as [verl](https://github.com/volcengine/verl) are optional adapters (Parquet mapping, rollout, distributed SFT/RL). The core package does not depend on torch, ray, or verl. See [`examples/verl/README.md`](examples/verl/README.md) for agentic tool RL.

## Install

```bash
cd spectune
pip install -e .                              # core package only
pip install -e ".[dev]"                       # + pytest, ruff
pip install -e ".[dev,chem,mcp,reaction]"     # + rdkit, fastmcp (nmr_forward_predict), pandas (chempile parquet)
pip install -e ".[dev,data]"                  # + pandas/pyarrow (dataloader parquet/CSV reads)
pip install -e ".[dev,data,classifier]"       # + sklearn/RDKit/transformers/torch/matplotlib (clustering)
```



## Credentials

Most tools work with no configuration beyond what's listed here; unset URLs/keys make the corresponding tool report `status="unavailable"` rather than failing.

```bash
export VOLCENGINE_WEBSEARCH_API_KEY=<key>   # web_search
export SANDBOX_FUSION_URL=<url>             # code_interpreter (sandbox backend)
export NMR_GENERATE_API_URL=<url>           # nmr_generate
export NMR_REPAIR_API_URL=<url>             # nmr_repair
export NMR_RANK_API_URL=<url>               # nmr_rerank
export NMR_PREDICT_MCP_URL=<url>            # nmr_forward_predict (needs `pip install spectune[mcp]`)
export NMREXP_SEARCH_MCP_BASE_URL=<url>     # nmrexp_search
...
```

See `secrets.env.example`. 

```
source secrets.env
```

## Tests

```bash
cd spectune
source secrets.env
pytest -v                                   # all tests; real-API tests skip if creds/config absent
pytest -v tests/test_augmentor.py           # augmentor; rdkit/LLM-dependent tests skip if unavailable
pytest -v tests/test_external_tools.py      # external calls (requires credentials/config above)
pytest tests/test_external_tools.py -s -m external
SPECTUNE_ENABLE_NETWORK_TESTS=1 pytest -v tests/test_external_tools.py  # + credential-free public APIs
pytest -v tests/test_dataloader_base.py tests/test_nmrexp_dataloader.py tests/test_specxmaster_dataloader.py  # dataloader only; NMRexp tests skip if pandas/pyarrow absent
pytest -v tests/test_classifier.py           # classifier; optional-dependency tests skip if unavailable
```

## Training Related

```bash
tensorboard --logdir /path/to/tb --port 6006 --bind_all
ssh -L 6006:localhost:6006 user@server
tmux new -s spectune
# tmux attach -t spectune
tail -f /tmp/ray/session_latest/logs/worker-*.out | grep "tool-call\|tool-result" # with export SPECTUNE_TOOL_LOG=1
```


## Contributing

Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests must pass the automated quality checks and receive maintainer review.