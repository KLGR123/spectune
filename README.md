# spectune

Train your own spectrum-interpretation agent for NMR and EI-MS through a full pipeline: dataset construction and augmentation, rollout, post-training (SFT and RL), and end-to-end evaluation.

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
...
```

See `secrets.env.example`. 

```
source secrets.env
```



## Datasets

Override paths via `NMREXP_RAW_DIR` / `SPECTUNE_DATASETS_DIR` (see `secrets.env.example`) or by passing a `NmrExpDataLoaderConfig` explicitly.

```python
from spectune import NmrExpDataLoader

loader = NmrExpDataLoader()  # raw_dir defaults to /fs_mol/liujiarun/data/NMRexp
truth_train = loader.load_truth("train")  # ./datasets/nmrexp_truth_train.jsonl
truth_test = loader.load_truth("test")    # ./datasets/nmrexp_truth_test.jsonl
print(len(truth_train), len(truth_test))
for sample in truth_test.take(3):
    print(sample["gt_smiles"], sample["nmr"]["type"], sample["modality"])
```

Override paths via `SPECXMASTER_RAW_DIR` / `SPECTUNE_DATASETS_DIR` (see `secrets.env.example`) or by passing a `SpecXMasterDataLoaderConfig` explicitly.

```python
from spectune import SpecXMasterDataLoader

loader = SpecXMasterDataLoader()  # raw_dir defaults to /fs_mol/liujiarun/data/SpecXMaster
queries = loader.load_queries()  # unzips + caches JSON-Lines under ./datasets
print(len(queries))
for sample in queries.take(3):
    print(sample["query"], sample["modality"])
```



## Classifier

NMR truth and SpecXMaster queries use independent cluster counts. Classification
visualizes both by default: NMR clusters are annotated with representative
molecules, while query clusters are annotated with representative conversation
text.

```python
from spectune import Classifier, ClassifierConfig

classifier = Classifier(
    ClassifierConfig(
        nmr_n_clusters=256,  # millions of molecular structures
        query_n_clusters=50,  # thousands of conversation-level queries
        nmr_num_workers=16,  # parallel RDKit structure featurization
        visualization_dim=2,  # 2 or 3; affects visualization only
    )
)
truth_train = classifier.classify(
    truth_train,
    visualization_path="./outputs/truth_train_clusters.png",
)
queries = classifier.classify(
    queries,
    visualization_path="./outputs/query_clusters.png",
)

# Equal allocation across clusters; falls back to random sampling before classification.
truth_sample = truth_train.sample(1_000, seed=42)
query_sample = queries.sample(200, seed=42)
```



## Tests

```bash
cd spectune
pytest -v                                   # all tests; real-API tests skip if creds/config absent
pytest -v tests/test_external_tools.py      # external calls (requires credentials/config above)
SPECTUNE_ENABLE_NETWORK_TESTS=1 pytest -v tests/test_external_tools.py  # + credential-free public APIs
pytest -v tests/test_dataloader_base.py tests/test_nmrexp_dataloader.py tests/test_specxmaster_dataloader.py  # dataloader only; NMRexp tests skip if pandas/pyarrow absent
pytest -v tests/test_classifier.py           # classifier; optional-dependency tests skip if unavailable
```



## Contributing

Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests must pass the automated quality checks and receive maintainer review.