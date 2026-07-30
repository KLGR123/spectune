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

Same for `SpecXMasterDataLoader`.

```python
from spectune import SpecXMasterDataLoader

loader = SpecXMasterDataLoader()  # raw_dir defaults to /fs_mol/liujiarun/data/SpecXMaster
queries = loader.load_queries()  # unzips + caches JSON-Lines under ./datasets
print(len(queries))
for sample in queries.take(3):
    print(sample["query"], sample["modality"])
```