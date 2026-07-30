## Classifier

NMR truth and SpecXMaster queries use independent cluster counts. NMR clusters are annotated with representative molecules.

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