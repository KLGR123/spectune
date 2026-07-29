"""Loader for the NMRexp experimental-NMR structure-elucidation dataset.

NMRexp pairs experimental NMR spectrum text (extracted from paper PDFs) with a
candidate molecular structure. It ships as two schema-compatible shapes:

- a single large **raw** export (currently ``NMRexp_10to24_1_1004.parquet``)
  with one ``SMILES`` column, produced by automatic page-proximity matching
  between a spectrum paragraph and a structure diagram, and never manually
  verified;
- several small **checked** CSV exports (``test_300_checked.csv`` plus a few
  heteronuclide-focused subsets) where a human reviewer additionally recorded
  ``smiles_actual`` (the corrected structure) and a handful of QA flags.

Both shapes share the same 15 base columns (enforced by :func:`_validate_schema`
below), so :class:`NmrExpDataLoader` normalizes every configured source into
one flat record schema (a resolved ``gt_smiles``, the NMR evidence, and
provenance/QC metadata). The raw export becomes ``truth_train`` while the
human-checked exports are merged into ``truth_test``. Downstream, each truth
split will be combined independently with seeds sampled from
:class:`~spectune.dataloader.specxmaster.SpecXMasterDataLoader`'s ``queries``
to produce the final train/test datasets.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spectune.tools.utils import canonical_smiles_and_formula, molecular_formula

from .base import JsonDict, JsonlDataset, has_pandas, progress_iter
from .config import NmrExpDataLoaderConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import IO

    import pandas as pd

# The full raw-file column set, used only to validate that a source file has the
# expected shape (see `_validate_schema`). `Page_in_file_*`/`Location_in_page_*`
# are PDF-extraction bookkeeping from the upstream pipeline, not needed for the
# spectrum-to-structure task, so they are checked for presence here but not
# projected into the normalized output record.
_BASE_COLUMNS = (
    "Filename",
    "SMILES",
    "Page_in_file_mol",
    "Page_in_file_para",
    "Location_in_page_mol",
    "Location_in_page_para",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_shift_text",
    "NMR_note",
    "NMR_processed",
    "Atom_number",
    "Atom_number_diff_env",
    "Atom_number_abstract",
)
# Present only on the human-checked CSV sources, never on the raw export.
_CHECKED_COLUMNS = (
    "smiles_actual",
    "text_in_pdf",
    "nmr_frequency_right",
    "nmr_solvent_right",
    "nmr_processed_right",
    "is_same_molecule",
    "is_same_skeleton",
    "num_chiral_centers",
)


class NmrExpDataLoader:
    """Loads NMRexp exports into separate ``truth_train``/``truth_test`` pools.

    1. :meth:`preprocess` reads the raw parquet/CSV file(s) named in
       ``config.sources``, resolves one ground-truth SMILES per row,
       normalizes NMR/provenance/QC fields into a single flat schema, and writes
       the source groups configured in ``config.truth_splits`` to separate
       cached JSON-Lines files under ``config.processed_dir``.
    2. :meth:`load` (aliased :meth:`load_truth`) returns a
       :class:`~spectune.dataloader.base.JsonlDataset` for one truth split --
       sized, indexable, and repeatably iterable -- building that split on
       first use if its cache is missing.

    Ground-truth SMILES resolution: the raw export only has a single
    ``SMILES`` column (machine-extracted, unverified), so it is used as-is. The
    human-checked CSV sources additionally carry ``smiles_actual`` -- the
    structure a reviewer confirmed against the source paper -- which is
    preferred whenever present, since ``is_same_molecule``/``is_same_skeleton``
    show it frequently differs from the raw ``SMILES`` (a stereochemistry fix,
    or a full re-identification when the extracted structure was wrong). Both
    values are kept in the output record (``gt_smiles`` / ``provenance.smiles_raw``)
    for traceability, and every record's ``provenance.source`` names which
    configured source it came from so raw/checked provenance is never lost by
    the merge.

    NMRexp itself carries no reaction-context fields (it is spectrum-to-structure
    data, not reaction data), so no ``reaction`` key is emitted; a future
    reaction-augmented dataset can introduce whatever reaction fields it needs
    without this loader having pre-guessed (and likely gotten wrong) their shape.
    """

    def __init__(self, config: NmrExpDataLoaderConfig | None = None) -> None:
        self.config = config or NmrExpDataLoaderConfig()

    def preprocess(
        self,
        splits: Sequence[str] | None = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, JsonDict]:
        """Build (or reuse) cached JSON-Lines files for requested truth splits.

        Returns a mapping from split name to its cache/build summary. With the
        default config this writes ``nmrexp_truth_train.jsonl`` from the raw
        parquet and ``nmrexp_truth_test.jsonl`` from all checked CSV sources.
        """
        if not has_pandas():
            raise RuntimeError("pandas is required for NmrExpDataLoader; install spectune[data]")

        config = self.config
        target_splits = list(splits) if splits is not None else list(config.truth_splits)
        unknown_splits = [name for name in target_splits if name not in config.truth_splits]
        if unknown_splits:
            raise KeyError(
                f"unknown NMRexp truth split(s) {unknown_splits}; configured splits: {sorted(config.truth_splits)}"
            )

        summaries: dict[str, JsonDict] = {}
        raw_dir = Path(config.raw_dir)
        for split in target_splits:
            out_path = self._processed_path(split)
            if out_path.exists() and not overwrite:
                summaries[split] = {"status": "cached", "output_path": str(out_path)}
                continue

            source_keys = config.truth_splits[split]
            unknown_sources = [name for name in source_keys if name not in config.sources]
            if unknown_sources:
                raise KeyError(
                    f"NMRexp truth split {split!r} references unknown source(s) {unknown_sources}; "
                    f"configured sources: {sorted(config.sources)}"
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            totals: JsonDict = {
                "source_rows": 0,
                "kept": 0,
                "dropped_empty_gt_smiles": 0,
                "dropped_nmr_type": 0,
                "dropped_quality": 0,
                "dropped_qc_wrong": 0,
            }
            per_source: JsonDict = {}
            with tmp_path.open("w", encoding="utf-8") as handle:
                for source_key in source_keys:
                    if config.max_records and totals["kept"] >= config.max_records:
                        break
                    record_limit = config.max_records - totals["kept"] if config.max_records else 0
                    source_name = config.sources[source_key]
                    source_path = raw_dir / source_name
                    if not source_path.exists():
                        raise FileNotFoundError(
                            f"NMRexp raw source {source_key!r} for truth split {split!r} not found: {source_path} "
                            "(set NMREXP_RAW_DIR or pass a NmrExpDataLoaderConfig(raw_dir=...))"
                        )
                    frame = _read_raw_table(source_path)
                    is_checked = _validate_schema(frame, source_path)
                    stats = _normalize_and_write(
                        frame,
                        handle,
                        truth_split=split,
                        source_key=source_key,
                        source_name=source_name,
                        is_checked=is_checked,
                        config=config,
                        record_limit=record_limit,
                    )
                    per_source[source_key] = {"is_checked": is_checked, **stats}
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + value
            tmp_path.replace(out_path)
            summaries[split] = {
                "status": "built",
                "output_path": str(out_path),
                "sources": per_source,
                **totals,
            }
        return summaries

    def load(self, split: str = "train", *, force_reprocess: bool = False) -> JsonlDataset:
        """Return one processed truth split, building its cache if needed."""
        if split not in self.config.truth_splits:
            raise KeyError(
                f"unknown NMRexp truth split {split!r}; configured splits: {sorted(self.config.truth_splits)}"
            )
        out_path = self._processed_path(split)
        if force_reprocess or not out_path.exists():
            self.preprocess([split], overwrite=force_reprocess)
        return JsonlDataset(out_path)

    def load_truth(self, split: str = "train", **kwargs: Any) -> JsonlDataset:
        """Alias for :meth:`load`, explicitly named for truth datasets."""
        return self.load(split, **kwargs)

    def _processed_path(self, split: str) -> Path:
        return Path(self.config.processed_dir) / f"{self.config.dataset_name.lower()}_truth_{split}.jsonl"


def _read_raw_table(path: Path) -> pd.DataFrame:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # NMRexp's checked CSVs are not consistently UTF-8 (some round-tripped through
    # GBK); the columns this loader actually reads are plain ASCII in every file
    # observed so far, so a permissive fallback chain is safe rather than fatal.
    for encoding in ("utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def _validate_schema(frame: pd.DataFrame, source_path: Path) -> bool:
    """Confirm ``frame`` has the shared NMRexp base schema; return whether it is a checked source.

    Guards the ground-truth resolution logic below: silently guessing which
    schema variant a file is (rather than checking column-by-column) would risk
    picking the wrong SMILES column if an upstream export ever changes shape.
    """
    missing_base = [name for name in _BASE_COLUMNS if name not in frame.columns]
    if missing_base:
        raise ValueError(f"{source_path}: missing expected NMRexp column(s) {missing_base}")
    present_checked = [name for name in _CHECKED_COLUMNS if name in frame.columns]
    if present_checked and len(present_checked) != len(_CHECKED_COLUMNS):
        missing_checked = [name for name in _CHECKED_COLUMNS if name not in frame.columns]
        raise ValueError(
            f"{source_path}: has some but not all checked-source columns (missing {missing_checked}); "
            "expected either the base-only schema or base+checked schema"
        )
    return bool(present_checked)


def _normalize_and_write(
    frame: pd.DataFrame,
    handle: IO[str],
    *,
    truth_split: str,
    source_key: str,
    source_name: str,
    is_checked: bool,
    config: NmrExpDataLoaderConfig,
    record_limit: int = 0,
) -> JsonDict:
    stats: JsonDict = {
        "source_rows": int(len(frame)),
        "kept": 0,
        "dropped_empty_gt_smiles": 0,
        "dropped_nmr_type": 0,
        "dropped_quality": 0,
        "dropped_qc_wrong": 0,
    }
    rows = progress_iter(
        frame.itertuples(index=True),
        total=len(frame),
        label=f"NMRexp/{truth_split}/{source_key}",
        enabled=config.show_progress,
    )
    for row in rows:
        record, drop_reason = _build_record(
            row,
            truth_split=truth_split,
            source_key=source_key,
            source_name=source_name,
            is_checked=is_checked,
            config=config,
        )
        if record is None:
            stats[f"dropped_{drop_reason}"] += 1
            continue
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        stats["kept"] += 1
        if record_limit and stats["kept"] >= record_limit:
            break
    return stats


def _build_record(
    row: Any,
    *,
    truth_split: str,
    source_key: str,
    source_name: str,
    is_checked: bool,
    config: NmrExpDataLoaderConfig,
) -> tuple[JsonDict | None, str | None]:
    raw_smiles = _clean_str(getattr(row, "SMILES", None))
    actual_smiles = _clean_str(getattr(row, "smiles_actual", None)) if is_checked else ""
    gt_source = actual_smiles or raw_smiles
    if not gt_source:
        return None, "empty_gt_smiles"
    if config.canonicalize_smiles:
        # One RDKit parse serves both the canonical SMILES and its formula,
        # instead of parsing the same molecule twice (see canonical_smiles_and_formula).
        gt_smiles, gt_formula = canonical_smiles_and_formula(gt_source)
    else:
        gt_smiles, gt_formula = gt_source, molecular_formula(gt_source)
    if not gt_smiles:
        return None, "empty_gt_smiles"

    nmr_type = _clean_str(getattr(row, "NMR_type", None))
    if config.allowed_nmr_types and nmr_type not in config.allowed_nmr_types:
        return None, "nmr_type"

    is_same_molecule = _clean_bool(getattr(row, "is_same_molecule", None)) if is_checked else None
    is_same_skeleton = _clean_bool(getattr(row, "is_same_skeleton", None)) if is_checked else None
    if is_checked:
        if config.min_quality == "same_skeleton" and is_same_skeleton is False:
            return None, "quality"
        if config.min_quality == "same_molecule" and is_same_molecule is False:
            return None, "quality"

    freq_right = _clean_str(getattr(row, "nmr_frequency_right", None)) if is_checked else ""
    solvent_right = _clean_str(getattr(row, "nmr_solvent_right", None)) if is_checked else ""
    processed_right = _clean_str(getattr(row, "nmr_processed_right", None)) if is_checked else ""
    if is_checked and config.drop_qc_wrong:
        if "wrong" in (freq_right.lower(), solvent_right.lower(), processed_right.lower()):
            return None, "qc_wrong"

    row_index = int(row.Index)
    record: JsonDict = {
        "sample_id": f"{config.dataset_name}:{truth_split}:{source_key}:{row_index}",
        "modality": "nmr",
        # "num_of_queries": 1, # no need for truths
        "gt_smiles": gt_smiles,
        "molecular_formula": gt_formula,
        "nmr": {
            "type": nmr_type or None,
            "frequency": _clean_str(getattr(row, "NMR_frequency", None)) or None,
            "solvent": _clean_str(getattr(row, "NMR_solvent", None)) or None,
            "shift_text": _clean_str(getattr(row, "NMR_shift_text", None)) or None,
            "note": _clean_str(getattr(row, "NMR_note", None)) or None,
            "processed": _parse_processed_peaks(getattr(row, "NMR_processed", None)),
            "atom_number": _clean_int(getattr(row, "Atom_number", None)),
            "atom_number_diff_env": _clean_int(getattr(row, "Atom_number_diff_env", None)),
            "atom_number_abstract": _clean_float(getattr(row, "Atom_number_abstract", None)),
        },
        # Reserved for future modalities (NMRexp itself carries no MS data).
        "ms": None,
        "provenance": {
            "dataset": config.dataset_name,
            "split": truth_split,
            "source": source_key,
            "source_file": source_name,
            "row_index": row_index,
            "filename": _clean_str(getattr(row, "Filename", None)) or None,
            "smiles_raw": raw_smiles or None,
        },
        "quality": None,
    }
    if is_checked:
        record["quality"] = {
            "smiles_actual": actual_smiles or None,
            "is_same_molecule": is_same_molecule,
            "is_same_skeleton": is_same_skeleton,
            "num_chiral_centers": _clean_int(getattr(row, "num_chiral_centers", None)),
            "nmr_frequency_right": freq_right or None,
            "nmr_solvent_right": solvent_right or None,
            "nmr_processed_right": processed_right or None,
            "text_in_pdf": _clean_str(getattr(row, "text_in_pdf", None)) or None,
        }
    return record, None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _clean_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "null") else text


def _clean_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _parse_processed_peaks(value: Any) -> Any:
    """Best-effort ``ast.literal_eval`` of the ``NMR_processed`` peak-list column.

    The column is stored as the ``repr()`` of a Python list of tuples (e.g.
    ``"[('m', [], '1H', 8.52, 8.49), ...]"``), so it round-trips cleanly through
    ``literal_eval``; if it ever doesn't, the raw string is kept rather than
    dropping the record over an unparseable auxiliary field.
    """
    if _is_missing(value):
        return None
    if isinstance(value, list | tuple):
        return list(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


__all__ = ["NmrExpDataLoader"]
