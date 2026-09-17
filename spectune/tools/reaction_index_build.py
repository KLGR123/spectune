"""Offline builder for prebuilt ``reaction_local_index_search`` parquet indexes.

Produces ``{output_dir}/{source}.parquet`` (one per source: ``uspto``,
``chempile``, ``pistachio``). Each row is already RDKit-canonicalized, so
``reaction_local_index_search`` can load these at runtime via
``RXN_LOCAL_INDEX_PREBUILT_DIR`` without paying any RDKit cost or being capped
by ``max_chempile_records``/``max_pistachio_records``.

This is a one-time, explicit, offline step -- run via
``python -m spectune.tools build-reaction-index`` -- not something any tool
triggers automatically at runtime.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .base import JsonDict
from .reaction_local_index_search import _REACTION_SMILES_RE, _record, _split_mixture

_DEFAULT_CHUNK_SIZE = 2_000
_SOURCES = ("uspto", "chempile", "pistachio")
_PARQUET_COLUMNS = (
    "source",
    "source_id",
    "reactant_components",
    "product",
    "canonical_product",
    "product_formula",
    "patent_id",
    "reaction_class_name",
)

_RawRow = tuple[str, str, str, JsonDict | None]


def build_reaction_index(
    *,
    output_dir: str,
    sources: Sequence[str] | None = None,
    uspto_csv_path: str = "",
    chempile_parquet_path: str = "",
    pistachio_smi_path: str = "",
    max_records: int = 0,
    workers: int = 1,
    show_progress: bool = True,
) -> list[JsonDict]:
    """Build prebuilt parquet indexes for each requested source with a configured raw path.

    ``max_records`` is a safety valve (0 = unlimited, matching the "full corpus
    by default" design); ``workers`` parallelizes RDKit canonicalization via
    ``ProcessPoolExecutor`` (1 = no subprocess, run in-process).
    """
    raw_paths = {
        "uspto": uspto_csv_path,
        "chempile": chempile_parquet_path,
        "pistachio": pistachio_smi_path,
    }
    selected = list(sources) if sources else list(_SOURCES)
    unknown = [s for s in selected if s not in raw_paths]
    if unknown:
        raise ValueError(f"unknown reaction index source(s): {unknown}; expected one of {_SOURCES}")

    output = Path(output_dir)
    summaries: list[JsonDict] = []
    for source in selected:
        raw_path = raw_paths[source]
        if not raw_path:
            summaries.append({"source": source, "status": "skipped", "reason": "no raw path configured"})
            continue
        summaries.append(
            _build_source_index(
                source,
                raw_path,
                output,
                max_records=max_records,
                workers=workers,
                show_progress=show_progress,
            )
        )
    return summaries


def _build_source_index(
    source: str,
    raw_path: str,
    output_dir: Path,
    *,
    max_records: int,
    workers: int,
    show_progress: bool,
) -> JsonDict:
    path = Path(raw_path)
    if not path.exists():
        return {"source": source, "status": "error", "reason": f"raw path does not exist: {raw_path}"}

    raw_rows = _raw_rows(source, path, max_records)
    chunks = ((source, chunk) for chunk in _chunked(raw_rows, _DEFAULT_CHUNK_SIZE))

    records: list[JsonDict] = []
    chunk_count = 0
    if workers <= 1:
        for chunk in chunks:
            records.extend(_canonicalize_chunk(chunk))
            chunk_count += 1
            _report_progress(source, chunk_count, len(records), show_progress)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for chunk_records in executor.map(_canonicalize_chunk, chunks):
                records.extend(chunk_records)
                chunk_count += 1
                _report_progress(source, chunk_count, len(records), show_progress)

    output_path = output_dir / f"{source}.parquet"
    _write_parquet(records, output_path)
    return {
        "source": source,
        "status": "built",
        "raw_path": str(path),
        "output_path": str(output_path),
        "records_written": len(records),
    }


def _report_progress(source: str, chunk_count: int, record_count: int, show_progress: bool) -> None:
    if show_progress and chunk_count % 50 == 0:
        print(f"[build-reaction-index] {source}: {chunk_count} chunks, {record_count} records so far")


def _raw_rows(source: str, path: Path, max_records: int) -> Iterator[_RawRow]:
    if source == "uspto":
        rows: Iterator[_RawRow] = _uspto_raw_rows(path)
        return _limited(rows, max_records)
    if source == "chempile":
        return _chempile_raw_rows(path, max_records)
    if source == "pistachio":
        return _pistachio_raw_rows(path, max_records)
    raise ValueError(f"unknown reaction index source: {source!r}")


def _limited(rows: Iterator[_RawRow], max_records: int) -> Iterator[_RawRow]:
    if max_records <= 0:
        yield from rows
        return
    for count, row in enumerate(rows):
        if count >= max_records:
            break
        yield row


def _uspto_raw_rows(path: Path) -> Iterator[_RawRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_id, row in enumerate(reader):
            yield f"uspto_csv:{row_id}", str(row.get("reactant") or ""), str(row.get("product") or ""), None


def _chempile_raw_rows(path: Path, max_records: int) -> Iterator[_RawRow]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["text"])
    if max_records > 0:
        frame = frame.head(int(max_records))
    for row_id, text in enumerate(frame["text"].fillna("").astype(str)):
        match = _REACTION_SMILES_RE.search(text)
        if not match:
            continue
        reaction_smiles = match.group(1).strip().rstrip(".,;:")
        if ">>" not in reaction_smiles:
            continue
        reactant_mix, product_mix = reaction_smiles.split(">>", 1)
        products = _split_mixture(product_mix)
        if not products:
            continue
        yield f"chempile_lift_uspto:{row_id}", reactant_mix, products[0], None


def _pistachio_raw_rows(path: Path, max_records: int) -> Iterator[_RawRow]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row_id, line in enumerate(handle):
            if max_records and row_id >= int(max_records):
                break
            fields = line.rstrip("\n").split("\t")
            first_field = fields[0].strip() if fields else ""
            reaction_smiles = first_field.split(None, 1)[0] if first_field else ""
            if ">>" not in reaction_smiles:
                continue
            reactant_mix, product_mix = reaction_smiles.split(">>", 1)
            products = _split_mixture(product_mix)
            if not products:
                continue
            metadata = {
                "patent_id": fields[1] if len(fields) > 1 else "",
                "reaction_class_name": fields[4] if len(fields) > 4 else "",
            }
            yield f"pistachio_smi:{row_id}", reactant_mix, products[0], metadata


def _chunked(rows: Iterable[_RawRow], size: int) -> Iterator[list[_RawRow]]:
    chunk: list[_RawRow] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _canonicalize_chunk(chunk: tuple[str, list[_RawRow]]) -> list[JsonDict]:
    """Run in a worker process (or in-process when ``workers<=1``): RDKit-canonicalize one chunk."""
    from rdkit import RDLogger  # type: ignore[import-not-found]

    RDLogger.DisableLog("rdApp.*")
    source, rows = chunk
    records: list[JsonDict] = []
    for source_id, reactant_mix, product_raw, metadata in rows:
        record = _record(
            source={"uspto": "uspto_csv", "chempile": "chempile_lift_uspto", "pistachio": "pistachio_smi"}[source],
            source_id=source_id,
            reactant_mix=reactant_mix,
            product_raw=product_raw,
            metadata=metadata,
        )
        if record is not None:
            records.append(record)
    return records


def _write_parquet(records: list[JsonDict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source": record.get("source", ""),
            "source_id": record.get("source_id", ""),
            "reactant_components": list(record.get("reactant_components") or []),
            "product": record.get("product", ""),
            "canonical_product": record.get("canonical_product", ""),
            "product_formula": record.get("product_formula") or "",
            "patent_id": record.get("patent_id", ""),
            "reaction_class_name": record.get("reaction_class_name", ""),
        }
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=list(_PARQUET_COLUMNS))
    frame.to_parquet(output_path, index=False)


__all__ = ["build_reaction_index"]
