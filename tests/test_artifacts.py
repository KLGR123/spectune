import json
from pathlib import Path

from spectune.artifacts import ArtifactCompileConfig, compile_sample, infer_data_type, write_jsonl
from spectune.artifacts.compile import compile_jsonl_file, load_jsonl_files
from spectune.format.v1 import FORMAT_SPEC_VERSION, SYSTEM_PROMPT
from spectune.tools.catalog import DEFAULT_RL_TOOL_NAMES
from spectune.tools.verl import build_tools_config, tools_config_to_yaml, write_tools_config


def _sample() -> dict:
    return {
        "sample_id": "demo:1",
        "source_sample_id": "raw:1",
        "gt_smiles": "CCO",
        "turns": [
            {"role": "user", "content": "解析下列分子结构：请给出可能的结构。"},
            {"role": "user", "content": "分子式可能是 C2H6O。"},
        ],
    }


def test_compile_sample_injects_system_prompt_and_ground_truth():
    row = compile_sample(
        _sample(),
        index=3,
        split="train",
        config=ArtifactCompileConfig(tool_names=("nmr_rerank", "web_search")),
    )

    assert row["agent_name"] == "tool_agent"
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["prompt"][1]["role"] == "user"
    assert row["reward_model"]["ground_truth"] == "CCO"
    assert row["extra_info"]["format_spec"] == FORMAT_SPEC_VERSION
    assert row["extra_info"]["index"] == 3
    assert row["extra_info"]["tool_names"] == ["nmr_rerank", "web_search"]
    assert set(row["extra_info"]["tools_kwargs"]) == {"nmr_rerank", "web_search"}
    # Null create_kwargs stays Parquet-safe (empty {} cannot be written by PyArrow).
    assert row["extra_info"]["tools_kwargs"]["nmr_rerank"] == {"create_kwargs": None}
    # Schemas are resolved at reward/rollout time by default (keeps Parquet lean).
    assert "tool_schemas" not in row["extra_info"]


def test_compile_sample_can_embed_full_schemas():
    row = compile_sample(
        _sample(),
        config=ArtifactCompileConfig(
            tool_names=("nmr_rerank", "web_search"),
            include_tool_schemas=True,
        ),
    )
    schemas = row["extra_info"]["tool_schemas"]
    assert {schema["function"]["name"] for schema in schemas} == {"nmr_rerank", "web_search"}
    rerank = next(schema for schema in schemas if schema["function"]["name"] == "nmr_rerank")
    assert rerank["function"]["parameters"]["properties"]["smiles_list"]["items"] == {"type": "string"}


def test_preserved_tool_schema_keeps_items():
    from spectune.tools.verl import PreservedOpenAIToolSchema, openai_schema_for

    schema = PreservedOpenAIToolSchema(openai_schema_for("nmr_rerank"))
    dumped = schema.model_dump()
    assert dumped["function"]["parameters"]["properties"]["smiles_list"]["items"] == {"type": "string"}
    assert "minimum" in dumped["function"]["parameters"]["properties"]["topk"]


def test_compile_jsonl_roundtrip(tmp_path: Path):
    source = tmp_path / "in.jsonl"
    source.write_text(
        '{"sample_id":"a","gt_smiles":"CCO","turns":[{"role":"user","content":"q"}]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"
    count = compile_jsonl_file(source, output, split="test")
    assert count == 1
    written = output.read_text(encoding="utf-8").strip()
    assert '"agent_name": "tool_agent"' in written or '"agent_name":"tool_agent"' in written


def test_write_jsonl_helper(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    write_jsonl([compile_sample(_sample())], path)
    assert path.exists()
    assert "CCO" in path.read_text(encoding="utf-8")


def _write_lines(path: Path, *sample_ids: str) -> Path:
    path.write_text(
        "\n".join(
            f'{{"sample_id":"{sample_id}","gt_smiles":"CCO","turns":[{{"role":"user","content":"q"}}]}}'
            for sample_id in sample_ids
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_jsonl_files_tags_records_with_per_path_data_source(tmp_path: Path):
    nmrexp = _write_lines(tmp_path / "nmrexp.jsonl", "a", "b")
    others = _write_lines(tmp_path / "others.jsonl", "c")

    records = load_jsonl_files(
        [nmrexp, others],
        shuffle=False,
        data_sources=["spectune/nmrexp", "spectune/others"],
    )

    by_id = {r["sample_id"]: r for r in records}
    assert by_id["a"]["_data_source"] == "spectune/nmrexp"
    assert by_id["b"]["_data_source"] == "spectune/nmrexp"
    assert by_id["c"]["_data_source"] == "spectune/others"


def test_load_jsonl_files_broadcasts_single_data_source(tmp_path: Path):
    nmrexp = _write_lines(tmp_path / "nmrexp.jsonl", "a")
    others = _write_lines(tmp_path / "others.jsonl", "b")

    records = load_jsonl_files([nmrexp, others], shuffle=False, data_sources="spectune/shared")

    assert {r["_data_source"] for r in records} == {"spectune/shared"}


def test_load_jsonl_files_rejects_mismatched_data_source_count(tmp_path: Path):
    nmrexp = _write_lines(tmp_path / "nmrexp.jsonl", "a")
    others = _write_lines(tmp_path / "others.jsonl", "b")

    try:
        load_jsonl_files([nmrexp, others], data_sources=["spectune/nmrexp", "spectune/others", "spectune/extra"])
    except ValueError as exc:
        assert "3 data_sources" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched data_sources count")


def test_compile_sample_prefers_per_record_data_source_over_config():
    sample = {**_sample(), "_data_source": "spectune/others"}
    row = compile_sample(sample, config=ArtifactCompileConfig(data_source="spectune/nmrexp"))
    assert row["data_source"] == "spectune/others"


def test_compile_jsonl_file_tags_each_input_path(tmp_path: Path):
    nmrexp = _write_lines(tmp_path / "nmrexp.jsonl", "nmr:1")
    others = _write_lines(tmp_path / "others.jsonl", "other:1")
    output = tmp_path / "out.jsonl"

    compile_jsonl_file(
        [nmrexp, others],
        output,
        shuffle=False,
        data_sources=["spectune/nmrexp", "spectune/others"],
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    by_id = {r["extra_info"]["sample_id"]: r for r in rows}
    assert by_id["nmr:1"]["data_source"] == "spectune/nmrexp"
    assert by_id["other:1"]["data_source"] == "spectune/others"


def test_tools_config_yaml_lists_default_tools():
    config = build_tools_config(DEFAULT_RL_TOOL_NAMES[:2])
    yaml_text = tools_config_to_yaml(config)
    assert "spectune.tools.verl.SpectuneTool" in yaml_text
    assert "tool_name: nmr_generate" in yaml_text
    assert "tool_name: nmr_repair" in yaml_text
    # Schemas must not be embedded — verl pydantic would strip them.
    assert "tool_schema" not in yaml_text


def test_write_tools_config(tmp_path: Path):
    path = write_tools_config(tmp_path / "tools.yaml", tool_names=("code_interpreter",))
    text = path.read_text(encoding="utf-8")
    assert "tool_name: code_interpreter" in text


def test_compile_sample_never_carries_augmentation_into_extra_info():
    # `augmentation` is only ever a compile-time *input* used to resolve
    # `data_type` (see infer_data_type) -- it has no downstream reader once
    # compiled, so it is never copied into extra_info, whether present,
    # absent, or containing keys beyond `information_type`.
    sample = {**_sample(), "augmentation": {"information_type": "reaction", "solvent": "CDCl3"}}
    row = compile_sample(sample)
    assert "augmentation" not in row["extra_info"]

    row_without = compile_sample(_sample())
    assert "augmentation" not in row_without["extra_info"]


def test_compile_sample_writes_canonical_turns():
    # extra_info.turns must carry every user turn (unlike raw_prompt, which
    # intentionally holds only the first for verl's follow-up interaction),
    # so offline consumers never need to reconstruct it from prompt/raw_prompt.
    row = compile_sample(_sample())
    assert row["extra_info"]["turns"] == _sample()["turns"]
    assert len(row["extra_info"]["turns"]) == 2


def test_compile_sample_infers_data_type_from_augmentation():
    sample = {**_sample(), "augmentation": {"information_type": "reaction"}}
    row = compile_sample(sample)
    assert row["extra_info"]["data_type"] == "reaction"


def test_compile_sample_infers_data_type_from_sample_id_prefix():
    sample = {**_sample(), "sample_id": "fragment:test:46"}
    row = compile_sample(sample)
    assert row["extra_info"]["data_type"] == "fragment"


def test_compile_sample_defaults_data_type_to_none():
    row = compile_sample(_sample())
    assert row["extra_info"]["data_type"] == "none"


def test_infer_data_type_direct():
    assert infer_data_type("NMRexp:bulk:raw:1:aug", {"information_type": "structure"}) == "structure"
    assert infer_data_type("structure:test:606", None) == "structure"
    assert infer_data_type("formula:train:1", None) == "formula"
    assert infer_data_type("unknown:1", None) == "none"


def test_infer_data_type_does_not_guess_unresolved_compound_prefixes():
    # sample_id prefixes must be an exact INFORMATION_TYPES match; a prefix
    # like the old "none_plus_formula" (which mixed two real scenarios and
    # was split/relabeled at the source, see spectune/artifacts/compile.py)
    # falls back to "none" rather than being guessed from substring matches.
    assert infer_data_type("none_plus_formula:train:1", None) == "none"


def test_write_parquet_accepts_null_create_kwargs(tmp_path: Path):
    pytest = __import__("pytest")
    pd = pytest.importorskip("pandas")
    __import__("pyarrow")

    from spectune.artifacts.compile import write_parquet

    row = compile_sample(
        _sample(),
        config=ArtifactCompileConfig(tool_names=("nmr_rerank", "web_search")),
    )
    path = tmp_path / "row.parquet"
    write_parquet([row], path)
    loaded = pd.read_parquet(path)
    tools_kwargs = loaded.iloc[0]["extra_info"]["tools_kwargs"]
    assert set(tools_kwargs) == {"nmr_rerank", "web_search"}
    assert tools_kwargs["nmr_rerank"]["create_kwargs"] is None
