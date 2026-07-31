from pathlib import Path

from spectune.artifacts import ArtifactCompileConfig, compile_sample, write_jsonl
from spectune.artifacts.compile import compile_jsonl_file
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
