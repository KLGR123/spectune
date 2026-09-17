import asyncio

from mock_servers import run_mock_json_server

from spectune import FragmentMatchConfig, FragmentMatchTool, ToolManager


def test_manager_exposes_fragment_match_schema():
    manager = ToolManager([FragmentMatchTool(FragmentMatchConfig(api_url="http://example.test/match"))])

    assert manager.names == ("fragment_match",)
    schema = manager.schemas[0]["function"]
    assert schema["name"] == "fragment_match"
    assert schema["parameters"]["required"] == ["rerank", "fragments"]
    assert schema["parameters"]["additionalProperties"] is False
    rerank_item = schema["parameters"]["properties"]["rerank"]["items"]
    assert rerank_item["required"] == ["rank", "smiles"]
    assert rerank_item["additionalProperties"] is False
    fragment_item = schema["parameters"]["properties"]["fragments"]["items"]
    assert fragment_item["required"] == ["format", "value"]
    assert set(fragment_item["properties"]["format"]["enum"]) == {"text", "smiles", "smarts"}


def test_reports_unavailable_without_api_url():
    tool = FragmentMatchTool(FragmentMatchConfig(api_url=""))

    result = asyncio.run(
        tool.execute({"rerank": [{"rank": 1, "smiles": "CCO"}], "fragments": [{"format": "text", "value": "ethyl"}]})
    )

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_rejects_empty_rerank_without_calling_backend():
    tool = FragmentMatchTool(FragmentMatchConfig(api_url="http://example.test/match"))

    result = asyncio.run(tool.execute({"rerank": [], "fragments": [{"format": "text", "value": "ethyl"}]}))

    assert result.completion == "failure"
    assert result.status == "error"
    assert "non-empty rerank" in result.warnings[0]


def test_rejects_when_no_fragment_resolves():
    tool = FragmentMatchTool(FragmentMatchConfig(api_url="http://example.test/match"))

    result = asyncio.run(
        tool.execute(
            {
                "rerank": [{"rank": 1, "smiles": "CCO"}],
                "fragments": [{"format": "xml", "value": "not-a-real-format"}],
            }
        )
    )

    assert result.completion == "failure"
    assert result.status == "error"
    assert any("unresolved format" in warning for warning in result.warnings)


def test_normalizes_request_and_parses_recommended_candidate():
    captured = {}

    def handle_request(body):
        captured.update(body)
        return {
            "tool": "fragment_match",
            "status": "ok",
            "resolved_fragment_count": 1,
            "matched_candidate_count": 1,
            "recommended": {"rank": 1, "smiles": "c1ccccc1OC"},
            "changed_from_rerank_top1": False,
            "decision_reason": "top1_satisfies_all_fragments",
            "last_tool": "fragment_match",
        }

    with run_mock_json_server(handle_request, path="/match") as api_url:
        tool = FragmentMatchTool(FragmentMatchConfig(api_url=api_url))
        result = asyncio.run(
            tool.execute(
                {
                    # rank deliberately wrong / non-sequential and items carry extra keys the
                    # remote API would reject (additionalProperties: False) -- the tool must
                    # rebuild {rank, smiles} from array position before sending.
                    "rerank": [
                        {"rank": 7, "smiles": "c1ccccc1OC", "score": 0.91, "canonical_smiles": "c1ccccc1OC"},
                        {"rank": 2, "smiles": "c1ccccc1O"},
                    ],
                    "fragments": [{"format": "text", "value": "4-methoxyphenyl", "extra": "ignored"}],
                }
            )
        )

    assert captured == {
        "rerank": [{"rank": 1, "smiles": "COc1ccccc1"}, {"rank": 2, "smiles": "Oc1ccccc1"}],
        "fragments": [{"format": "text", "value": "4-methoxyphenyl"}],
    }
    assert result.completion == "success"
    assert result.status == "ok"
    assert result.data["recommended"] == {"rank": 1, "smiles": "COc1ccccc1", "canonical_smiles": "COc1ccccc1"}
    assert result.data["changed_from_rerank_top1"] is False
    assert result.data["decision_reason"] == "top1_satisfies_all_fragments"


def test_fallback_status_is_reported_as_success_with_fallback_status():
    def handle_request(_body):
        return {
            "tool": "fragment_match",
            "status": "fallback",
            "resolved_fragment_count": 0,
            "matched_candidate_count": 0,
            "recommended": {"rank": 1, "smiles": "CCO"},
            "changed_from_rerank_top1": False,
            "decision_reason": "fragment_unresolved_keep_rerank_top1",
            "last_tool": "fragment_match",
        }

    with run_mock_json_server(handle_request, path="/match") as api_url:
        tool = FragmentMatchTool(FragmentMatchConfig(api_url=api_url))
        result = asyncio.run(
            tool.execute(
                {
                    "rerank": [{"rank": 1, "smiles": "CCO"}],
                    "fragments": [{"format": "text", "value": "zzz_unresolvable_fragment"}],
                }
            )
        )

    assert result.completion == "success"
    assert result.status == "fallback"
    assert result.data["decision_reason"] == "fragment_unresolved_keep_rerank_top1"


def test_reports_error_when_backend_omits_recommended_candidate():
    def handle_request(_body):
        return {"tool": "fragment_match", "status": "ok"}

    with run_mock_json_server(handle_request, path="/match") as api_url:
        tool = FragmentMatchTool(FragmentMatchConfig(api_url=api_url))
        result = asyncio.run(
            tool.execute(
                {
                    "rerank": [{"rank": 1, "smiles": "CCO"}],
                    "fragments": [{"format": "text", "value": "ethyl"}],
                }
            )
        )

    assert result.completion == "failure"
    assert result.status == "error"
    assert "missing a valid recommended candidate" in result.warnings[0]


def test_reports_error_on_backend_http_error():
    def handle_request(_body):
        return {"error": "request must contain exactly rerank and fragments"}

    with run_mock_json_server(handle_request, path="/match") as api_url:
        tool = FragmentMatchTool(FragmentMatchConfig(api_url=api_url))
        result = asyncio.run(
            tool.execute(
                {
                    "rerank": [{"rank": 1, "smiles": "CCO"}],
                    "fragments": [{"format": "text", "value": "ethyl"}],
                }
            )
        )

    # The mock server returns HTTP 200 with an error-shaped body (no "recommended"),
    # which the tool must still treat as a failed call rather than crash.
    assert result.completion == "failure"
    assert result.status == "error"
