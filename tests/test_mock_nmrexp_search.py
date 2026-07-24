import asyncio

from mock_servers import run_mock_json_server

from spectune import NmrExpSearchConfig, NmrExpSearchTool


def test_nmrexp_search_reports_unavailable_without_base_url():
    tool = NmrExpSearchTool(NmrExpSearchConfig(mcp_base_url=""))

    result = asyncio.run(tool.execute({"h_shifts": [1.2, 7.3]}))

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_nmrexp_search_requires_spectral_evidence():
    tool = NmrExpSearchTool(NmrExpSearchConfig(mcp_base_url="http://example.test"))

    result = asyncio.run(tool.execute({}))

    assert result.completion == "failure"
    assert result.status == "error"


def test_nmrexp_search_sends_expected_payload_and_parses_candidates():
    captured = {}

    def handle_request(body):
        captured["body"] = body
        return {
            "candidates": [
                {"smiles": "CCO", "score": 0.91},
                {"smiles": "COC", "score": 0.42},
            ]
        }

    with run_mock_json_server(handle_request, path="/sync_nmr_service_mcp") as api_url:
        base_url = api_url.rsplit("/sync_nmr_service_mcp", 1)[0]
        tool = NmrExpSearchTool(NmrExpSearchConfig(mcp_base_url=base_url))
        result = asyncio.run(
            tool.execute(
                {
                    "h_shifts": [1.2, 3.6],
                    "h_split": ["t", "q"],
                    "c_shifts": [18.0, 58.0],
                    "num_search": 500,
                    "topk": 2,
                    "allowed_elements": ["C", "H", "O"],
                }
            )
        )

    body = captured["body"]
    assert body["input_data"]["search"]["H_shifts"] == [1.2, 3.6]
    assert body["input_data"]["search"]["H_split"] == ["t", "q"]
    assert body["input_data"]["search"]["C_shifts"] == [18.0, 58.0]
    assert body["input_data"]["search"]["num_search"] == 500
    assert body["input_data"]["search"]["topk"] == 2
    assert body["input_data"]["search"]["allowed_elements"] == ["C", "H", "O"]
    assert body["input_data"]["config"]["sigma_h"] == NmrExpSearchConfig().default_sigma_h
    assert body["input_data"]["config"]["num_search"] == 500

    assert result.completion == "success"
    assert result.status == "ok"
    candidates = result.data["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["smiles"] == "CCO"
    assert candidates[0]["rank"] == 1


def test_nmrexp_search_reports_no_candidates_on_empty_response():
    with run_mock_json_server(lambda _body: {"candidates": []}, path="/sync_nmr_service_mcp") as api_url:
        base_url = api_url.rsplit("/sync_nmr_service_mcp", 1)[0]
        tool = NmrExpSearchTool(NmrExpSearchConfig(mcp_base_url=base_url))
        result = asyncio.run(tool.execute({"c_shifts": [58.0]}))

    assert result.completion == "success"
    assert result.status == "no_candidates"
