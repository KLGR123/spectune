import asyncio

from mock_servers import run_mock_json_server

from spectune import Unimol3ReactionForwardPredictConfig, Unimol3ReactionForwardPredictTool


def test_reports_unavailable_without_api_url():
    tool = Unimol3ReactionForwardPredictTool(Unimol3ReactionForwardPredictConfig(api_url=""))

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.completion == "failure"
    assert result.status == "unavailable"
    assert "UNIMOL3_REACTION_FORWARD_PREDICT_API_URL" in result.warnings[0]


def test_requires_valid_reactants():
    tool = Unimol3ReactionForwardPredictTool(Unimol3ReactionForwardPredictConfig(api_url="http://localhost:0/predict"))

    result = asyncio.run(tool.execute({"reactants": ["not-a-smiles"]}))

    assert result.completion == "failure"
    assert result.status == "error"


def test_sends_expected_payload_and_parses_candidates():
    captured = {}

    def handle_request(body):
        captured["body"] = body
        return {"candidates": [{"smiles": "CCO", "score": 0.9}, {"smiles": "COC", "score": 0.1}]}

    with run_mock_json_server(handle_request, path="/predict") as api_url:
        tool = Unimol3ReactionForwardPredictTool(Unimol3ReactionForwardPredictConfig(api_url=api_url))
        result = asyncio.run(tool.execute({"reactants": ["CCBr", "[OH-]"], "topk": 2}))

    body = captured["body"]
    assert "." in body["reactants"][0] or len(body["reactants"]) == 2
    assert body["topk"] == 2

    assert result.completion == "success"
    assert result.status == "ok"
    candidates = result.data["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["smiles"] == "CCO"
    assert candidates[0]["rank"] == 1
    assert candidates[0]["source"] == "unimol3"


def test_parses_products_scores_response_shape():
    def handle_request(_body):
        return {"products": ["CCO", "COC"], "scores": [0.7, 0.3]}

    with run_mock_json_server(handle_request, path="/predict") as api_url:
        tool = Unimol3ReactionForwardPredictTool(Unimol3ReactionForwardPredictConfig(api_url=api_url))
        result = asyncio.run(tool.execute({"reactants": ["CCBr", "[OH-]"]}))

    assert result.completion == "success"
    assert result.status == "ok"
    candidates = result.data["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["smiles"] == "CCO"


def test_reports_error_on_remote_failure():
    # Nothing is listening on this port, so post_json raises a connection error,
    # exercising the same exception-to-ToolResult path a real HTTP failure would take.
    tool = Unimol3ReactionForwardPredictTool(
        Unimol3ReactionForwardPredictConfig(api_url="http://127.0.0.1:1/predict", timeout_s=2.0)
    )

    result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.completion == "failure"
    assert result.status == "error"
