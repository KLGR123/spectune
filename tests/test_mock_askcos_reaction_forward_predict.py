import asyncio

from mock_servers import run_mock_json_server

from spectune import AskcosReactionForwardPredictConfig, AskcosReactionForwardPredictTool


def test_requires_valid_reactants():
    tool = AskcosReactionForwardPredictTool(AskcosReactionForwardPredictConfig())

    result = asyncio.run(tool.execute({"reactants": ["not-a-smiles"]}))

    assert result.completion == "failure"
    assert result.status == "error"


def test_sends_expected_payload_and_parses_candidates():
    captured = {}

    def handle_request(body):
        captured["body"] = body
        return {
            "status_code": 200,
            "result": [
                {"products": ["CCO", "COC"], "scores": [0.9, 0.1]},
            ],
        }

    with run_mock_json_server(handle_request, path="/api/forward/controller/call-sync") as api_url:
        base_url = api_url.rsplit("/api/forward/controller/call-sync", 1)[0]
        tool = AskcosReactionForwardPredictTool(AskcosReactionForwardPredictConfig(base_url=base_url))
        result = asyncio.run(tool.execute({"reactants": ["CCBr", "[OH-]"], "topk": 2}))

    body = captured["body"]
    assert body["backend"] == "wldn5"
    assert body["model_name"] == "pistachio"
    assert "." in body["smiles"][0]

    assert result.completion == "success"
    assert result.status == "ok"
    candidates = result.data["candidates"]
    assert len(candidates) == 2
    assert candidates[0]["smiles"] == "CCO"
    assert candidates[0]["rank"] == 1


def test_reports_askcos_reaction_forward_prediction_ignores_products_field_warning():
    # ``products`` is only derived from an explicit ``reaction_smiles`` shorthand
    # (there is no standalone "products" argument); ASKCOS forward prediction never
    # consumes it, so the tool warns that it was ignored.
    def handle_request(_body):
        return {"status_code": 200, "result": []}

    with run_mock_json_server(handle_request, path="/api/forward/controller/call-sync") as api_url:
        base_url = api_url.rsplit("/api/forward/controller/call-sync", 1)[0]
        tool = AskcosReactionForwardPredictTool(AskcosReactionForwardPredictConfig(base_url=base_url))
        result = asyncio.run(tool.execute({"reaction_smiles": "CCO>>CCOC"}))

    assert "askcos_public_forward_prediction_ignores_products_field" in result.warnings
    assert result.status == "no_candidates"


def test_reports_error_on_remote_failure():
    def handle_request(_body):
        return {"status_code": 500, "message": "boom"}

    with run_mock_json_server(handle_request, path="/api/forward/controller/call-sync") as api_url:
        base_url = api_url.rsplit("/api/forward/controller/call-sync", 1)[0]
        tool = AskcosReactionForwardPredictTool(AskcosReactionForwardPredictConfig(base_url=base_url))
        result = asyncio.run(tool.execute({"reactants": ["CCO"]}))

    assert result.completion == "failure"
    assert result.status == "error"
