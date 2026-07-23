import asyncio

from mock_servers import run_mock_sandbox_server, run_mock_web_search_server

from spectune import (
    CodeInterpreterConfig,
    CodeInterpreterTool,
    ToolManager,
    WebSearchConfig,
    WebSearchTool,
)


def test_manager_exposes_openai_schemas():
    manager = ToolManager([WebSearchTool(WebSearchConfig(api_key="test"))])

    assert manager.names == ("web_search",)
    assert manager.schemas[0]["function"]["name"] == "web_search"
    assert manager.schemas[0]["function"]["parameters"]["required"] == ["query"]


def test_mock_web_search_normalizes_hits():
    def handle_request(body):
        assert body["Query"] == "ethanol NMR"
        return [
            {
                "results": [
                    {
                        "title": "Ethanol",
                        "url": "https://example.test/ethanol",
                        "summary": "Reference spectrum",
                    }
                ]
            }
        ]

    with run_mock_web_search_server(handle_request) as api_url:
        tool = WebSearchTool(WebSearchConfig(api_url=api_url, api_key="test"))
        result = asyncio.run(tool.execute({"query": "ethanol NMR"}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert result.data["parsed_hits"][0]["title"] == "Ethanol"
    assert result.data["parsed_hits"][0]["rank"] == 1


def test_web_search_reports_unavailable_without_api_key():
    tool = WebSearchTool(WebSearchConfig(api_key=""))

    result = asyncio.run(tool.execute({"query": "ethanol NMR"}))

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_local_code_execution_requires_explicit_opt_in():
    tool = CodeInterpreterTool(CodeInterpreterConfig(backend="local"))

    result = asyncio.run(tool.execute({"code": "print(1)"}))

    assert result.completion == "failure"
    assert result.status == "unavailable"


def test_local_code_execution_returns_structured_result():
    tool = CodeInterpreterTool(CodeInterpreterConfig(backend="local", allow_local_execution=True))

    result = asyncio.run(tool.execute({"code": 'print("hello")\nprint(\'{"answer": 42}\')'}))

    assert result.completion == "success"
    assert result.status == "ok"
    assert result.data["stdout"] == 'hello\n{"answer": 42}\n'
    assert result.data["result_json"] == {"answer": 42}


def test_manager_accepts_json_arguments():
    tool = CodeInterpreterTool(CodeInterpreterConfig(backend="local", allow_local_execution=True))
    manager = ToolManager([tool])

    result = asyncio.run(manager.invoke("code_interpreter", '{"code":"print(6 * 7)"}'))

    assert result.data["stdout"] == "42\n"


def test_mock_sandbox_code_execution_sends_http_request():
    def handle_request(body):
        assert body["language"] == "python"
        assert "6 * 7" in body["code"]
        return {
            "status": "Success",
            "run_result": {
                "stdout": "42\n",
                "stderr": "",
                "return_code": 0,
                "execution_time": 0.01,
            },
        }

    with run_mock_sandbox_server(handle_request) as sandbox_url:
        tool = CodeInterpreterTool(CodeInterpreterConfig(backend="sandbox", sandbox_url=sandbox_url))
        result = asyncio.run(tool.execute({"code": "print(6 * 7)"}))

    assert result.completion == "success"
    assert result.data["backend"] == "sandbox"
    assert result.data["stdout"] == "42\n"
    assert result.data["result_json"] is None


def test_mock_sandbox_code_execution_reports_remote_failure():
    def handle_request(_body):
        return {
            "status": "Failed",
            "run_result": {
                "stdout": "",
                "stderr": "boom",
                "return_code": 1,
                "execution_time": 0.01,
            },
        }

    with run_mock_sandbox_server(handle_request) as sandbox_url:
        tool = CodeInterpreterTool(CodeInterpreterConfig(backend="sandbox", sandbox_url=sandbox_url))
        result = asyncio.run(tool.execute({"code": "raise RuntimeError('boom')"}))

    assert result.completion == "failure"
    assert result.status == "error"
    assert result.data["stderr"] == "boom"
