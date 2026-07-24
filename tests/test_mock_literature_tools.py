import asyncio

from mock_servers import run_mock_get_json_server

from spectune import (
    CrossrefSearchConfig,
    CrossrefSearchTool,
    SemanticScholarSearchConfig,
    SemanticScholarSearchTool,
    WikipediaSearchConfig,
    WikipediaSearchTool,
)


def test_semantic_scholar_search_parses_hits():
    def handle_request(path, query):
        assert path == "/graph/v1/paper/search"
        assert query["query"] == ["pyridine NMR"]
        return {
            "data": [
                {
                    "title": "Pyridine NMR study",
                    "year": 2021,
                    "venue": "J. Org. Chem.",
                    "url": "https://example.test/paper",
                    "authors": [{"name": "A. Chemist"}],
                    "abstract": "An NMR study of pyridine.",
                }
            ]
        }

    with run_mock_get_json_server(handle_request) as base_url:
        config = SemanticScholarSearchConfig(api_url=f"{base_url}/graph/v1/paper/search")
        tool = SemanticScholarSearchTool(config)
        result = asyncio.run(tool.execute({"query": "pyridine NMR"}))

    assert result.completion == "success"
    assert result.status == "ok"
    hits = result.data["hits"]
    assert hits[0]["title"] == "Pyridine NMR study"
    assert hits[0]["authors"] == ["A. Chemist"]
    assert hits[0]["rank"] == 1


def test_semantic_scholar_search_requires_query():
    tool = SemanticScholarSearchTool(SemanticScholarSearchConfig())

    result = asyncio.run(tool.execute({"query": ""}))

    assert result.completion == "failure"
    assert result.status == "error"


def test_crossref_search_parses_hits():
    def handle_request(path, query):
        assert path == "/works"
        assert query["query"] == ["Suzuki coupling mechanism"]
        return {
            "message": {
                "items": [
                    {
                        "title": ["Mechanistic study of Suzuki coupling"],
                        "issued": {"date-parts": [[2019]]},
                        "publisher": "ACS",
                        "container-title": ["JACS"],
                        "DOI": "10.1000/example",
                        "URL": "https://doi.org/10.1000/example",
                    }
                ]
            }
        }

    with run_mock_get_json_server(handle_request) as base_url:
        config = CrossrefSearchConfig(api_url=f"{base_url}/works")
        tool = CrossrefSearchTool(config)
        result = asyncio.run(tool.execute({"query": "Suzuki coupling mechanism"}))

    assert result.completion == "success"
    assert result.status == "ok"
    hits = result.data["hits"]
    assert hits[0]["doi"] == "10.1000/example"
    assert hits[0]["year"] == 2019


def test_crossref_search_no_hits():
    with run_mock_get_json_server(lambda _p, _q: {"message": {"items": []}}) as base_url:
        tool = CrossrefSearchTool(CrossrefSearchConfig(api_url=f"{base_url}/works"))
        result = asyncio.run(tool.execute({"query": "nonexistent-topic-xyz"}))

    assert result.completion == "success"
    assert result.status == "no_hits"


def test_wikipedia_search_parses_hits_and_fetches_summary():
    def handle_request(path, query):
        if path == "/w/api.php":
            assert query["srsearch"] == ["benzene"]
            return {
                "query": {
                    "search": [
                        {"title": "Benzene", "pageid": 1, "snippet": "A <span>ring</span> molecule"},
                    ]
                }
            }
        assert path == "/summary/Benzene"
        return {
            "extract": "Benzene is an organic chemical compound.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Benzene"}},
        }

    with run_mock_get_json_server(handle_request) as base_url:
        config = WikipediaSearchConfig(
            search_base_url=f"{base_url}/w/api.php",
            summary_base_url=f"{base_url}/summary",
        )
        tool = WikipediaSearchTool(config)
        result = asyncio.run(tool.execute({"query": "benzene", "fetch_summaries": True}))

    assert result.completion == "success"
    assert result.status == "ok"
    hits = result.data["hits"]
    assert hits[0]["title"] == "Benzene"
    assert hits[0]["snippet"] == "A ring molecule"
    assert hits[0]["summary"] == "Benzene is an organic chemical compound."


def test_wikipedia_search_without_summary_lookup():
    def handle_request(path, _query):
        assert path == "/w/api.php"
        return {"query": {"search": [{"title": "Toluene", "pageid": 2, "snippet": "solvent"}]}}

    with run_mock_get_json_server(handle_request) as base_url:
        config = WikipediaSearchConfig(search_base_url=f"{base_url}/w/api.php")
        tool = WikipediaSearchTool(config)
        result = asyncio.run(tool.execute({"query": "toluene", "fetch_summaries": False}))

    assert result.completion == "success"
    assert result.status == "ok"
    hits = result.data["hits"]
    assert hits[0]["title"] == "Toluene"
    assert "summary" not in hits[0]


def test_wikipedia_search_requires_query():
    tool = WikipediaSearchTool(WikipediaSearchConfig())

    result = asyncio.run(tool.execute({"query": ""}))

    assert result.completion == "failure"
    assert result.status == "error"
