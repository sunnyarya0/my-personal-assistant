"""
tests/skills/test_web_search.py
===============================
Unit tests for the web search and scraping skill in skills/web_search.py.
Mocks bs4 and httpx to provide fully isolated offline tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import LLMResponse
from skills.web_search import fetch_webpage, summarise_webpage, web_search


class MockBeautifulSoup:
    """A highly robust mock for BeautifulSoup to support all operations in web_search.py."""

    def __init__(self, markup: str, parser: str | None = None) -> None:
        self.markup = markup

    def find_all(self, tag: str, class_: str | None = None) -> list[MagicMock]:
        if tag == "div" and class_ == "result__body":
            if "No results!" in self.markup:
                return []

            # Element 1
            e1 = MagicMock()
            title_a1 = MagicMock()
            title_a1.get_text.return_value = "Python Programming Language"
            title_a1.get.return_value = "https://example.com/?uddg=https%3A%2F%2Fpython.org"
            snippet_a1 = MagicMock()
            snippet_a1.get_text.return_value = (
                "The official home of the Python Programming Language."
            )

            def find_mock1(sub_tag: str, class_: str | None = None) -> MagicMock | None:
                if sub_tag == "a" and class_ == "result__title":
                    return title_a1
                if sub_tag == "a" and class_ == "result__snippet":
                    return snippet_a1
                return None

            e1.find.side_effect = find_mock1

            # Element 2
            e2 = MagicMock()
            title_a2 = MagicMock()
            title_a2.get_text.return_value = "Python Tips and Tricks"
            title_a2.get.return_value = "https://example.com/tips"
            snippet_a2 = MagicMock()
            snippet_a2.get_text.return_value = "Great tips for programming in Python."

            def find_mock2(sub_tag: str, class_: str | None = None) -> MagicMock | None:
                if sub_tag == "a" and class_ == "result__title":
                    return title_a2
                if sub_tag == "a" and class_ == "result__snippet":
                    return snippet_a2
                return None

            e2.find.side_effect = find_mock2

            return [e1, e2]
        return []

    def __call__(self, *args: object, **kwargs: object) -> list[MagicMock]:
        # Handle soup(["script", "style", ...]) and decompose
        mock_el = MagicMock()
        return [mock_el]

    def get_text(self, separator: str = "") -> str:
        if "Real Main Content" in self.markup:
            return (
                "Real Main Content\n"
                "This is the first paragraph of clean text.\n"
                "This is the second paragraph."
            )
        return ""


# Create a mock module representing bs4
mock_bs4_module = MagicMock()
mock_bs4_module.BeautifulSoup = MockBeautifulSoup


class TestWebSearchSkills:
    @pytest.fixture(autouse=True)
    def mock_bs4_import(self) -> Iterator[None]:
        """Patch sys.modules to return the mock BeautifulSoup class when importing bs4."""
        with patch.dict("sys.modules", {"bs4": mock_bs4_module}):
            yield

    def test_web_search_success(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "<html>some results</html>"
        mock_response.raise_for_status = MagicMock()

        # Mock httpx client
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            res = web_search("python", num_results=2)

        assert "Web search results for: 'python':" in res
        assert "Python Programming Language" in res
        assert "URL: https://python.org" in res  # un-redirected URL
        assert "The official home of the Python Programming Language." in res
        assert "Python Tips and Tricks" in res
        assert "URL: https://example.com/tips" in res
        mock_client.post.assert_called_once()

    def test_web_search_no_results(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "<html><body>No results!</body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            res = web_search("query-with-no-results")

        assert res == "No web results found."

    def test_web_search_error(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = Exception("Connection refused")

        with patch("httpx.Client", return_value=mock_client):
            res = web_search("python")

        assert "Error performing web search: Connection refused" in res

    def test_fetch_webpage_success(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "<html>Real Main Content</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            res = fetch_webpage("https://example.com/page")

        # Content should remain
        assert "Real Main Content" in res
        assert "This is the first paragraph of clean text." in res
        assert "This is the second paragraph." in res
        mock_client.get.assert_called_once_with("https://example.com/page")

    def test_fetch_webpage_empty_content(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "<html>empty</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            res = fetch_webpage("https://example.com/empty")

        assert "no text content was found" in res

    def test_fetch_webpage_error(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = Exception("HTTP 404 Not Found")

        with patch("httpx.Client", return_value=mock_client):
            res = fetch_webpage("https://example.com/404")

        assert "Error fetching webpage: HTTP 404 Not Found" in res

    def test_summarise_webpage_success(self) -> None:
        mock_llm_client = MagicMock()
        mock_llm_client.chat.return_value = LLMResponse(
            content="- Bullet point summary from LLM",
            model="llama3.2",
            provider="ollama",
        )

        with (
            patch("skills.web_search.fetch_webpage", return_value="Scraped text from page"),
            patch("skills.web_search.get_llm_client", return_value=mock_llm_client),
        ):
            res = summarise_webpage("https://example.com/summary")

        assert res == "- Bullet point summary from LLM"
        mock_llm_client.chat.assert_called_once()
        args, kwargs = mock_llm_client.chat.call_args
        assert "Scraped text from page" in args[0]
        assert "You are an expert summarization assistant" in kwargs["system_prompt"]

    def test_summarise_webpage_scrape_error_returns_directly(self) -> None:
        with patch(
            "skills.web_search.fetch_webpage",
            return_value="Error fetching webpage: connection timed out",
        ):
            res = summarise_webpage("https://example.com/timeout")

        assert res == "Error fetching webpage: connection timed out"

    def test_import_errors_handled_gracefully(self) -> None:
        # Force ImportError on bs4 import
        with patch.dict("sys.modules", {"bs4": None}):
            res1 = web_search("test")
            res2 = fetch_webpage("https://example.com")

            assert "beautifulsoup4 is not installed" in res1
            assert "beautifulsoup4 is not installed" in res2
