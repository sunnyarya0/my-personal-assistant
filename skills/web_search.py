"""
skills/web_search.py
====================
Web search, scraping, and summarisation skills for the Personal AI Agent Assistant.
"""

from __future__ import annotations

import urllib.parse
import httpx
from core.config import settings
from core.llm_client import get_llm_client
from core.registry import register_tool

# Standard headers to resemble a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _is_domain_allowed(url: str) -> bool:
    """
    Check if the domain of the given URL is allowed according to settings.
    If settings.allowed_domains is empty, all domains are allowed.
    """
    if not settings.allowed_domains:
        return True
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if not netloc and parsed.path:
            # Handle cases where url might be 'example.com' without scheme
            parsed = urllib.parse.urlparse("http://" + url)
            netloc = parsed.netloc.lower()
            
        if not netloc:
            return False
            
        host = netloc.split(":")[0]
        for allowed in settings.allowed_domains:
            if host == allowed or host.endswith("." + allowed):
                return True
    except Exception:
        pass
    return False


@register_tool(description="Search the web for information matching a query.")
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search DuckDuckGo Lite for information matching a query.

    Args:
        query: Natural language search query.
        num_results: Max search results to return (default 5).

    Returns:
        Formatted text containing search results (titles, snippets, URLs) or an error.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        return "Error: beautifulsoup4 is not installed. Run: pip install beautifulsoup4"

    url = "https://html.duckduckgo.com/html/"
    try:
        with httpx.Client(headers=HEADERS, timeout=10.0) as client:
            response = client.post(url, data={"q": query})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # Find result containers
        for body in soup.find_all("div", class_="result__body"):
            title_a = body.find("a", class_="result__title")
            snippet_a = body.find("a", class_="result__snippet")
            
            if not title_a:
                continue

            title = title_a.get_text(strip=True)
            link = title_a.get("href", "")
            snippet = snippet_a.get_text(strip=True) if snippet_a else ""

            # Unpack URL if wrapped by DuckDuckGo redirect
            if "uddg=" in link:
                parsed = urllib.parse.urlparse(link)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    link = qs["uddg"][0]

            if not _is_domain_allowed(link):
                continue

            results.append({
                "title": title,
                "url": link,
                "snippet": snippet
            })

            if len(results) >= num_results:
                break

        if not results:
            return "No web results found."

        lines = [f"Web search results for: '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            lines.append(f"   {r['snippet']}\n")

        return "\n".join(lines)
    except Exception as exc:
        return f"Error performing web search: {exc}"


@register_tool(description="Scrape and extract clean text content from a webpage URL.")
def fetch_webpage(url: str) -> str:
    """
    Fetch a webpage and return clean, readable text content.

    Args:
        url: The absolute HTTP/HTTPS URL of the page.

    Returns:
        Cleaned text content or an error message.
    """
    if not _is_domain_allowed(url):
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or url
        return f"Error: Access to domain '{domain}' is blocked by security policies."

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        return "Error: beautifulsoup4 is not installed. Run: pip install beautifulsoup4"

    try:
        with httpx.Client(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "form", "aside", "iframe"]):
            element.decompose()

        # Extract text
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        if not clean_text:
            return "Webpage fetched successfully, but no text content was found."

        return clean_text
    except Exception as exc:
        return f"Error fetching webpage: {exc}"


@register_tool(description="Fetch a webpage URL and generate a concise TL;DR summary of its content.")
def summarise_webpage(url: str) -> str:
    """
    Scrape a webpage and generate a summary using the local LLM.

    Args:
        url: The absolute HTTP/HTTPS URL to summarize.

    Returns:
        Concise webpage summary or an error message.
    """
    content = fetch_webpage(url)
    if content.startswith("Error fetching webpage:") or content.startswith("Error:"):
        return content

    # Truncate content to avoid LLM context overflow
    truncated = content[:8000]

    try:
        client = get_llm_client()
        system_prompt = (
            "You are an expert summarization assistant. "
            "Summarize the webpage text into a highly structured TL;DR bullet-point list."
        )
        user_prompt = (
            f"Please summarize the following webpage content focusing on key takeaways:\n\n"
            f"URL: {url}\n\n"
            f"CONTENT:\n{truncated}"
        )
        
        response = client.chat(user_prompt, system_prompt=system_prompt)
        return response.content
    except Exception as exc:
        return f"Error generating webpage summary: {exc}"
