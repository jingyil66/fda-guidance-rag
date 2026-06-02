"""
FDA guidance MCP server (stdio).

Run from repository root:
    python -m backend.mcp.fda_guidance_server
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from mcp.server.fastmcp import FastMCP

from backend.app.services import agent_tools

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

mcp = FastMCP(
    "fda-guidance-rag",
    instructions=(
        "Query FDA medical guidance documents. "
        "Use search_guidance for passage-level facts; "
        "list_guidance to browse or filter the catalog; "
        "get_guidance_detail for one document summary (then search_guidance with pdf_id if needed)."
    ),
)


@mcp.tool()
def search_guidance(query: str, top_k: int = 5, pdf_id: str = "") -> str:
    """Semantic search over ingested FDA guidance chunks (Qdrant + embeddings).

    Args:
        query: Natural-language question or search phrase.
        top_k: Number of passages to return (default 5).
        pdf_id: Optional document id to restrict search to one guidance PDF.
    """
    text, _sources = agent_tools.search_guidance(
        query,
        top_k=top_k,
        pdf_id=pdf_id.strip() or None,
    )
    return text


@mcp.tool()
def list_guidance(
    keyword: str = "",
    center: str = "",
    communication_type: str = "",
    year: int | None = None,
    limit: int = 10,
) -> str:
    """List or filter FDA guidances from metadata (no vector search).

    Args:
        keyword: Match title, summary, or topics.
        center: Filter by FDA center (e.g. CDER, CBER).
        communication_type: e.g. Final or Draft.
        year: Match issue year in metadata when set.
        limit: Max rows to return (1-25).
    """
    return agent_tools.list_guidance(
        keyword=keyword.strip() or None,
        center=center.strip() or None,
        communication_type=communication_type.strip() or None,
        year=year,
        limit=limit,
    )


@mcp.tool()
def get_guidance_detail(pdf_id: str = "", title_keyword: str = "") -> str:
    """Get metadata and summary for one guidance document.

    Args:
        pdf_id: Document id from list_guidance (preferred).
        title_keyword: Partial title match if pdf_id is unknown.
    """
    return agent_tools.get_guidance_detail(
        pdf_id=pdf_id.strip() or None,
        title_keyword=title_keyword.strip() or None,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
