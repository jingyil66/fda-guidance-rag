from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 200


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_fixed(
    docs: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    if not docs:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def chunk_unstructured_by_section(
    docs: list[Document],
    *,
    max_chars: int = 1200,
    min_chars: int = 200,
) -> list[Document]:
    raise NotImplementedError(
        "unstructured section chunking is not implemented yet; use chunk_fixed instead"
    )


def chunks_to_records(chunks: list[Document], start_id: int = 0) -> list[dict]:
    return [
        {
            "id": start_id + i,
            "text": chunk.page_content,
            "metadata": chunk.metadata or {},
            "score": None,
        }
        for i, chunk in enumerate(chunks)
    ]


def load_pdf_pages(
    pdf_path: str | Path,
    *,
    title: str | None = None,
    pdf_id: str | None = None,
    extra_metadata: dict | None = None,
) -> list[Document]:
    path = Path(pdf_path)
    stem = path.stem
    resolved_title = title or stem
    resolved_pdf_id = pdf_id or stem
    extra_metadata = extra_metadata or {}

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    docs = []

    for index, page in enumerate(pages):
        metadata = {
            "title": resolved_title,
            "pdf_id": resolved_pdf_id,
            "page": index + 1,
            **extra_metadata,
        }
        docs.append(Document(page_content=page.page_content, metadata=metadata))

    return docs
