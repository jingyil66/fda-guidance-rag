from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.app.services.passage_format import (
    PASSAGE_FORMAT_LEGACY,
    format_passage_for_context,
    get_passage_body,
)

DEFAULT_LLM_MODEL = "gpt-4o-mini"

DEFAULT_PROMPT_TEMPLATE = """
    ### Role
    You are a precise and comprehensive Medical/Regulatory Affairs Assistant. Your goal is to answer questions based STRICTLY on the provided FDA guidance context.

    ### Context Information
    Below are relevant segments retrieved from the database. Each segment is labeled [1], [2], etc.

    {context}

    ### Instructions
    1. **Analyze all segments**: Some information may be spread across multiple chunks. Synthesize them into a single, cohesive answer.
    2. **Be Comprehensive**: Include all specific details, dates, names, and requirements mentioned in the context that are relevant to the question.
    3. **Accuracy First**: Do not infer or assume information not explicitly stated. If the context is insufficient to provide a full answer, state what is available and note what is missing.
    4. **Tone**: Professional, direct, and factual.
    5. **Citations**: Cite segment numbers inline using [1], [2], etc. when stating requirements or facts drawn from the context.

    ### Response Format
    - If the answer is found: Provide a clear, structured response.
    - If the answer is NOT in the context: Respond exactly with: "The answer to this question is not available in the provided content."

    Question: {query}
    Answer:
"""


def get_prompt(template: str = DEFAULT_PROMPT_TEMPLATE) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(template)


def get_llm(model_name: str = DEFAULT_LLM_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model_name)


def get_parser() -> StrOutputParser:
    return StrOutputParser()


def format_context(
    passages: list[dict],
    *,
    passage_format: str = PASSAGE_FORMAT_LEGACY,
    numbered: bool = False,
) -> str:
    blocks = []
    for index, passage in enumerate(passages, start=1):
        block = format_passage_for_context(passage, passage_format)
        if numbered:
            block = f"[{index}]\n{block}"
        blocks.append(block)
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    context: str,
    prompt: ChatPromptTemplate,
    llm: ChatOpenAI,
    parser: StrOutputParser,
) -> str:
    chain = prompt | llm | parser
    return chain.invoke({"query": query, "context": context})


def build_sources(passages: list[dict], *, snippet_max_chars: int = 800) -> list[dict]:
    sources = []
    for passage in passages:
        metadata = passage.get("metadata") or {}
        snippet = get_passage_body(passage).strip()
        if snippet_max_chars and len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars].rstrip() + "…"
        sources.append(
            {
                "title": metadata.get("title", "Unknown"),
                "page": metadata.get("page", "?"),
                "pdf_id": metadata.get("pdf_id", ""),
                "url": metadata.get("url", ""),
                "field_communication_type": metadata.get("field_communication_type", ""),
                "snippet": snippet,
            }
        )
    return sources


def build_documents(passages: list[dict]) -> list[dict]:
    return [
        {
            "text": passage.get("text", ""),
            "metadata": passage.get("metadata") or {},
        }
        for passage in passages
    ]
