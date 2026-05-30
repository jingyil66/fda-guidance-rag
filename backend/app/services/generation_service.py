from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

DEFAULT_LLM_MODEL = "gpt-4o-mini"

DEFAULT_PROMPT_TEMPLATE = """
    ### Role
    You are a precise and comprehensive Medical/Regulatory Affairs Assistant. Your goal is to answer questions based STRICTLY on the provided FDA guidance context.

    ### Context Information
    Below are relevant segments retrieved from the database. Each segment is formatted as [Index] (Title | Page): Content.

    {context}

    ### Instructions
    1. **Analyze all segments**: Some information may be spread across multiple chunks. Synthesize them into a single, cohesive answer.
    2. **Be Comprehensive**: Include all specific details, dates, names, and requirements mentioned in the context that are relevant to the question.
    3. **Accuracy First**: Do not infer or assume information not explicitly stated. If the context is insufficient to provide a full answer, state what is available and note what is missing.
    4. **Tone**: Professional, direct, and factual.

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


def format_context(passages: list[dict]) -> str:
    blocks = []
    for passage in passages:
        metadata = passage.get("metadata") or {}
        blocks.append(
            f"Content: {passage.get('text', '')}\n"
            f"Title: {metadata.get('title', 'Unknown')}\n"
            f"Page: {metadata.get('page', '?')}"
        )
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


def build_sources(passages: list[dict]) -> list[dict]:
    sources = []
    for passage in passages:
        metadata = passage.get("metadata") or {}
        sources.append(
            {
                "title": metadata.get("title", "Unknown"),
                "page": metadata.get("page", "?"),
                "pdf_id": metadata.get("pdf_id", ""),
                "url": metadata.get("url", ""),
                "field_communication_type": metadata.get("field_communication_type", ""),
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
