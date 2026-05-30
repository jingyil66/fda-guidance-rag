import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flashrank import Ranker, RerankRequest
from backend.app.core.config import settings
from backend.app.services.retrieval_service import retrieve_embedding
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY or ""

client = QdrantClient(url="http://localhost:6333")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

template = """
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
rag_prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model='gpt-4o-mini')
str_parser = StrOutputParser()
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt/flashrank")

def get_answer(query: str, collection_name="test") -> dict:
    passages = retrieve_embedding(
        query,
        collection_name,
        top_k=20,
        client=client,
        embeddings=embeddings,
    )
    rerankrequest = RerankRequest(query=query, passages=passages)
    rerank_results = ranker.rerank(rerankrequest)
    top_5_results = rerank_results[:5]


    context = "\n\n".join([
        f"Content: {res['text']}\n"
        f"Title: {res['metadata'].get('title', 'Unknown')}\n"
        f"Page: {res['metadata'].get('page', '?')}"
        for res in top_5_results
    ])

    manual_rag_chain = rag_prompt | llm | str_parser

    answer = manual_rag_chain.invoke({
        "query": query,
        "context": context
    })

    sources = [
        {
            "title": res['metadata'].get("title", "Unknown"), # 改为 ['metadata']
            "page": res['metadata'].get("page", "?"),
            "pdf_id": res['metadata'].get("pdf_id", ""),
            "url": res['metadata'].get("url", ""),
            "field_communication_type": res['metadata'].get("field_communication_type", "")
        }
        for res in top_5_results # 确保遍历的是 Rerank 后的列表
    ]

    documents = [
        {
            "text": res['text'], 
            "metadata": res['metadata']
        } 
        for res in top_5_results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents
    }

if __name__ == "__main__":
    while True:
        query = input("User's query: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = get_answer(query)
        print("LLM answer:", result["answer"])
        print("Source chunks:", result["sources"])