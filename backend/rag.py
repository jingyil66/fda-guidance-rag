import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = QdrantClient(url="http://localhost:6333")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

template = """You are an AI model trained for question answering. You should answer the
given question based on the given context only.

Question: {query}

Context:
{context}
Only answer based on the context. Do NOT assume or infer anything not explicitly stated.
If the answer is not present in the given context, respond as: The answer to this question is not available
in the provided content.
"""
rag_prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model='gpt-4o-mini')
str_parser = StrOutputParser()

def get_answer(query: str) -> str:
    results = vector_store.similarity_search(query=query, k=15)
    context = "\n\n".join([result.page_content for result in results])

    manual_rag_chain = rag_prompt | llm | str_parser

    answer = manual_rag_chain.invoke({
        "query": query,
        "context": context
    })
    return answer