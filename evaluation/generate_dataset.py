import os
import json
import random
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import models
from backend.app.core.config import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ===== Configuration =====
collection_name = "experiment_chunk600_overlap200"
qdrant_url = "http://localhost:6333"

num_samples = 100
max_qa = 50

# ===== Initialization =====
client = QdrantClient(url=qdrant_url)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== Prompt =====
question_prompt = ChatPromptTemplate.from_template("""
You are generating a professional question based on the provided context.
If multiple sections are provided, create a question that connects information between them.

Context:
{context}

Return ONLY the question.
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a factual extraction tool. Your goal is to provide a Ground Truth answer based STRICTLY on the context.

Rules:
1. **No Outside Knowledge**: Do NOT explain terms or add examples (like "oncology" or "clinical trials") if they are not in the context.
2. **Strict Adherence**: Only include points explicitly stated in the text. 
3. **Concise structure**: Use bullet points for lists, but keep the descriptions short.
4. **Factual Density**: Focus on the requirements, definitions, and categories mentioned in the text.

Question: {query}
Context: {context}

Answer:
""")

q_chain = question_prompt | llm | StrOutputParser()
rag_chain = answer_prompt | llm | StrOutputParser()

# ===== 1. Retrieve Data =====

points, _ = client.scroll(
    collection_name=collection_name,
    limit=200,
    with_payload=True
)

dataset = []
already_used_ids = set()

# ===== 2. Generate QA =====
pbar = tqdm(total=max_qa)
while len(dataset) < max_qa:
    random.shuffle(points)
    
    for p in points:
        if p.id in already_used_ids or len(dataset) >= max_qa:
            continue
        
        meta = p.payload.get("metadata", {})
        pdf_id = meta.get("pdf_id")
        page = meta.get("page")
        
        context = p.payload["page_content"]
        already_used_ids.add(p.id)
        
        neighbor_points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="metadata.pdf_id", match=models.MatchValue(value=pdf_id)),
                    models.FieldCondition(key="metadata.page", match=models.MatchValue(value=page))
                ]
            ),
            limit=5
        )
        
        neighbors = [np for np in neighbor_points if np.id != p.id and np.id not in already_used_ids]
        if neighbors:
            n = neighbors[0]
            context = f"{context}\n\n{n.payload['page_content']}"
            already_used_ids.add(n.id)

        try:
            q = q_chain.invoke({"context": context}).strip()
            answer = rag_chain.invoke({"query": q, "context": context}).strip()
            
            if "not available" in answer.lower() or len(q) < 10:
                continue
                
            dataset.append({
                "question": q,
                "answer": answer,
                "context": context,
                "source_id": str(p.id)
            })
            pbar.update(1)
        except:
            continue

pbar.close()
# ===== 3. Save =====

with open("qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(dataset)} QA pairs")