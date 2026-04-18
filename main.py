import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()

)


# Custom Groq adapter for Inngest
class GroqAdapter(ai.BaseAdapter):
    def __init__(self, auth_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=auth_key)
        self.model = model

    async def infer(self, body: dict) -> dict:
        # Convert OpenAI format to Groq format
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature", 0.7)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Convert Groq response to OpenAI format (so Inngest understands it)
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }




@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs= embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

# In theory we'll be able to see the results running inside of ingest and we'll be able to unsert this into the qudrant database !
    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

# This next function is gonna allow us to query our pdfs:
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    from groq import Groq

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    def _ask_llm(question: str, contexts: list[str]) -> str:
        # Call Groq directly
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        context_block = "\n\n".join(f"- {c}" for c in contexts)
        user_content = (
            "Use the following context to answer the question. \n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question}\n"
            "Answer concisely using the context above."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1024,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Step 1: Search the vector database
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Step 2: Ask Groq to answer (wrap in step.run for retries)
    answer = await ctx.step.run("llm-answer", lambda: _ask_llm(question, found.contexts))

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}



app = FastAPI()



inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
