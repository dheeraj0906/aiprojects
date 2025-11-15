import os
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class RAGEngine:
    def __init__(
        self,
        embed_model: str = "text-embedding-004",
        gen_model: str = "gemini-2.5-flash"
    ):
        # Embeddings via LangChain adapter
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embed_model)

        # New SDK client
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gen_model = gen_model

        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        """Embed text chunks and build a FAISS cosine-sim index."""
        self.text_chunks = chunks
        vecs = np.array(self.embedding_model.embed_documents(chunks), dtype="float32")
        faiss.normalize_L2(vecs)
        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vecs)
        print(f"Index built with {self.index.ntotal} vectors.")

    def query(self, question: str, top_k: int = 5) -> str:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        qv = np.array(self.embedding_model.embed_query(question), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(qv)

        _, idxs = self.index.search(qv, top_k)
        hits = [self.text_chunks[i] for i in idxs[0] if i < len(self.text_chunks)]
        context = "\n\n---\n\n".join(hits)

        prompt = (
            "Answer ONLY using the context below. "
            "If the answer isn't present, say you can't find it.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        resp = self.client.models.generate_content(model=self.gen_model, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
