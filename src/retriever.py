class Retriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query, top_k=3):
        q_emb = self.embedder.embed_documents([query])[0]
        return self.vector_store.search(q_emb, top_k=top_k)
