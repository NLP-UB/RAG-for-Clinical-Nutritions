from langchain_ollama import OllamaEmbeddings

class Embedder:
    def __init__(self, model_name="embeddinggemma"):
        self.model = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
        self._embedding_dim = None

    def embed_query(self, text: str):
        # langsung return embedding dari query
        return self.model.embed_query(text)

    def embed_documents(self, texts: list):
        # embed_documents sudah disediakan
        return self.model.embed_documents(texts)

    def embed_text_with_values(self, text: str):
        """
        Takes a single text string and returns a dictionary with:
        - 'text': the original text
        - 'embedding': the embedding vector from OllamaEmbeddings
        """
        emb = self.embed_query(text)
        return {"text": text, "embedding": emb}

    def embedding_dimension(self):
        if self._embedding_dim is None:
            self._embedding_dim = len(self.embed_query("dimension probe"))
        return self._embedding_dim
