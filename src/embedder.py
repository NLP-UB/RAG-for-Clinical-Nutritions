from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='mixedbread-ai/mxbai-embed-large-v1'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=True)
