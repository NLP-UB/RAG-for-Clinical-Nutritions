from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import uuid
import os

class VectorStore:
    def __init__(self, dimension, collection_name="test", storage_path="./qdrant_storage", recreate_on_dimension_mismatch=False):
        os.makedirs(storage_path, exist_ok=True)
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection only if it does not exist
        collections = [col.name for col in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
        else:
            current_dimension = self._get_collection_dimension()
            if current_dimension != self.dimension:
                if recreate_on_dimension_mismatch:
                    print(
                        f"Collection '{self.collection_name}' dimension mismatch "
                        f"({current_dimension} -> {self.dimension}). Recreating collection."
                    )
                    self.client.recreate_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                    )
                else:
                    raise ValueError(
                        f"Collection '{self.collection_name}' uses dimension {current_dimension}, "
                        f"but embedder produces {self.dimension}. "
                        "Use a different collection_name, delete/recreate this collection, "
                        "or set recreate_on_dimension_mismatch=True."
                    )

    def _get_collection_dimension(self):
        info = self.client.get_collection(self.collection_name)
        vectors_cfg = info.config.params.vectors
        if hasattr(vectors_cfg, "size"):
            return vectors_cfg.size
        if isinstance(vectors_cfg, dict):
            first_vector = next(iter(vectors_cfg.values()))
            if hasattr(first_vector, "size"):
                return first_vector.size
        raise ValueError(f"Unable to determine vector dimension for collection '{self.collection_name}'.")

    def add(self, embeddings, texts):
        points = []
        for emb, text in zip(embeddings, texts):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(id=point_id, vector=np.array(emb, dtype=np.float32), payload={"text": text})
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k=3):
        query_vector = np.array(query_embedding, dtype=np.float32)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [(hit.payload.get("text", "Text not found"), hit.score) for hit in results]
