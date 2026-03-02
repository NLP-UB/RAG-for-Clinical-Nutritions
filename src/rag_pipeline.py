import os
import glob
from .loader import load_pdf, chunk_text
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator
from .ner_processor import NERProcessor

class RAGPipeline:
    def __init__(self, data_path="data", embed_model='embeddinggemma', gen_model='llama3.2:3b',
                 storage_path="./qdrant_storage", collection_name="gizi_klinis", method="recursive",
                 recreate_on_dimension_mismatch=False):
        """
        RAG Pipeline that uses Qdrant persistent storage for embeddings.

        Args:
            data_path (str): Directory containing PDF files for initial indexing
            embed_model (str): Sentence Transformer model for embeddings
            gen_model (str): Text generation model
            storage_path (str): Local folder for persistent Qdrant storage
            collection_name (str): Name of Qdrant collection
        """
        self.embedder = Embedder(embed_model)
        self.generator = Generator(gen_model)
        embedding_dim = self.embedder.embedding_dimension()
        self.vector_store = VectorStore(
            embedding_dim,
            storage_path=storage_path,
            collection_name=collection_name,
            recreate_on_dimension_mismatch=recreate_on_dimension_mismatch,
        )
        self.retriever = Retriever(self.vector_store, self.embedder)
        self.collection_name = collection_name
        self.method = method
        self.ner = NERProcessor()

        # Perform indexing only once at initialization (if data_path provided)
        if data_path and self._is_vector_store_empty():
            self._index_all_pdfs(data_path)
        else:
            print("Using existing Qdrant vector store, skipping indexing.")

    def _is_vector_store_empty(self):
        """Check if the vector store collection already contains points."""
        try:
            count = self.vector_store.client.count(collection_name=self.collection_name).count
            return count == 0
        except Exception:
            return True

    def _index_all_pdfs(self, data_path):
        """
        Load all PDF documents in the directory (including subfolders),
        split into chunks, embed, and store in Qdrant.
        """
        pdf_files = glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True)

        if not pdf_files:
            print(f"No PDF files found in directory: {data_path}")
            return

        all_chunks = []
        all_embeddings = []

        print(f"Found {len(pdf_files)} PDF(s). Processing...")
        for file_path in pdf_files:
            try:
                print(f"🔹 Reading: {file_path}")
                text = load_pdf(file_path)
                if not text.strip():
                    print(f"Skipping empty PDF: {file_path}")
                    continue

                chunks = chunk_text(text, method=self.method, chunk_size=500, overlap=50)
                # chunks = self.ner.process_chunks(chunks)

                embeddings = self.embedder.embed_documents(chunks)

                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if all_chunks:
            self.vector_store.add(all_embeddings, all_chunks)
            print(f"Indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs into persistent Qdrant storage.")
        else:
            print("No valid PDF content to index.")

    def answer_question(self, query, top_k=3, use_rag=True):
        """
        Generate an answer with or without RAG retrieval.
        """
        if not use_rag:
            return self.answer_question_without_rag(query), []

        retrieved = self.retriever.retrieve(query, top_k)
        contexts = self._extract_contexts(retrieved)
        context = " ".join([c[:800] for c in contexts])
        try:
            answer = self.generator.generate(context, query)
        except Exception:
            answer = ""
        return answer, retrieved

    def answer_question_without_rag(self, query):
        """Generate answer directly from the LLM without retrieval."""
        try:
            return self.generator.generate("", query)
        except Exception:
            return ""

    @staticmethod
    def _extract_contexts(retrieved):
        """Normalize retriever output into a list of context strings."""
        return [row[0] for row in retrieved if row and len(row) > 0]

    def close(self):
        """Close underlying vector store client."""
        self.vector_store.client.close()
