import re
# from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF

def load_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(
    text,
    chunk_size=500,
    overlap=50,
    method="character",  # Options: "character", "recursive", "document", "semantic"
    separators=["\n\n", "\n", ".", " ", ""],
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
):
    """
    Splits text into chunks based on the selected chunking method.

    Parameters:
        text (str): The input text to split
        chunk_size (int): Max characters per chunk (for non-semantic methods)
        overlap (int): Overlap between chunks
        method (str): Chunking method name
                      Options:
                        - "character"  : Simple static character chunks
                        - "recursive"  : Recursive chunking based on separators
                        - "document"   : Heuristics for PDFs, Markdown, or code blocks
                        - "semantic"   : Embedding-based semantic chunking
        separators (list): Separators for recursive splitting
        model_name (str): Model for semantic similarity (only used if method="semantic")

    Returns:
        list: List of text chunks
    """

    chunks = []

    if method == "recursive":
        def recursive_split(text, separators):
            if not separators:
                return [text]
            sep = separators[0]
            if sep and sep in text:
                parts = text.split(sep)
            else:
                parts = [text]
            result = []
            for part in parts:
                if len(part) > chunk_size and len(separators) > 1:
                    result.extend(recursive_split(part, separators[1:]))
                else:
                    result.append(part.strip())
            return result

        all_parts = recursive_split(text, separators)
        temp = ""
        for part in all_parts:
            if len(temp) + len(part) + 1 <= chunk_size:
                temp += part + " "
            else:
                chunks.append(temp.strip())
                temp = part
        if temp:
            chunks.append(temp.strip())

    else:
        raise ValueError(f"Invalid chunking method: {method}. Choose from "
                         f"['character', 'recursive', 'document', 'semantic'].")

    return [c.strip() for c in chunks if c.strip()]
