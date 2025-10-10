import json
import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, Document
from src.app.utils.logger import get_logger
from src.app.utils.helpers.text_splitter import split_fallback, split_member_blocks, split_sentences

logger = get_logger(__name__)
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))


def load_documents_from_dir(data_dir: Path) -> List[Document]:

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found")
    reader = SimpleDirectoryReader(str(data_dir))
    docs = reader.load_data()
    if not docs:
        logger.warning(f"No documents found in {data_dir}") 
        return [] 
    logger.info(f"Loaded {len(docs)} documents from {data_dir}")
    return docs

def chunk_documents(docs: List[Document],
                    chunk_size: Optional[int] = None,
                    chunk_overlap: Optional[int] = None,
                    mode: str = "auto") -> List[Dict[str, Any]]:
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for doc in docs:
        text = doc.text.strip()
        metadata = getattr(doc, "metadata", {})
        try:
            if mode in ("auto", "member") and re.search(r"\d+\.\s*Name:", text):
                parts = split_member_blocks(text)
            elif mode in ("auto", "sentence"):
                parts = split_sentences(text, size, overlap)
                # print("Use sentences")
            else:
                parts = split_fallback(text, size, overlap)
                # print("Use fallback")
            for p in parts:
                chunks.append({
                    "id": f"chunk-{chunk_id}",
                    "text": p,
                    "metadata": metadata,
                })
                chunk_id += 1

        except Exception as e:
            logger.error(f"Error chunking document: {e}", exc_info=True)
            continue

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} document(s).")
    return chunks

def save_chunks(chunks: List[Dict[str, Any]], out_dir: Path, basename: str = "chunks.json"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / basename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(json.dump(chunks, f, ensure_ascii=False, indent=2))
    print(out_path)
    logger.info(f"Saved {len(chunks)} chunks to {out_path}")
    return out_path
