from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from src.app.utils.logger import get_logger
from src.app.models.settings import settings 

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = getattr(settings, "chunk_size", 512)
DEFAULT_CHUNK_OVERLAP = getattr(settings, "chunk_overlap", 50)


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


def chunk_documents(
    docs: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_sentence_splitter: bool = False,
) -> List[Dict[str, Any]]:
 
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    effective_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    if effective_overlap >= effective_chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if use_sentence_splitter:
        parser = SentenceSplitter(chunk_size=effective_chunk_size, chunk_overlap=effective_overlap)
    else:
        # SimpleNodeParser.from_defaults uses SentenceSplitter by default internally
        parser = SimpleNodeParser.from_defaults(chunk_size=effective_chunk_size, chunk_overlap=effective_overlap)

    nodes = parser.get_nodes_from_documents(docs)
    logger.info(f"Parser produced {len(nodes)} nodes (chunks)")

    chunks: List[Dict[str, Any]] = []
    # for i, node in enumerate(nodes):
    #     # node likely has .text and .extra_info / .metadata depending on LlamaIndex version
    #     text = getattr(node, "text", None) or getattr(node, "get_text", lambda: str(node))()
    #     metadata = {}
    #     # try common metadata attributes; keep it defensive
    #     if hasattr(node, "doc_id"):
    #         metadata["doc_id"] = getattr(node, "doc_id")
    #     if hasattr(node, "extra_info"):
    #         metadata.update(getattr(node, "extra_info") or {})
    #     if hasattr(node, "metadata"):
    #         # some versions store Document.metadata
    #         metadata.update(getattr(node, "metadata") or {})

    #     chunk_id = f"chunk-{i}"
    #     chunks.append({"id": chunk_id, "text": text, "metadata": metadata})

    for i, node in enumerate(nodes):
        text = getattr(node, "text", None) or getattr(node, "get_text", lambda: str(node))()
        # merge metadata: extra_info trước, metadata sau
        meta: dict = {}
        if hasattr(node, "extra_info") and getattr(node, "extra_info"):
            meta.update(getattr(node, "extra_info"))
        if hasattr(node, "metadata") and getattr(node, "metadata"):
            meta.update(getattr(node, "metadata"))
        if hasattr(node, "doc_id"):
            meta.setdefault("doc_id", getattr(node, "doc_id"))
        chunks.append({"id": f"chunk-{i}", "text": text, "metadata": meta})
    return chunks

def save_chunks(chunks: List[Dict[str, Any]], out_dir: Path, basename: str = "chunks.json"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / basename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {out_path}")
    return out_path
