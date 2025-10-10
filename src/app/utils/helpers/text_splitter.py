import os
import re
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def split_member_blocks(text: str) -> List[str]:
    text = re.sub(r'\r\n', '\n', text).strip()

    pattern = r"(?P<num>\d{1,2})\.\s*Name:"

    matches = list(re.finditer(pattern, text))
    blocks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        blocks.append(block)
    return blocks

def split_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return [n.text for n in nodes]


def split_fallback(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return [n.text for n in nodes]

