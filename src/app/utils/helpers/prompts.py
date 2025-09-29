RAG_BASE_PROMPT = (
    "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
    "If the answer cannot be found in the context, reply exactly: 'I don't know.'\n\n"
    "CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer:"
)
