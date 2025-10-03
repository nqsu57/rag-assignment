RAG_BASE_PROMPT = """You are a helpful assistant. 
Use ONLY the context below to answer the question.
If the answer cannot be found in the context, reply exactly: "I don't know."

CONTEXT:
{context}

QUESTION:
{query}

Answer:
"""