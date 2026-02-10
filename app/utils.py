def format_docs(docs):
    return "\n\n".join(doc.page_context for doc in docs)