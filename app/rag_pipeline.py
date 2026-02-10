from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.evaluator import evaluate_answer


def evaluation_temp(user_input: str, retriever):
    answer = rag_pipeline_response(user_input=user_input, retriever=retriever)
    return evaluate_answer(user_input, answer,retriever)


def rag_pipeline_response(user_input: str, retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using ONLY the context below.\n"
         "2. Include relevant details, numbers, and explanations for clarity.\n"
         "3. Use related information from context to give a fuller picture.\n"
         "4. Summarize long info concisely, preferably in bullets or numbered steps.\n"
         "5. Do NOT use any knowledge outside the context. Do not make assumptions.\n"
         "6. If information is not in the context, say politely that it’s not available.\n"
         "7. Indicate clearly which statements are directly supported by the context.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(user_input)