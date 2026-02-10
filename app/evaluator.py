import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.ingestion import get_retriever

def evaluate(args, kwargs, result):
    user_input = args[0] if args else kwargs.get("user_input")
    retriever = get_retriever()
    evaluation = evaluate_answer(user_input=user_input,
                    answer=result,
                    retriever=retriever)
    formatted_evaluation = json.loads(evaluation)

def evaluate_answer(user_input, answer, retriever, model="gpt-3.5-turbo"):
    llm = ChatOpenAI(
        model=model,
        temperature=0.3,
        max_tokens=1000
    )

    context_docs = retriever.invoke(user_input)
    context_text = "\n".join([doc.page_content for doc in context_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator LLM. Your job is to check the quality of an AI-generated answer based on the provided context from a document. Check the answer's relevance, correctness, and grounding."),
        ("human", """Guidelines:
            1. Check if the answer is **directly related to the question AND supported by the provided context**.
            2. If the answer says the information is not in the context, mark:
               - grounded: false
               - relevance: low
            3. If the answer partially uses the context but misses details, mark relevance: medium.
            4. If the answer fully uses the context to answer the question, mark relevance: high.
            5. Identify hallucinations or invented information.
            6. Assign a confidence score between 0 and 1 (1 = fully confident, 0 = not confident).
            7. Be concise and structured. Respond in JSON format exactly like this:
               {{
                 "grounded": true/false,
                 "relevance": "high/medium/low",
                 "hallucination_risk": "low/medium/high",
                 "confidence_score": 0.xx
               }}
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            {answer}
            """)
    ])

    chain = prompt | llm | StrOutputParser()

    evaluation = chain.invoke({
        "context": context_text,
        "question": user_input,
        "answer": answer
    })
    return evaluation
