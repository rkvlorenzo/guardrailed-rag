from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def get_eval_prompt(context, question, answer):
    return f"""
        Guidelines:
        1. Check if the answer is **grounded** in the context. Only information explicitly in the context counts.
        2. Check if the answer is **relevant** to the question.
        3. Identify if there is any **hallucination** or invented information.
        4. Assign a **confidence score** between 0 and 1 (1 = fully confident, 0 = not confident).
        5. Be concise and structured. Respond in **JSON format** exactly like this:
        
        Format it in JSON format with the following keys/value:
        1. grounded: true/false,
        2. relevance: high/medium/low,
        3. hallucination_risk: low/medium/high,
        4. confidence_score: 0.xx
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        {answer}
    """

def evaluate_answer(user_input, answer, retriever):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
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
    return evaluation, answer
