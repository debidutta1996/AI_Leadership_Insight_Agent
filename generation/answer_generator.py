from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(question, context_chunks, metadatas, scores):

    # If nothing retrieved
    if not context_chunks:
        return "Based on the available documents, this information is not currently available."

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI Leadership Insight Agent.

Answer ONLY the question asked.
Do NOT provide extra summaries.
Do NOT include unrelated insights.
Do NOT speculate.

Rules:
- Use strictly the provided context.
- If answer is not explicitly present in context, say:
  "The information is not available in the provided documents."
- Keep the response concise and factual.
- Maximum 3 sentences.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You provide precise, grounded, leadership-focused answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    fallback_phrases = [
        "not available",
        "not present",
        "not provided",
        "not mentioned",
        "do not contain"
    ]

    if any(phrase in answer.lower() for phrase in fallback_phrases):
        return "Based on the available documents, this information is not currently available."

    unique_sources = list(set([meta["source"] for meta in metadatas]))

    return f"{answer}\n\nSources:\n" + "\n".join(unique_sources)