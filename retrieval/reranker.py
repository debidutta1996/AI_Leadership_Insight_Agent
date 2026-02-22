from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def rerank(query, documents, metadatas, top_k=4):

    scored_docs = []

    for doc, meta in zip(documents, metadatas):

        prompt = f"""
Rate how relevant the following document is to the query.

Query:
{query}

Document:
{doc}

Return only a number between 0 and 10.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 0

        scored_docs.append((doc, meta, score))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[2], reverse=True)

    top_docs = scored_docs[:top_k]

    final_docs = [d[0] for d in top_docs]
    final_metas = [d[1] for d in top_docs]
    final_scores = [d[2] for d in top_docs]

    return final_docs, final_metas, final_scores