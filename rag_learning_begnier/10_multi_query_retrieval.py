import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(BASE_DIR / "db/chroma_db")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class QueryVariations(BaseModel):
    queries: List[str] = Field(
        description="3 rephrased search queries that preserve the original meaning"
    )


def get_gemini_llm(temperature=0):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your environment or .env file.")

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-002",
    ]

    last_error = None
    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key,
            )
            print(f"Using Gemini model: {model_name}")
            return llm
        except ChatGoogleGenerativeAIError as error:
            last_error = error
            print(f"Model '{model_name}' unavailable, trying next fallback...")

    raise RuntimeError(
        "No supported Gemini model was available for your API key/project. "
        "Update candidate model names in 10_multi_query_retrieval.py."
    ) from last_error


def generate_query_variations(llm, original_query):
    llm_with_tools = llm.with_structured_output(QueryVariations)

    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents.

Original query: {original_query}

Return exactly 3 alternative queries that rephrase or approach the same question from different angles."""

    try:
        response = llm_with_tools.invoke(prompt)
        variations = [q.strip() for q in response.queries if q.strip()]
    except ChatGoogleGenerativeAIError:
        print("Gemini quota/rate limit reached while generating variations. Using original query only.")
        variations = [original_query]

    if original_query not in variations:
        variations.insert(0, original_query)

    # Keep only unique queries while preserving order
    seen = set()
    unique = []
    for query in variations:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append(query)

    return unique[:4]


def dedupe_documents(docs):
    unique_docs = []
    seen_content = set()
    for doc in docs:
        key = doc.page_content.strip().lower()
        if key and key not in seen_content:
            seen_content.add(key)
            unique_docs.append(doc)
    return unique_docs


def main():
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector DB not found at {PERSIST_DIRECTORY}. Run ingestion_pipline.py first."
        )

    llm = get_gemini_llm(temperature=0)
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDING_MODEL,
        collection_metadata={"hnsw:space": "cosine"},
    )

    original_query = "How does Tesla make money?"
    print(f"Original Query: {original_query}\n")

    variations = generate_query_variations(llm, original_query)
    print("Generated Query Variations:")
    for i, variation in enumerate(variations, 1):
        print(f"{i}. {variation}")

    print("\n" + "=" * 60)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    all_results = []

    for i, query in enumerate(variations, 1):
        docs = retriever.invoke(query)
        all_results.extend(docs)
        print(f"Query {i} returned {len(docs)} docs")

    unique_docs = dedupe_documents(all_results)
    top_docs = unique_docs[:8]

    print("\nTop Combined Context:")
    for i, doc in enumerate(top_docs, 1):
        preview = doc.page_content.replace("\n", " ")[:180]
        print(f"{i}. {preview}...")

    context = "\n\n".join(doc.page_content for doc in top_docs)
    final_prompt = f"""You are an assistant for question-answering tasks.
Use the following retrieved context to answer the user question.
If you do not know the answer, say you do not know.

Context:
{context}

Question: {original_query}

Answer:"""

    try:
        result = llm.invoke(final_prompt)
        print("\n=== FINAL ANSWER ===")
        print(result.content)
    except ChatGoogleGenerativeAIError:
        print("\nGemini quota/rate limit reached. Showing retrieved context only.")


if __name__ == "__main__":
    main()
