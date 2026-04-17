import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(BASE_DIR / "db/chroma_db")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def print_docs(title, docs):
    print(f"\n=== {title} ===")
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        preview = doc.page_content.replace("\n", " ")[:220]
        print(f"{i}. {preview}...")


def main():
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector DB not found at {PERSIST_DIRECTORY}. Run ingestion_pipline.py first."
        )

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=EMBEDDING_MODEL,
        collection_metadata={"hnsw:space": "cosine"},
    )

    query = "How much did Microsoft pay to acquire GitHub?"
    print(f"Query: {query}")

    # 1) Basic similarity retrieval
    retriever_similarity = db.as_retriever(search_kwargs={"k": 3})
    docs_similarity = retriever_similarity.invoke(query)
    print_docs("METHOD 1: Similarity Search (k=3)", docs_similarity)

    # 2) Similarity with score threshold
    retriever_threshold = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )
    docs_threshold = retriever_threshold.invoke(query)
    print_docs("METHOD 2: Similarity Score Threshold", docs_threshold)

    # 3) MMR retrieval to reduce redundancy
    retriever_mmr = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    )
    docs_mmr = retriever_mmr.invoke(query)
    print_docs("METHOD 3: MMR (diverse results)", docs_mmr)


if __name__ == "__main__":
    main()
