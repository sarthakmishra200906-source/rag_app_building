import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = str(BASE_DIR / "docs")
PERSIST_DIRECTORY = str(BASE_DIR / "db/chroma_db")
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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
        "Update candidate model names in 8_multi_modal_rag.py."
    ) from last_error


def load_documents(docs_path):
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs


def split_documents(documents, chunk_size=900, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def summarize_chunks_with_gemini(chunks, llm, max_chunks=20):
    processed = []

    for i, chunk in enumerate(chunks):
        raw_text = chunk.page_content.strip()
        if not raw_text:
            continue

        if i < max_chunks:
            prompt = f"""Create a searchable summary of this chunk.
Include key facts, entities, numbers, and alternate search terms.

Chunk:\n{raw_text}\n
Searchable summary:"""
            try:
                summary = llm.invoke(prompt).content
            except ChatGoogleGenerativeAIError:
                # Fall back to raw content when quota/rate limits are hit.
                summary = raw_text
        else:
            summary = raw_text

        processed.append(
            Document(
                page_content=summary,
                metadata={
                    "source": chunk.metadata.get("source", "unknown"),
                    "original_content": json.dumps({"raw_text": raw_text}, ensure_ascii=False),
                },
            )
        )

    print(f"Prepared {len(processed)} enhanced chunks")
    return processed


def create_vector_store(documents, persist_directory):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=EMBEDDING_MODEL,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store saved to {persist_directory}")
    return vectorstore


def chunks_to_documents(chunks):
    docs = []
    for chunk in chunks:
        raw_text = chunk.page_content.strip()
        if not raw_text:
            continue
        docs.append(
            Document(
                page_content=raw_text,
                metadata={
                    "source": chunk.metadata.get("source", "unknown"),
                    "original_content": json.dumps({"raw_text": raw_text}, ensure_ascii=False),
                },
            )
        )
    return docs


def answer_question(db, llm, query, k=4):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        print("No relevant context found")
        return

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Answer the question using only this context.
If the answer is not in the context, say you do not know.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        print("\n=== ANSWER ===")
        print(response.content)
    except ChatGoogleGenerativeAIError:
        print("\nGemini quota/rate limit reached. Showing top retrieved context instead:")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content.replace("\n", " ")[:200]
            print(f"{i}. {preview}...")


def main():
    print("Lesson 8 (adapted): Gemini + open-source embeddings pipeline")

    documents = load_documents(DOCS_PATH)
    if not documents:
        raise FileNotFoundError(f"No .txt documents found in {DOCS_PATH}")

    chunks = split_documents(documents)
    use_ai_summaries = os.getenv("USE_GEMINI_SUMMARY", "0") == "1"

    if use_ai_summaries:
        llm = get_gemini_llm(temperature=0)
        # Keep summary count low to avoid hitting free-tier Gemini limits.
        enhanced_chunks = summarize_chunks_with_gemini(chunks, llm, max_chunks=5)
    else:
        llm = get_gemini_llm(temperature=0)
        enhanced_chunks = chunks_to_documents(chunks)

    db = create_vector_store(enhanced_chunks, PERSIST_DIRECTORY)

    query = "How does Tesla make money?"
    answer_question(db, llm, query)


if __name__ == "__main__":
    main()
