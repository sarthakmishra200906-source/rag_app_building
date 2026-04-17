import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import ChatPromptTemplate


def get_gemini_llm(temperature=0.3):
    """Create a Gemini client using a supported model name, with fallbacks."""
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
        "Update candidate model names in retrevalpipeline.py."
    ) from last_error


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
persisten_directory = str(BASE_DIR / "db/chroma_db")
# to load embedding and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
db=Chroma(
    persist_directory=persisten_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)
#serch for revlent document
llm = get_gemini_llm(temperature=0.3)
retriever =db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.3 #only return chunk with simalirity >=0.3
    }
)

template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


def answer_query(query):
    relevant_docs = retriever.invoke(query)
    print(f"\nUser Query: {query}\n")

    if not relevant_docs:
        print("No relevant context found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    print("Generating answer from Gemini...")
    final_prompt = prompt.format(context=context_text, question=query)
    try:
        response = llm.invoke(final_prompt)
        print("\n--- GEMINI'S ANSWER ---")
        print(response.content)
    except ChatGoogleGenerativeAIError:
        print("\nGemini quota/rate limit reached. Showing retrieved context instead:")
        for i, doc in enumerate(relevant_docs, 1):
            preview = doc.page_content.replace("\n", " ")[:200]
            print(f"{i}. {preview}...")


def main():
    print("\nRAG assistant is ready.")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("Enter your question: ").strip()

        if not user_query:
            print("Please enter a question.")
            continue

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer_query(user_query)


if __name__ == "__main__":
    main()



