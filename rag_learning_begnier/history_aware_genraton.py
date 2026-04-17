import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


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
        "Update candidate model names in history_aware_genraton.py."
    ) from last_error

#  Connect to  document database using  local embeddings
persistent_directory = "db/chroma_db"
# This matches the model you used to build the database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=str(BASE_DIR / "db/chroma_db"), embedding_function=embeddings)

#  Set up Gemini using same fallback strategy as retrieval pipeline
model = get_gemini_llm(temperature=0.3)

# Store conversation history
chat_history = []

def ask_question(user_question):
    print(f"\n--- Processing: {user_question} ---")
    
    #  Query Reformulation (using Chat History)
   
    if chat_history:
        reformulate_prompt = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        try:
            result = model.invoke(reformulate_prompt)
            search_question = result.content.strip()
            print(f"Standalone Search Query: {search_question}")
        except ChatGoogleGenerativeAIError:
            print("Gemini quota/rate limit reached for query rewrite. Using original question.")
            search_question = user_question
    else:
        search_question = user_question
    
    #  Retrieve from ChromaDB
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Retrieved {len(docs)} snippets from your local docs.")
    
    #  Create RAG Prompt
    context_data = "\n".join([f"- {doc.page_content}" for doc in docs])
    
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {context_data}

    Instructions:
    Use ONLY the documents above. If the answer isn't there, say you don't have enough info.
    """
    
    # Final Generation
    messages = [
        SystemMessage(content="You are a precise technical assistant for the Code Dazzlers team."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    print("Gemini is thinking...")
    try:
        result = model.invoke(messages)
        answer = result.content
    except ChatGoogleGenerativeAIError:
        answer = "Gemini quota/rate limit reached. I can only show retrieval context right now."
    
    #  Update History
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nAnswer: {answer}")
    return answer

#  chat loop
def start_chat():
    print("\n RAG Chat Active (Gemini)")
    print("Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            print("Goodbye, Sarthak!")
            break
        if not question.strip():
            continue
            
        try:
            ask_question(question)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()