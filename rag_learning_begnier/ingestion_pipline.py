import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(docs_path):
    print(f"Checking for documents in: {os.path.abspath(docs_path)}")
   # to load all the txt file from the docs
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()
    print(f"total document loded: {len(documents)}")
    
    if len(documents) == 0:
        print(f"No documents found in the specified path: {docs_path}")
        return [] 
    
    # to print a preview of the documents
    for i, doc in enumerate(documents[:2]): # to show first 2 document 
      print(f"\nDocument {i+1}:")
      print(f" source: {doc.metadata["source"]}")
      print(f" content length: {len(doc.page_content)} chareters")
    print(f"Content preview: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
        
    return documents
def split_documents(documents,chunk_size=800,chunk_overlap=0):
    # split larg document into smaller chunk 
     print("Splitting documents into chunk... ")
     text_splitter=CharacterTextSplitter(
         separator="\n",
         chunk_size=chunk_size,
         chunk_overlap=chunk_overlap
     )
     chunks= text_splitter.split_documents(documents)
     if chunks:
         for i,chunk in enumerate(chunks[:5]):
            print(f"\n---chunk {i+1}---")
            print(f"source: {chunk.metadata['source']}")
            print(f"length: {len(chunk.page_content)}  characters")
            print(f"content:")
            print(chunk.page_content)
            print("-"*50)

            if len(chunks)>5:
                print(f"\n.. and{len(chunks)-5}more chunks")
            return chunks
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating local embeddings and storing in Chroma...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
    print("creating a vector store")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore
             

     
         
def main():
    # This is where the function run
    #load the file
    documents = load_documents(docs_path="docs")
    #to chunk file
    chunks=split_documents(documents)
    # embedding and storing in vector db 
    vectorstore=create_vector_store(chunks)
    print("\n--- Pipeline Success! ---")
    

 


# This tells Python to run the main() function when  start the script
if __name__ == "__main__":
    main()
   


