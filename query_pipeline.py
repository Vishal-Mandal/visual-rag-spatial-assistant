import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# Load env variables (for Gemini API)
load_dotenv()

# Initialize embeddings and load existing DB
DB_DIR = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Gemini Client
client = genai.Client()

def query_visual_rag(user_query, k=2):
    # Load database dynamically
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    # 1. Retrieve matching documents from Chroma DB
    print(f"\n[Retrieving top {k} memories for query: '{user_query}'...]")
    retrieved_docs = vector_store.similarity_search(user_query, k=k)
    
    if not retrieved_docs:
        return "No relevant memories found.", []

    # 2. Format the retrieved memories for the LLM
    context_str = ""
    retrieved_metadata = []
    
    for i, doc in enumerate(retrieved_docs):
        timestamp = doc.metadata.get("timestamp", "unknown")
        image_path = doc.metadata.get("image_path", "unknown")
        filename = doc.metadata.get("filename", "unknown")
        
        # Save metadata for UI / returning
        retrieved_metadata.append({
            "timestamp": timestamp,
            "image_path": image_path,
            "filename": filename
        })
        
        context_str += f"Memory #{i+1} (Observed at Timestamp {timestamp}, Image: {filename}):\n"
        context_str += f"{doc.page_content}\n"
        context_str += "-" * 50 + "\n"

    # 3. Create the prompt for the reasoning LLM
    prompt = f"""
You are an Embodied AI spatial assistant. A user is asking about the location of an object in their room.
Use the following retrieved visual memories (which are text descriptions of the room at specific timestamps) to answer their question.

Retrieved Memories:
{context_str}

User Question: {user_query}

Provide a friendly, helpful response. Specifically tell the user:
1. Exactly where the object is located in the room based on the memories.
2. The surrounding context (what is next to or near it).
3. The exact timestamp(s) when the object was observed (e.g. "at timestamp 00:49").

If the retrieved memories do not contain the object or if the information is too vague to locate it, state that honestly. Do not make up information that isn't in the memories.

Answer:
"""

    # 4. Generate answer using Gemini 2.5 Flash (or gemini-flash-lite-latest)
    # We use gemini-flash-lite-latest because it is extremely fast and free
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt
    )
    
    return response.text.strip(), retrieved_metadata

if __name__ == "__main__":
    print("\n=== Visual RAG Interactive Assistant ===")
    print("Ask any question about your room (e.g., 'Where is the red box?' or 'Where did I leave the laptop?')")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nAsk: ")
            if query.strip().lower() == "exit":
                break
            if not query.strip():
                continue
                
            answer, metadata = query_visual_rag(query)
            
            print("\n=== AI Spatial Reasoner ===")
            print(answer)
            print("\n=== Supporting Frames ===")
            for meta in metadata:
                print(f"- {meta['filename']} at {meta['timestamp']} ({meta['image_path']})")
                
        except KeyboardInterrupt:
            break
            
    print("\nExiting Assistant. Goodbye!")
