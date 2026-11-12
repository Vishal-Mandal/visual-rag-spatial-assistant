import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def build_database(json_path, db_dir):
    # 1. Load the generated descriptions
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run describe_frames.py first.")
        return

    with open(json_path, 'r') as f:
        descriptions_data = json.load(f)
    
    print(f"Loaded {len(descriptions_data)} descriptions from JSON.")

    # 2. Convert to LangChain Document objects
    documents = []
    for filename, info in descriptions_data.items():
        doc = Document(
            page_content=info["description"],
            metadata={
                "image_path": info["image_path"],
                "timestamp": info["timestamp"],
                "filename": filename
            }
        )
        documents.append(doc)
    
    print(f"Converted descriptions into {len(documents)} Document objects.")

    # 3. Initialize HuggingFace Embeddings (runs locally!)
    print("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    # This downloads the model to your cache on the first run
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and persist the Chroma DB
    print(f"Creating local Chroma DB in folder '{db_dir}'...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print("Vector database built and saved successfully!")

    # 5. Run a quick validation test
    print("\n--- Running Validation Query ---")
    test_query = "Where is the red box?"
    print(f"Query: '{test_query}'")
    
    # Retrieve the top 2 matching documents
    results = vector_store.similarity_search(test_query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\nMatch {i+1}:")
        print(f"  File: {doc.metadata.get('filename')}")
        print(f"  Timestamp: {doc.metadata.get('timestamp')}")
        print(f"  Image Path: {doc.metadata.get('image_path')}")
        print(f"  Excerpt: {doc.page_content[:150]}...")

if __name__ == "__main__":
    JSON_PATH = "descriptions.json"
    DB_DIR = "chroma_db"
    
    build_database(JSON_PATH, DB_DIR)
