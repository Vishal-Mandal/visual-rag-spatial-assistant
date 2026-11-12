import streamlit as st
import os
import shutil
import time

# Import pipeline functions
from extract_frames import extract_frames
from describe_frames import describe_all_frames
from build_vector_db import build_database
from query_pipeline import query_visual_rag

# Page configuration
st.set_page_config(
    page_title="Visual RAG Spatial Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #F8F9FA;
        border-left: 5px solid #FF4B4B;
        padding: 1.5rem;
        border-radius: 0 10px 10px 0;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    @media (prefers-color-scheme: dark) {
        .answer-box {
            background-color: #262730;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper function to delete old directories and clear session
def clear_existing_data():
    # Delete frames and descriptions files
    paths_to_delete = ["extracted_frames", "descriptions.json"]
    for path in paths_to_delete:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
                
    # Under Windows, shutil.rmtree("chroma_db") throws WinError 32 (file lock) because Python
    # holds open connections to the SQLite database. Instead, we dynamically load the DB and
    # delete the collection, which clears the data cleanly without deleting the files on disk.
    if os.path.exists("chroma_db"):
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings
            )
            vector_store.delete_collection()
        except Exception as e:
            st.warning(f"Note: Could not reset vector store automatically: {e}")
            
    # Force a database reload
    st.cache_resource.clear()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=120)
    st.markdown("## 🧠 Visual RAG System")
    
    st.divider()
    
    # Selection for source video
    source_option = st.selectbox(
        "Choose Video Source",
        ["Default Pre-indexed Video", "Upload Your Own Video"]
    )
    
    st.divider()
    
    # Manage Uploader Option
    if source_option == "Upload Your Own Video":
        st.markdown("### 📤 Upload Video File")
        uploaded_file = st.file_uploader("Upload MP4 or MOV", type=["mp4", "mov"])
        
        if uploaded_file is not None:
            # Save uploaded video
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Video uploaded successfully!")
            
            # Warn user about rate limits / processing times
            st.warning(
                "⚠️ **Processing Info:** Visual descriptions are processed at 1 FPS using Gemini. "
                "Due to API rate limits (15 RPM), processing a 30-second video will take about **2 minutes**."
            )
            
            if st.button("🚀 Process & Index Video", use_container_width=True):
                # Run complete indexing pipeline
                with st.status("Indexing your video... Please do not close this tab.", expanded=True) as status:
                    # 1. Clean up old data
                    status.update(label="Cleaning up old indices...", state="running")
                    clear_existing_data()
                    
                    # 2. Extract frames
                    status.update(label="Extracting frames (1 frame per second)...", state="running")
                    extract_frames(video_path, "extracted_frames")
                    
                    # 3. Translate frames using VLM
                    status.update(label="Translating frames to text using Gemini (Rate limits applied)...", state="running")
                    describe_all_frames("extracted_frames", "descriptions.json")
                    
                    # 4. Cognitive Mapping (Vector DB)
                    status.update(label="Generating vector embeddings and building database...", state="running")
                    build_database("descriptions.json", "chroma_db")
                    
                    # Complete
                    status.update(label="Indexing complete! You can now query your video.", state="complete")
                    st.success("Database successfully updated!")
                    st.rerun()
                    
    else:
        st.markdown("### 📊 Database Statistics")
        st.markdown("- **Video File:** `room.mp4`")
        # Quick validation of existing data
        if os.path.exists("chroma_db"):
            st.markdown("- **Status:** Ready 🟢")
        else:
            st.markdown("- **Status:** Not built 🔴 (Please run build_vector_db.py)")
            
    st.divider()
    
    # Video preview player
    if source_option == "Default Pre-indexed Video" and os.path.exists("room.mp4"):
        st.markdown("### 📹 Source Video")
        st.video("room.mp4")
    elif source_option == "Upload Your Own Video" and os.path.exists("uploaded_video.mp4"):
        st.markdown("### 📹 Uploaded Video")
        st.video("uploaded_video.mp4")

# --- Main Page Layout ---
st.markdown('<div class="main-title">👁️ Visual RAG Spatial Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask about any object in the physical space. The system will search its memories, describe the location, and display the matching video frames.</div>', unsafe_allow_html=True)

# Main Query Section
if not os.path.exists("chroma_db"):
    st.info("👋 To begin, select your video source in the sidebar and ensure the video has been processed.")
else:
    st.success("Ready to Chat! 🟢 The spatial memory is fully indexed and loaded.")
    query = st.chat_input("Where did I leave my laptop? / Where is the red box?")
    
    if query:
        st.subheader(f"💬 Question: {query}")
        
        with st.spinner("Searching memories & reasoning..."):
            try:
                # Call pipeline function (loads database dynamically now)
                answer, metadata = query_visual_rag(query, k=2)
                
                # Output the natural language answer
                st.markdown("### 🤖 Spatial AI Reasoner")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Display matching frames
                if metadata:
                    st.markdown("### 📸 Supporting Visual Evidence")
                    cols = st.columns(len(metadata))
                    
                    for i, meta in enumerate(metadata):
                        with cols[i]:
                            img_path = meta["image_path"]
                            img_path = os.path.normpath(img_path)
                            
                            if os.path.exists(img_path):
                                st.image(
                                    img_path, 
                                    caption=f"Frame: {meta['filename']} | Timestamp: {meta['timestamp']}", 
                                    use_container_width=True
                                )
                            else:
                                st.warning(f"Image not found at path: {img_path}")
                else:
                    st.info("No supporting frames were retrieved for this query.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
