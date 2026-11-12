# 👁️ Visual RAG Spatial Assistant

Have you ever walked through a room while recording a quick video and later wished you could just ask an AI, *"Hey, where did I leave my laptop?"* or *"What's sitting next to the coffee table?"*—without scrubbing through the footage yourself?

That's exactly what this project is built for. **Visual RAG Spatial Assistant** turns raw room or environment videos into a searchable **spatial memory** using **Retrieval-Augmented Generation (RAG)**. Instead of stuffing an entire video into a large multimodal prompt every time you ask a question, the system converts video frames into rich spatial descriptions once, indexes them locally in a vector database, and retrieves only the most relevant "visual memories" to answer your questions with exact timestamps and supporting frames.

---

## 🧠 Core Concepts: How Visual RAG Works

Traditional RAG works on text documents (PDFs, notes, docs). **Visual RAG** extends that exact same retrieval-and-grounding pattern to video and physical spaces:

1. **Temporal Sampling (Video → Keyframes)**
   Processing every single frame of a 30 FPS video is redundant—most consecutive frames look nearly identical. Sampling **1 frame per second (1 FPS)** captures changes in viewpoint and object layout while keeping API usage and indexing time manageable.
2. **Visual-to-Text Translation (Vision-Language Model)**
   A multimodal model (**Gemini**) inspects each frame and writes a structured spatial description capturing physical objects, their colors, and their relationships to nearby items (for example, *what is on, beside, or behind what*).
3. **Cognitive Mapping (Semantic Embeddings + Vector Store)**
   Each frame's description is paired with metadata (`timestamp`, `filename`, `image_path`) inside a LangChain `Document`, converted into dense numerical vectors locally using **`all-MiniLM-L6-v2`**, and stored in **ChromaDB**. That gives the assistant semantic search—asking about *"a dark seat"* or *"where to sit"* can match frames mentioning an *"armchair"* or *"couch"*.
4. **Grounded Spatial Reasoning (Retrieve → Reason → Show Evidence)**
   When you ask a question, the system retrieves the top-`k` (`k=2` by default) closest memories from ChromaDB and feeds them to Gemini with strict grounding instructions: cite the exact timestamps, describe surrounding context, and honestly say if the memories don't contain the answer instead of guessing. The UI also shows the actual extracted frames alongside the answer so you can verify it with your own eyes.

### Pipeline Architecture

```text
  Video (.mp4 / .mov)
         │
         ▼  extract_frames.py (OpenCV @ 1 FPS)
  extracted_frames/frame_MM_SS.jpg
         │
         ▼  describe_frames.py (Google Gemini API: gemini-flash-lite-latest)
  descriptions.json  { image_path, timestamp, description }
         │
         ▼  build_vector_db.py (HuggingFace all-MiniLM-L6-v2 + ChromaDB)
  chroma_db/  (Persisted local vector store)
         │
         ▼  query_pipeline.py / app.py
  User Query ──► Similarity Search (top-k=2) ──► Gemini Spatial Reasoner ──► Answer + Supporting Frames
```

---

## 🛠️ Tools, Libraries & APIs Used

Here is a breakdown of every tool and API powering the project and why it's in the stack:

### APIs & Models

| Component | Model / Service | Role in the Project |
| :--- | :--- | :--- |
| **Google Gemini API** (`google-genai`) | `gemini-flash-lite-latest` | **1. Vision-Language Describer** (`describe_frames.py`): Reads each JPEG frame via PIL and generates detailed object + spatial relationship descriptions.<br>**2. Spatial Reasoner** (`query_pipeline.py`): Takes retrieved frame memories and answers user questions with location details and timestamps. |
| **Local Embedding Model** (`sentence-transformers` / HuggingFace) | `all-MiniLM-L6-v2` | Converts text descriptions and search queries into vector embeddings **100% locally**—zero embedding API cost and fast offline similarity search after the initial model download. |

> **Rate-Limit & Resume Design:** `describe_frames.py` includes a `4.5`-second pause (`time.sleep(4.5)`) between Gemini requests so it stays comfortably within a **15 RPM** (requests per minute) rate limit, and saves progress to `descriptions.json` after every frame. If a run is interrupted, restarting it automatically picks up where it left off without re-processing completed frames.

### Python Frameworks & Libraries

| Package | Purpose |
| :--- | :--- |
| **`opencv-python` (`cv2`)** | Opens video files, reads FPS/frame metadata, seeks to exact millisecond timestamps (`CAP_PROP_POS_MSEC`), and exports 1 FPS JPEGs. |
| **`google-genai`** | Official Google GenAI Python SDK (`from google import genai`) used to call Gemini models and list available models (`test.py`). |
| **`langchain` / `langchain-core`** | Provides the `Document` abstraction (`page_content` + `metadata`) connecting frame descriptions with file paths and timestamps. |
| **`langchain-huggingface`** | Wraps HuggingFace's `all-MiniLM-L6-v2` embeddings cleanly for LangChain retrieval. |
| **`chromadb` & `langchain-chroma`** | Stores embeddings and metadata persistently on disk inside `chroma_db/` for fast similarity search. |
| **`sentence-transformers`** | Underlying local transformer runtime used by `HuggingFaceEmbeddings`. |
| **`streamlit`** | Powers the interactive web UI (`app.py`) with video selection/upload, live pipeline progress, chat input, and side-by-side visual frame evidence. |
| **`Pillow` (`PIL`)** | Loads extracted `.jpg` images before passing them to the Gemini client. |
| **`python-dotenv`** | Loads your `GEMINI_API_KEY` from a local `.env` file. |
| **`pandas`** | Listed in `requirements.txt` for data handling support. |

---

## 📁 Project Structure

```text
visual-rag/
├── app.py                  # Streamlit web app (video upload, indexing pipeline, visual Q&A)
├── extract_frames.py       # Step 1: Samples video at 1 FPS into extracted_frames/
├── describe_frames.py      # Step 2: Uses Gemini to describe frames -> descriptions.json
├── build_vector_db.py      # Step 3: Embeds descriptions locally -> chroma_db/
├── query_pipeline.py       # Step 4: Retrieval + Gemini spatial reasoning (also has a CLI mode)
├── test.py                 # Utility script to list available Gemini models for your API key
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (holds GEMINI_API_KEY)
├── room.mp4                # Default sample room video
├── extracted_frames/       # Output folder for frame_MM_SS.jpg images
├── descriptions.json       # Cached frame-by-frame descriptions & timestamps
└── chroma_db/              # Persisted Chroma vector database
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3** (tested locally with **Python 3.13.1**)
- A **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/))

### 2. Activate (or Create) Your Virtual Environment

On **Windows (PowerShell)** — if the `env/` virtual environment is already present in the workspace, activate it directly (or create a fresh one first with `python -m venv env`):
```powershell
.\env\Scripts\Activate.ps1
```

On **macOS / Linux**:
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If `python-dotenv` or `pillow` aren't already pulled into your environment, install them with `pip install python-dotenv pillow`.)*

### 4. Configure Your API Key
Create a `.env` file in the root directory of the project and add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

*(Optional)* You can verify that your key works and check which Gemini models are enabled on your account by running:
```bash
python test.py
```

---

## 🚀 How to Run the Project

You can run the assistant either through the **Streamlit Web App** (recommended for the full visual experience) or directly from the **Terminal / CLI** step by step.

### Option A: Run the Streamlit Web App (Recommended)

Launch the interactive UI:
```bash
streamlit run app.py
```

Once it opens in your browser:
1. **Using the default pre-indexed video (`room.mp4`):**
   - If `chroma_db/` is already built, you'll see **Ready to Chat! 🟢** right away.
   - Preview `room.mp4` in the sidebar and ask questions in the chat box at the bottom—for example:
     - *"Where is the potted plant?"*
     - *"What is on the wooden coffee table?"*
     - *"Where is the black bookshelf?"*
   - The app displays both the AI's spatial explanation and the exact **Supporting Visual Evidence** frames with timestamps.
2. **Uploading and indexing your own video:**
   - In the sidebar under **Choose Video Source**, select **Upload Your Own Video**.
   - Upload an `.mp4` or `.mov` file (saved locally as `uploaded_video.mp4`).
   - Click **🚀 Process & Index Video**.
   - The app clears the previous session data, extracts frames at 1 FPS, describes them with Gemini, builds a fresh Chroma vector store, and reloads automatically when finished.

> **Heads-up on processing time:** Because `describe_frames.py` waits 4.5 seconds per frame to respect the 15 RPM API rate limit, indexing a **30-second video takes roughly 2 minutes**.

---

### Option B: Run the Pipeline via CLI (Step-by-Step)

If you want to inspect each stage individually or run queries in your terminal:

1. **Extract frames from `room.mp4` (1 frame per second):**
   ```bash
   python extract_frames.py
   ```
   Frames are saved as `extracted_frames/frame_00_00.jpg`, `frame_00_01.jpg`, etc.

2. **Generate visual descriptions using Gemini:**
   ```bash
   python describe_frames.py
   ```
   Descriptions and timestamps are written incrementally to `descriptions.json`. Safe to stop (`Ctrl+C`) and re-run to resume.

3. **Build the local Chroma vector database:**
   ```bash
   python build_vector_db.py
   ```
   Embeds all descriptions with `all-MiniLM-L6-v2` into `chroma_db/` and runs a quick sample validation query (`"Where is the red box?"`).

4. **Chat with your spatial memory in the terminal:**
   ```bash
   python query_pipeline.py
   ```
   Type any question at the `Ask:` prompt (and type `exit` when you're done). It prints both the Spatial Reasoner's answer and the matching frame file paths.

---

## 💡 Implementation Notes & Quirks Worth Knowing

- **Windows SQLite File Locks in ChromaDB:** In `app.py`, resetting the vector store before indexing a new video does *not* call `shutil.rmtree("chroma_db")` because Windows throws `WinError 32` while Python holds active SQLite handles. Instead, `clear_existing_data()` opens the Chroma collection and calls `vector_store.delete_collection()` followed by `st.cache_resource.clear()`.
- **First-Run Model Download:** The very first time `build_vector_db.py` or `query_pipeline.py` runs, `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` downloads the embedding model weights into your local HuggingFace cache. Subsequent runs load instantaneously from cache.
- **Customizing Retrieval Count (`k`):** Both `query_pipeline.py` and `app.py` default to retrieving `k=2` top matching frames per question. If your video is long or covers multiple rooms, bumping `k` in `query_visual_rag(query, k=...)` gives the reasoner broader context.
- **Keep `.env` Private:** Never commit `.env` (or your `GEMINI_API_KEY`) to version control.
