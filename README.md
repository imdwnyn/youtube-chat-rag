# YouTube RAG Chatbot

A conversational AI chatbot that lets you ask questions about any YouTube video using its transcript. Built with LangChain, OpenAI, and FAISS — demonstrating Retrieval-Augmented Generation (RAG) with multi-layered memory.

---

## What It Does

Paste any YouTube URL and the chatbot will:
- Fetch the video's transcript automatically
- Let you ask questions about the video in plain English
- Remember the full conversation across multiple turns
- Answer based on actual transcript content, not hallucination

---

## How It Works

This project is a practical implementation of a **RAG (Retrieval-Augmented Generation)** pipeline. Here's the flow:

```
YouTube URL
    ↓
Fetch Transcript (YouTubeTranscriptApi)
    ↓
Split into Chunks (RecursiveCharacterTextSplitter)
    ↓
Embed Chunks (OpenAI text-embedding-3-small)
    ↓
Store in FAISS Vector Database
    ↓
User asks a question
    ↓
Retrieve relevant chunks (MMR search)  +  Retrieve relevant memory
    ↓
Fill prompt with context + memory + chat history + summary
    ↓
GPT-4o-mini generates answer
    ↓
Update memory (chat history + vector memory + rolling summary)
```

---

## Memory System

One of the key features of this project is its **3-layer memory system** that keeps the conversation coherent across many turns:

| Layer | What it stores | How long it lasts |
|---|---|---|
| Chat history | Last 5 raw Q&A pairs | Sliding window, resets each session |
| Vector memory | All Q&A pairs, searchable by meaning | Full session, grows with conversation |
| Rolling summary | Compressed summary of the whole conversation | Full session, updated every turn |

All three are injected into the prompt on every question so the LLM always has full context.

---

## Project Structure

```
project/
│
├── main.py            # Full pipeline: transcript → RAG → chat loop
├── evaluate.py        # RAGAS evaluation script
├── ragas_scores.txt   # Saved evaluation scores
├── .env               # API keys and config values (never push this)
├── .gitignore         # Excludes .env, venv, cache files
├── requirements.txt   # All dependencies
└── README.md          # This file
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `youtube-transcript-api` | Fetches YouTube captions |
| `langchain` | RAG pipeline, prompt templates, chain building |
| `langchain-openai` | OpenAI LLM and embeddings wrapper |
| `langchain-community` | FAISS vector store integration |
| `faiss-cpu` | In-memory vector database for similarity search |
| `openai` | GPT-4o-mini for answer generation |
| `python-dotenv` | Loads API keys from .env file |
| `ragas` | RAG evaluation framework |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here

# LLM
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
DOC_RETRIEVER_K=4
MEMORY_RETRIEVER_K=2
```

### 5. Run the chatbot

```bash
python main.py
```

Paste a YouTube URL when prompted, then start asking questions. Type `exit` to quit.

---

## Example Usage

```
Enter YouTube URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ

YouTube RAG Assistant Ready!

Ask the question (type 'exit' to exit): What is this video about?

Answer:
This video is about...

Ask the question (type 'exit' to exit): Who is the main speaker?

Answer:
The main speaker is...

Ask the question (type 'exit' to exit): exit
```

---

## Configuration

All tunable values live in `.env` — no need to touch `main.py` to change behaviour:

| Variable | Default | What it controls |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Which OpenAI model to use |
| `LLM_TEMPERATURE` | `0.2` | Randomness: 0 = factual, 1 = creative |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Which embedding model to use |
| `CHUNK_SIZE` | `1000` | Max characters per transcript chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared between adjacent chunks |
| `DOC_RETRIEVER_K` | `4` | Transcript chunks retrieved per question |
| `MEMORY_RETRIEVER_K` | `2` | Memory entries retrieved per question |

---

## Key Concepts Demonstrated

**RAG (Retrieval-Augmented Generation)** — Instead of relying purely on the LLM's training data, relevant transcript chunks are retrieved and injected into the prompt, grounding the answer in the actual video content.

**MMR Search (Maximal Marginal Relevance)** — Retrieval is done using MMR instead of plain similarity search. MMR balances relevance and diversity, ensuring the retrieved chunks cover different angles rather than returning near-identical repeated content.

**Vector embeddings** — Text is converted into high-dimensional vectors using OpenAI's embedding model. Semantically similar text ends up close together in vector space, enabling meaning-based search rather than keyword matching.

**FAISS** — Facebook AI Similarity Search is an in-memory vector database that makes nearest-neighbor search fast even across thousands of chunks.

**Multi-layer memory** — Three complementary memory mechanisms work together: a sliding window of recent exchanges, a vector store for semantic retrieval of past conversations, and a rolling summary that compresses the full history into a concise context.

**LangChain LCEL** — The pipeline uses LangChain Expression Language (LCEL) with `RunnableParallel` and `RunnableLambda` to run transcript and memory retrieval simultaneously, then pipe the results through the prompt and LLM in a clean, readable chain.

---

## Evaluation

The pipeline was evaluated using RAGAS, an LLM-based evaluation framework that measures 4 metrics:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved chunks, or did the LLM hallucinate? |
| Answer Relevancy | Does the answer actually address the question asked? |
| Context Precision | Of the chunks retrieved, how many were actually relevant? |
| Context Recall | Of all relevant info in the transcript, how much was retrieved? |

### Results

Evaluated on **StatQuest: Recurrent Neural Networks** — a verbally dense lecture where every concept is explained out loud, making it ideal source material for transcript-based RAG.

| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.96 |
| Context Precision | 0.87 |
| Context Recall | 1.00 |

### What the scores tell us

The pipeline achieves near-perfect retrieval (Context Recall: 1.00) and high answer quality (Answer Relevancy: 0.96) on verbally dense content. Scores are strongest on text-heavy videos where the transcript carries the full meaning without relying on visuals.

### Running evaluation yourself

```bash
python evaluate.py
```

Scores are saved to `ragas_scores.txt` after each run.

---

## Future Improvements

These are planned enhancements that weren't included in the current version to keep the codebase beginner-friendly:

**Persist FAISS index to disk**
Currently the transcript is re-embedded every time the script runs. FAISS supports `save_local()` and `load_local()` which would let you cache the index using the video ID as the folder name, saving both time and OpenAI API costs on repeated runs of the same video.

**Auto-translate non-English transcripts**
The chatbot fetches transcripts in English with a fallback to any available language. However, if a non-English transcript is fetched, retrieval quality drops because English questions are compared against non-English vectors. The fix would be to auto-detect the transcript language and translate it to English before embedding using a library like `deep-translator`.

**Stream the LLM response**
Currently `.invoke()` waits for the full answer before printing. Switching to `.stream()` would print tokens as they arrive — exactly like ChatGPT — for a much better user experience on long answers.

**Inject video metadata**
Using the YouTube Data API (free), the video title and channel name could be fetched and injected into the system prompt, giving the LLM richer context and allowing it to answer questions like "what video is this?" or "who made this?".

**Validate transcript length**
After cleaning `[Music]` and `[Applause]` tags, the transcript could be nearly empty for videos that are mostly music or sound effects. A simple length check before embedding would catch this edge case and exit early with a helpful message.

**Handle API errors in the chat loop**
Wrapping `main_chain.invoke()` in a try/except would prevent a rate limit or network error from crashing the entire session. The user could simply retry their question without losing their conversation history.

**Build a web UI**
The current interface is a terminal. A simple Streamlit or Gradio frontend would make this significantly more shareable and presentable as a portfolio piece.

**Support playlist URLs**
Extend the URL parser to accept YouTube playlist URLs, fetch all video transcripts, and build a combined knowledge base across multiple videos.

---

## License

MIT

---

### Author

Dwinayan Deb