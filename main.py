# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os          # To read environment variables (like API keys and config values)
import re          # For regex: extracting video ID from URL and cleaning transcript text

from dotenv import load_dotenv                              # Reads the .env file and loads values into os.environ
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled  # Fetches YouTube captions; TranscriptsDisabled handles videos with no captions
from langchain_text_splitters import RecursiveCharacterTextSplitter            # Splits long transcript into smaller overlapping chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                      # OpenAIEmbeddings: text to vectors | ChatOpenAI: GPT wrapper
from langchain_community.vectorstores import FAISS                             # In-memory vector database for fast similarity search
from langchain_core.prompts import PromptTemplate                              # Defines reusable prompt structures with {placeholders}
from langchain_core.runnables import RunnableParallel, RunnableLambda          # RunnableParallel: runs branches simultaneously | RunnableLambda: wraps a plain Python function into a chain step
from langchain_core.output_parsers import StrOutputParser                      # Strips the LLM response object down to a plain string


# ==========================================
# 2. LOAD ENV VARIABLES
# ==========================================

# Reads the .env file and injects all key=value pairs into the environment
# Must be called before any os.getenv() calls below
load_dotenv()


# ==========================================
# 3. CONFIG (all tunable values live here)
# ==========================================

# All hardcoded values now live in .env — change them there without touching logic
# The second argument to os.getenv() is the fallback default if the key is missing from .env

LLM_MODEL          = os.getenv("LLM_MODEL", "gpt-4o-mini")                  # Which GPT model to use
LLM_TEMPERATURE    = float(os.getenv("LLM_TEMPERATURE", 0.2))                # 0 = deterministic, 1 = creative
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # OpenAI embedding model
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 1000))                      # Max characters per transcript chunk
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 200))                    # Characters repeated between chunks to preserve context
DOC_RETRIEVER_K    = int(os.getenv("DOC_RETRIEVER_K", 4))                    # How many transcript chunks to retrieve per question
MEMORY_RETRIEVER_K = int(os.getenv("MEMORY_RETRIEVER_K", 2))                 # How many past memory entries to retrieve per question


# ==========================================
# 4. GET YOUTUBE TRANSCRIPT
# ==========================================

url = input("Enter YouTube URL: ")  # Accepts both youtube.com/watch?v=... and youtu.be/... formats

# Regex to extract the 11-character video ID from the URL
# (?:v=|youtu.be/) matches either format, ([^&]+) captures everything after it until & or end
match = re.search(r"(?:v=|youtu.be/)([^&]+)", url)

if not match:
   print("Invalid URL")
   exit()

video_id = match.group(1)  # The captured video ID e.g. "dQw4w9WgXcQ"

try:
   api = YouTubeTranscriptApi()

   # Fetch the English transcript — returns a list of caption chunks
   # Each chunk has .text (the words), .start (timestamp), .duration
   transcript_list = api.fetch(video_id, languages=["en"])

   # Flatten all chunks into one long string
   transcript = " ".join(chunk.text for chunk in transcript_list)

   # Remove noise like [Music], [Applause], [Laughter] from auto-generated captions
   transcript = re.sub(r"\[.*?\]", "", transcript)

except TranscriptsDisabled:
   # Creator has turned off captions for this video
   print("Captions disabled")
   exit()

except Exception as e:
   # Catches network errors, invalid video ID, or any other unexpected issue
   print("Error:", e)
   exit()


# ==========================================
# 5. TEXT SPLITTING
# ==========================================

# RecursiveCharacterTextSplitter splits text by natural boundaries:
# paragraph -> sentence -> word -> character (in that order of preference)
# chunk_size: each chunk is at most N characters
# chunk_overlap: N characters are shared between consecutive chunks
#   -> overlap ensures context is not lost at the boundary between chunks
splitter = RecursiveCharacterTextSplitter(
   chunk_size=CHUNK_SIZE,
   chunk_overlap=CHUNK_OVERLAP
)

# Wraps each chunk in a LangChain Document object (has .page_content and .metadata)
docs = splitter.create_documents([transcript])


# ==========================================
# 6. VECTOR STORES
# ==========================================

# text-embedding-3-small: OpenAI's lightweight embedding model
# Converts text into high-dimensional vectors that capture semantic meaning
# Similar meaning = vectors that are close together in vector space
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Build a FAISS index from all transcript chunks
# FAISS stores the vectors in memory and enables fast nearest-neighbor search
vector_store = FAISS.from_documents(docs, embeddings)

# A separate FAISS store for conversation memory (past Q&A exchanges)
# Seeded with a placeholder string so the store is never empty
# (FAISS requires at least one vector to initialize)
memory_store = FAISS.from_texts([" "], embeddings)


# ==========================================
# 7. RETRIEVERS
# ==========================================

# Converts the FAISS index into a LangChain retriever object
# search_type="mmr": MMR (Maximal Marginal Relevance) balances relevance AND diversity
#   -> avoids returning 4 near-identical chunks when the topic repeats in the transcript
#   -> each retrieved chunk is relevant to the question but different from the others
# k=DOC_RETRIEVER_K: returns the top K most relevant transcript chunks
doc_retriever = vector_store.as_retriever(
   search_type="mmr",
   search_kwargs={"k": DOC_RETRIEVER_K}
   )

# Same MMR pattern for memory -- ensures retrieved past exchanges cover different angles
# rather than returning the same conversation repeated across similar chunks
memory_retriever = memory_store.as_retriever(
   search_type="mmr",
   search_kwargs={"k": MEMORY_RETRIEVER_K}
   )


# ==========================================
# 8. HELPERS
# ==========================================

def format_docs(docs):
   # Takes a list of LangChain Document objects
   # Joins their text content with double newlines into one readable string
   # This formatted string gets injected into the {context} slot in the prompt
   return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history):
   # Converts the list of (question, answer) tuples into a readable conversation string
   # Example output:
   #   User: What is this video about?
   #   Assistant: This video is about...
   #   User: Who is the speaker?
   #   Assistant: The speaker is...
   return "\n".join(
      [f"User: {q}\nAssistant: {a}" for q, a in history]
      )


# ==========================================
# 9. LLM
# ==========================================

# ChatOpenAI wraps OpenAI's chat models (GPT-3.5, GPT-4, GPT-4o, etc.)
# model: which model to use -- gpt-4o-mini is fast and cost-efficient
# temperature: controls randomness -- 0.2 gives mostly factual, consistent answers
llm = ChatOpenAI(
   model=LLM_MODEL,
   temperature=LLM_TEMPERATURE
   )


# ==========================================
# 10. PROMPT
# ==========================================

# The master prompt template -- all {placeholders} are filled in dynamically at runtime:
#   {summary}      -> rolling compressed summary of the full conversation so far
#   {chat_history} -> the last 5 raw Q&A pairs (short-term memory)
#   {memory}       -> semantically relevant past exchanges retrieved from the vector memory store
#   {context}      -> the most relevant transcript chunks for the current question
#   {question}     -> the user's actual question
prompt = PromptTemplate(
   template="""
   You are a helpful assistant answering questions about a YouTube video.

   Conversation Summary:
   {summary}

   Recent Chat History:
   {chat_history}

   Relevant Past Memory:
   {memory}

   Transcript Context:
   {context}

   Question:
   {question}

   Answer clearly based on transcript and memory.
   """,
   input_variables=["summary", "chat_history", "memory", "context", "question"]
   )

# Converts the LLM's AIMessage response object into a plain Python string
parser = StrOutputParser()


# ==========================================
# 11. PARALLEL CHAIN
# ==========================================

# RunnableParallel runs all branches at the same time, then merges outputs into one dict
# This is efficient -- transcript retrieval and memory retrieval happen simultaneously

parallel_chain = RunnableParallel({

   # Branch 1: pull the question -> search transcript chunks -> format into string -> fills {context}
   "context": RunnableLambda(lambda x: x["question"]) | doc_retriever | RunnableLambda(format_docs),

   # Branch 2: pull the question -> search memory store -> format into string -> fills {memory}
   "memory": RunnableLambda(lambda x: x["question"]) | memory_retriever | RunnableLambda(format_docs),

   # Pass-throughs: these just forward values from the input dict to the next step unchanged
   "question": RunnableLambda(lambda x: x["question"]),
   "summary": RunnableLambda(lambda x: x["summary"]),
   "chat_history": RunnableLambda(lambda x: x["chat_history"])
   })

# Full pipeline: parallel retrieval -> fill prompt placeholders -> LLM -> parse to plain string
main_chain = parallel_chain | prompt | llm | parser


# ==========================================
# 12. SUMMARY MEMORY CHAIN
# ==========================================

# A separate prompt specifically for updating the rolling summary after each turn
# Keeps a compressed "big picture" of the conversation even as chat_history scrolls away
summary_prompt = PromptTemplate(
   template="""
   Update the conversation summary.

   Current summary:
   {summary}

   New interaction:
   User: {question}
   Assistant: {answer}

   Return concise updated summary.
   """,
   input_variables=["summary", "question", "answer"]
   )

# Lightweight chain: just the prompt -> LLM -> plain string (no retrieval needed here)
summary_chain = summary_prompt | llm | StrOutputParser()


# ==========================================
# 13. MEMORY VARIABLES
# ==========================================

summary_memory = ""   # Starts empty; grows into a rolling summary of the whole conversation
chat_history   = []   # List of (question, answer) tuples; capped at the last 5 turns


# ==========================================
# 14. CHAT LOOP
# ==========================================

print("\nYouTube RAG Assistant Ready!")

while True:

   question = input("\n\nAsk the question (type 'exit' to exit): ")

   if question.lower() == "exit":
      break  # End the session cleanly

   # Format the last 5 Q&A pairs into a readable string for the prompt
   formatted_history = format_chat_history(chat_history)

   # -------- MAIN RAG --------
   # Invoke the full chain: retrieve context + memory -> fill prompt -> LLM -> answer string
   answer = main_chain.invoke({
      "question": question,
      "summary": summary_memory,
      "chat_history": formatted_history
      })

   print("\n\nAnswer:\n\n", answer)

   # -------- UPDATE CHAT HISTORY --------
   # Append the new Q&A pair and keep only the most recent 5 turns
   # Acts as a sliding window of short-term memory
   chat_history.append((question, answer))
   chat_history = chat_history[-5:]

   # -------- UPDATE SUMMARY --------
   # Feed the latest Q&A into the summary chain to keep the rolling summary up to date
   # Compresses the full conversation history into a concise "story so far"
   summary_memory = summary_chain.invoke({
      "summary": summary_memory,
      "question": question,
      "answer": answer
      })

   # -------- STORE IN VECTOR MEMORY --------
   # Add the raw Q&A exchange as a new entry in the FAISS memory store
   # Future questions can semantically retrieve relevant past exchanges from here
   memory_store.add_texts([
      f"User: {question}\nAssistant: {answer}"
   ])