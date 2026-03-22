# ==========================================
# evaluate.py
# Runs RAGAS evaluation on the YouTube RAG chatbot
# using the StatQuest "Recurrent Neural Networks" video
#
# RAGAS measures 4 things:
#   Faithfulness      -- is the answer grounded in the retrieved chunks?
#   Answer Relevancy  -- does the answer actually address the question?
#   Context Precision -- of the chunks retrieved, how many were relevant?
#   Context Recall    -- of all relevant info, how much was retrieved?
# ==========================================


# ==========================================
# 1. TEST CASES
# ==========================================

# Manually written Q&A pairs that serve as ground truth for evaluation
# Key rule: questions must reflect what the video ACTUALLY covers
# Generic ML questions score poorly -- specific, video-grounded questions score well
test_cases = [
    {
        "question": "Why are recurrent neural networks (RNNs) useful for stock price prediction?",
        "ground_truth": "Because stock prices are sequential data that change over time, and RNNs can handle variable-length sequences, allowing them to use different amounts of past data to make predictions."
    },
    {
        "question": "What is a key limitation of traditional neural networks compared to RNNs?",
        "ground_truth": "Traditional neural networks require a fixed number of inputs, while RNNs can process sequences of varying lengths."
    },
    {
        "question": "What is the main structural feature that distinguishes RNNs from standard neural networks?",
        "ground_truth": "RNNs include feedback loops that allow information from previous inputs to influence future predictions."
    },
    {
        "question": "How does the feedback loop help in processing sequential data?",
        "ground_truth": "The feedback loop allows outputs from previous steps to be combined with current inputs, enabling both to influence predictions."
    },
    {
        "question": "What is meant by unrolling an RNN?",
        "ground_truth": "Unrolling converts the feedback loop into multiple copies of the network, one for each time step, making it easier to visualize and compute sequential processing."
    },
    {
        "question": "How are inputs processed in an unrolled RNN?",
        "ground_truth": "Inputs are fed sequentially from oldest to newest, and the final output gives the prediction for the next time step."
    },
    {
        "question": "What happens to weights and biases when an RNN is unrolled multiple times?",
        "ground_truth": "The weights and biases are shared across all time steps, meaning they remain the same regardless of how many times the network is unrolled."
    },
    {
        "question": "What is the exploding gradient problem in RNNs?",
        "ground_truth": "It occurs when weights greater than 1 cause gradients to grow exponentially during backpropagation, leading to very large updates and unstable training."
    },
    {
        "question": "What is the vanishing gradient problem?",
        "ground_truth": "It occurs when weights less than 1 cause gradients to shrink exponentially, becoming close to zero and preventing effective learning."
    },
    {
        "question": "Why are basic RNNs not commonly used in practice?",
        "ground_truth": "Because they are difficult to train due to vanishing and exploding gradient problems, especially when dealing with long sequences."
    }
]


# ==========================================
# 2. IMPORTS
# ==========================================

import os
import re
import numpy as np

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset


# ==========================================
# 3. CONFIG
# ==========================================

# Loads the same .env as main.py so evaluation uses identical settings
load_dotenv()

LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 200))
DOC_RETRIEVER_K = int(os.getenv("DOC_RETRIEVER_K", 4))

# Video used for evaluation -- StatQuest RNN video chosen because:
#   1. Verbally dense -- every concept is explained out loud, no reliance on visuals
#   2. Factual and specific -- clear Q&A-able content
#   3. Directly related to ML portfolio theme
VIDEO_URL = "https://www.youtube.com/watch?v=AsNTP8Kwu80"


# ==========================================
# 4. FETCH + PROCESS TRANSCRIPT
# ==========================================

print("Fetching transcript...")

# Extract video ID from URL
match = re.search(r"(?:v=|youtu.be/)([^&]+)", VIDEO_URL)
video_id = match.group(1)

# Fetch transcript with language fallback chain
api = YouTubeTranscriptApi()
transcript_list = api.fetch(video_id, languages=["en", "en-US", "en-GB", ""])

# Flatten all caption chunks into one string and clean noise tags
transcript = " ".join(chunk.text for chunk in transcript_list)
transcript = re.sub(r"\[.*?\]", "", transcript).strip()

# Split into overlapping chunks so context isn't lost at boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
docs = splitter.create_documents([transcript])

# Embed all chunks and store in FAISS for similarity search
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(docs, embeddings)


# ==========================================
# 5. BUILD RETRIEVER
# ==========================================

# MMR (Maximal Marginal Relevance) retriever
# Balances relevance AND diversity -- avoids returning 4 near-identical chunks
# k=DOC_RETRIEVER_K: how many chunks to retrieve per question
doc_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": DOC_RETRIEVER_K}
)


# ==========================================
# 6. BUILD CHAIN
# ==========================================

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# Simplified prompt -- no memory or summary needed for evaluation
# We want to isolate retrieval + generation quality without memory adding noise
prompt = PromptTemplate(
    template="""
    You are a helpful assistant answering questions about a YouTube video.

    Transcript Context:
    {context}

    Question:
    {question}

    Answer clearly based on the transcript.
    """,
    input_variables=["context", "question"]
)

def format_docs(docs):
    # Joins retrieved Document objects into one readable string for the prompt
    return "\n\n".join(doc.page_content for doc in docs)

# Pipeline: retrieve context -> fill prompt -> LLM -> plain string
chain = (
    RunnableParallel({
        "context" : RunnableLambda(lambda x: x["question"]) | doc_retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"])
    })
    | prompt | llm | StrOutputParser()
)


# ==========================================
# 7. RUN PIPELINE ON ALL TEST CASES
# ==========================================

print("Running pipeline on test cases...")

results = []

for i, case in enumerate(test_cases):
    print(f"  [{i+1}/{len(test_cases)}] {case['question']}")

    # Manually retrieve chunks for this question
    # We do this separately so we can pass contexts to RAGAS
    # (the chain uses them internally but doesn't expose them)
    retrieved_docs = doc_retriever.invoke(case["question"])
    contexts = [doc.page_content for doc in retrieved_docs]

    # Get the LLM's answer for this question
    answer = chain.invoke({"question": case["question"]})

    # Package into the format RAGAS expects:
    #   question     -- what was asked
    #   answer       -- what the pipeline answered
    #   contexts     -- the chunks retrieved (RAGAS checks if answer is grounded in these)
    #   ground_truth -- the ideal correct answer (RAGAS checks answer correctness against this)
    results.append({
        "question"    : case["question"],
        "answer"      : answer,
        "contexts"    : contexts,
        "ground_truth": case["ground_truth"]
    })

print(f"\nDone! Collected {len(results)} results.\n")


# ==========================================
# 8. RAGAS EVALUATION
# ==========================================

print("Running RAGAS evaluation...")

# Convert results list to HuggingFace Dataset format -- required by RAGAS internally
dataset = Dataset.from_list(results)

# Run evaluation -- RAGAS uses GPT under the hood to judge outputs
# This makes ~32 LLM calls and takes about 30-60 seconds
# Faithfulness     -- does the answer stay within what the chunks say?
# Answer Relevancy -- does the answer address what was actually asked?
# Context Precision -- of the retrieved chunks, how many were actually useful?
# Context Recall   -- of all relevant info in the transcript, how much was retrieved?
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)


# ==========================================
# 9. PRINT + SAVE RESULTS
# ==========================================

# scores returns lists -- take mean across all test cases for a single score per metric
faithfulness_score      = np.mean(scores['faithfulness'])
answer_relevancy_score  = np.mean(scores['answer_relevancy'])
context_precision_score = np.mean(scores['context_precision'])
context_recall_score    = np.mean(scores['context_recall'])

print("\n========== RAGAS SCORES ==========\n")
print(f"Faithfulness      : {faithfulness_score:.2f}   (is the answer grounded in the chunks?)")
print(f"Answer Relevancy  : {answer_relevancy_score:.2f}   (does it actually answer the question?)")
print(f"Context Precision : {context_precision_score:.2f}   (were retrieved chunks relevant?)")
print(f"Context Recall    : {context_recall_score:.2f}   (was all relevant info retrieved?)")
print("\n===================================\n")

# Save scores to a text file for reference in README and portfolio
with open("ragas_scores.txt", "w") as f:
    f.write("RAGAS Evaluation Scores\n")
    f.write("Video: Recurrent Neural Networks (RNNs) -- StatQuest\n")
    f.write("URL: https://www.youtube.com/watch?v=AsNTP8Kwu80\n\n")
    f.write(f"Faithfulness      : {faithfulness_score:.2f}\n")
    f.write(f"Answer Relevancy  : {answer_relevancy_score:.2f}\n")
    f.write(f"Context Precision : {context_precision_score:.2f}\n")
    f.write(f"Context Recall    : {context_recall_score:.2f}\n")

print("Scores saved to ragas_scores.txt")