import os
import streamlit as st
import dspy
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

INTRO = (
    "**Hi! I'm the Nexarion Sports Complex assistant.** "
    "I can answer questions about our facilities, memberships, fees, schedules, and rules. "
    "What would you like to know?"
)


@st.cache_data
def load_pdf_chunks(path: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


class AnswerFromContext(dspy.Signature):
    """Answer the question using ONLY the context below.
If you don't know, say "I don't have enough information." """

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    response: str = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, collection):
        self.collection = collection
        self.hypothesize = dspy.ChainOfThought("question -> hypothetical_answer")
        self.respond = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question: str):
        hypothesis = self.hypothesize(question=question).hypothetical_answer
        results = self.collection.query(query_texts=[hypothesis], n_results=5)
        context = results["documents"][0]
        return self.respond(context=context, question=question)


@st.cache_resource
def _get_collection():
    chunks = load_pdf_chunks("data.pdf")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("pdf_docs", embedding_function=ef)
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return collection


@st.cache_resource
def _get_lm():
    lm = dspy.LM(model="groq/llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    dspy.configure(lm=lm)
    return lm


if "rag" not in st.session_state:
    with st.status("Getting your assistant ready...", expanded=True) as status:
        st.write("Reading and chunking PDF...")
        collection = _get_collection()

        st.write("Connecting to language model...")
        _get_lm()

        st.write("Building RAG pipeline...")
        st.session_state.rag = RAG(collection)
        status.update(label="Assistant ready!", state="complete", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": INTRO}]

rag = st.session_state.rag

st.title("Nexarion Sports Complex")

st.markdown(
    """
    <style>
    div[data-testid="stExpander"] {
        border: 1px solid rgba(49, 130, 206, 0.4);
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(49, 130, 206, 0.05), rgba(128, 90, 213, 0.05));
        margin-bottom: 1rem;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(49, 130, 206, 0.8);
        background: linear-gradient(135deg, rgba(49, 130, 206, 0.09), rgba(128, 90, 213, 0.09));
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 0.95rem;
        color: rgba(49, 130, 206, 0.95);
        letter-spacing: 0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("About this project"):
    st.markdown(
        """
        #### Source Document

        The knowledge base is a fictional handbook for **Nexarion Sports Complex**, a made-up
        multi-sport facility in Thornvale, Wyoming. It covers 12 facility zones, membership tiers
        and fees, operating hours, rules and policies, staff contacts, and parking.

        ---

        #### What this app does

        A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions grounded in
        that document — no hallucination of facts outside the source material.

        #### How it works

        1. **Ingestion** — The PDF is parsed with `pypdf` and split into overlapping 300-word
           chunks (50-word overlap) so table headers stay attached to their rows across chunk boundaries.
        2. **Indexing** — Each chunk is embedded with `all-MiniLM-L6-v2` (a 22M-parameter
           sentence-transformer model from HuggingFace) and stored in an in-memory
           **ChromaDB** vector database.
        3. **HyDE retrieval** — When a question arrives, a language model first generates
           a *hypothetical answer*. That answer is embedded and used as the search query instead
           of the raw question — this technique (Hypothetical Document Embeddings) significantly
           improves retrieval of sparse content like table rows.
        4. **Generation** — The top-5 retrieved chunks are passed as context to the LLM with
           a strict instruction: answer using only the provided context, or admit it doesn't know.

        The entire pipeline — HyDE, retrieval, and generation — is composed using **DSPy**
        (Stanford), a framework that replaces hand-written prompt strings with typed, declarative
        modules (`dspy.Signature`, `dspy.ChainOfThought`). This makes the pipeline modular,
        testable, and optimizable without touching raw prompts.

        #### Architecture
        """
    )
    st.image("architecture.svg")
    st.markdown(
        """
        #### Stack

        | Layer | Technology |
        |---|---|
        | LLM | Llama 3.1 8B via **Groq** API |
        | RAG framework | **DSPy** (Stanford) |
        | Vector store | **ChromaDB** (in-memory) |
        | Embeddings | **HuggingFace** `all-MiniLM-L6-v2` |
        | PDF parsing | `pypdf` |
        | UI | **Streamlit** |
        """
    )


st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "Can a 15-year-old use the spa without a parent present?",
    "What is the target water temperature of the Olympic Training Pool?",
    "What is the cancellation fee for Elite members who freeze their membership?",
    "Which membership tier is cheapest on a monthly basis for a full-time 22-year-old university student?",
]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

only_intro = len(st.session_state.messages) == 1
if only_intro:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

chat_input = st.chat_input("Ask a question...")
prompt = st.session_state.pop("pending_prompt", None) or chat_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag(question=prompt).response
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
