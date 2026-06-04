import os
import dspy
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"]
)
dspy.configure(lm=lm)


def load_pdf_chunks(path: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("pdf_docs", embedding_function=ef)

chunks = load_pdf_chunks("data.pdf")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)


def retrieve(query: str, k: int = 5) -> list[str]:
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


class AnswerFromContext(dspy.Signature):
    """Answer the question using ONLY the context below.
If you don't know, say "I don't have enough information." """

    context: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    response: str = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self):
        # HyDE: generate a hypothetical answer whose embedding is closer to the
        # target table row than the raw question embedding
        self.hypothesize = dspy.ChainOfThought("question -> hypothetical_answer")
        self.respond = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question: str):
        hypothesis = self.hypothesize(question=question).hypothetical_answer
        context = retrieve(hypothesis)
        return self.respond(context=context, question=question)


rag = RAG()

prediction = rag(question="What is the target water temperature of the Olympic Training Pool?")
print(prediction.response)
print(lm.inspect_history(1))
