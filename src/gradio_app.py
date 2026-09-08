from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from textwrap import dedent

import chromadb
import gradio as gr
import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "resume_rag"

GENERATION_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY")

index_state = {
    "ready": False,
    "running": False,
    "error": None,
    "status": "Waiting to index resumes...",
    "chunk_count": 0,
    "pdf_count": 0,
}
index_lock = threading.Lock()
collection = None
genai_client = genai.Client(api_key=API_KEY) if API_KEY else None


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdfs() -> list[dict]:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    documents = []
    for pdf_path in pdf_paths:
        with pymupdf.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))
                if not text:
                    continue
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_index,
                            "parser": "pymupdf",
                        },
                    }
                )

    if not documents:
        raise ValueError("No text was extracted. If a PDF is scanned, add OCR before indexing.")
    return documents


SECTION_HEADINGS = {
    "experience",
    "education",
    "technical skills",
    "projects and research publications",
    "projects",
    "research publications",
    "extracurricular activities",
}


def sectionize_text(text: str) -> list[dict]:
    sections = []
    current_title = "Profile"
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        normalized = stripped.lower().rstrip(":")
        if normalized in SECTION_HEADINGS:
            if current_lines:
                sections.append({"section": current_title, "text": "\n".join(current_lines).strip()})
            current_title = stripped
            current_lines = [stripped]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"section": current_title, "text": "\n".join(current_lines).strip()})
    return [section for section in sections if section["text"]]


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = paragraph

        while len(current) > chunk_size:
            split_at = current.rfind("\n", 0, chunk_size)
            if split_at < chunk_size * 0.5:
                split_at = current.rfind(" ", 0, chunk_size)
            if split_at < chunk_size * 0.5:
                split_at = chunk_size

            chunks.append(current[:split_at].strip())
            current = current[max(split_at - overlap, 0) :].strip()

    if current:
        chunks.append(current)

    with_offsets = []
    search_from = 0
    for chunk in chunks:
        start = text.find(chunk[:80], search_from)
        if start == -1:
            start = search_from
        end = start + len(chunk)
        with_offsets.append({"text": chunk, "char_start": start, "char_end": end})
        search_from = max(end - overlap, 0)
    return with_offsets


def create_chunks(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        for section in sectionize_text(doc["text"]):
            for local_index, chunk in enumerate(split_text(section["text"])):
                metadata = dict(doc["metadata"])
                metadata.update(
                    {
                        "section": section["section"],
                        "chunk_index": len(chunks),
                        "section_chunk_index": local_index,
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    }
                )
                chunk_id = hashlib.sha1(
                    f"{metadata['source']}:{metadata['page']}:{section['section']}:{local_index}:{chunk['text']}".encode()
                ).hexdigest()
                chunks.append({"id": chunk_id, "text": chunk["text"], "metadata": metadata})
    return chunks


def embed_texts(texts: list[str], task_type: str, batch_size: int = 16) -> list[list[float]]:
    if genai_client is None:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to .env and restart the app.")

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        embeddings.extend([embedding.values for embedding in result.embeddings])
    return embeddings


def build_index() -> None:
    global collection

    with index_lock:
        index_state.update(
            {
                "ready": False,
                "running": True,
                "error": None,
                "status": "Parsing PDFs from data/...",
                "chunk_count": 0,
                "pdf_count": len(list(DATA_DIR.glob("*.pdf"))),
            }
        )

    try:
        documents = parse_pdfs()
        chunks = create_chunks(documents)
        chunk_texts = [chunk["text"] for chunk in chunks]

        with index_lock:
            index_state["status"] = f"Embedding {len(chunks)} resume chunk(s)..."

        chunk_embeddings = embed_texts(chunk_texts, task_type="RETRIEVAL_DOCUMENT")

        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        new_collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        new_collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=[chunk["metadata"] for chunk in chunks],
        )

        with index_lock:
            collection = new_collection
            index_state.update(
                {
                    "ready": True,
                    "running": False,
                    "error": None,
                    "status": "Ready. Ask me about the resume data.",
                    "chunk_count": len(chunks),
                }
            )
    except Exception as exc:
        with index_lock:
            index_state.update(
                {
                    "ready": False,
                    "running": False,
                    "error": str(exc),
                    "status": f"Indexing failed: {exc}",
                }
            )


def start_background_indexing() -> None:
    with index_lock:
        if index_state["running"]:
            return

    thread = threading.Thread(target=build_index, daemon=True)
    thread.start()


def retrieve(query: str, k: int = 4) -> list[dict]:
    with index_lock:
        active_collection = collection

    if active_collection is None:
        raise RuntimeError("The resume index is not ready yet.")

    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY", batch_size=1)[0]
    results = active_collection.query(query_embeddings=[query_embedding], n_results=k)

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({"text": text, "metadata": metadata, "distance": distance})
    return matches


def format_context(matches: list[dict]) -> str:
    blocks = []
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        blocks.append(
            f"[Context {index} | source={metadata['source']} | page={metadata['page']} | section={metadata.get('section', 'Unknown')}]\n"
            f"{match['text']}"
        )
    return "\n\n".join(blocks)


def response_finish_reason(response) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:
        return ""


def looks_incomplete(answer: str, finish_reason: str = "") -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    if "MAX_TOKENS" in finish_reason.upper():
        return True
    if stripped.endswith((",", ":", ";", "-", "and", "or")):
        return True
    return False


def answer_question(question: str) -> str:
    matches = retrieve(question)
    context = format_context(matches)

    prompt = dedent(
        f"""
        Answer the user's question using only the resume context below.

        Rules:
        - If the answer is not supported by the context, say: "I don't know based on the data."
        - Be specific and concise. If the user asks for a short answer, answer in 1-3 complete sentences.
        - If the exact project name in the question is not present, answer using the closest matching project only if the context clearly supports it.
        - Do not start with meta commentary like "Based on the context" unless needed for clarity.
        - Mention the source filename and page number only when useful.
        - Do not invent dates, employers, projects, skills, education, or experience.
        - Always finish the answer; do not end with an unfinished list or colon.

        Resume context:
        {context}

        Question:
        {question}
        """
    ).strip()

    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=4096),
    )
    answer = response.text or ""
    finish_reason = response_finish_reason(response)

    if looks_incomplete(answer, finish_reason):
        completion_prompt = dedent(
            f"""
            Complete this answer using only the same resume context. Return one polished final answer,
            not an explanation of what went wrong.

            Resume context:
            {context}

            Question:
            {question}

            Incomplete answer:
            {answer}
            """
        ).strip()
        response = genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=completion_prompt,
            config=types.GenerateContentConfig(max_output_tokens=4096),
        )
        answer = response.text or answer

    return answer.strip() or "I could not generate an answer."


def status_text() -> str:
    with index_lock:
        status = index_state["status"]
        pdf_count = index_state["pdf_count"]
        chunk_count = index_state["chunk_count"]
        error = index_state["error"]

    details = f"{status}\nPDFs found: {pdf_count} | Chunks indexed: {chunk_count}"
    if error:
        details += f"\nError: {error}"
    return details


def chat(user_message: str, history: list[dict] | None) -> tuple[list[dict], str]:
    history = history or []
    user_message = user_message.strip()
    if not user_message:
        return history, ""

    history = history + [{"role": "user", "content": user_message}]

    with index_lock:
        ready = index_state["ready"]
        running = index_state["running"]
        error = index_state["error"]

    if error:
        reply = f"The resume index failed to build: {error}"
    elif not ready:
        reply = "I am still indexing the resume PDFs. Try again in a moment." if running else "The resume index is not ready yet."
    else:
        try:
            reply = answer_question(user_message)
        except Exception as exc:
            reply = f"I hit an error while answering: {exc}"

    return history + [{"role": "assistant", "content": reply}], ""


def rebuild_index() -> str:
    start_background_indexing()
    return status_text()


with gr.Blocks(title="Resume RAG Assistant") as demo:
    gr.Markdown("# Resume RAG Assistant")
    timer = gr.Timer(2)
    status_box = gr.Textbox(label="Index status", value=status_text, interactive=False, lines=3)

    chatbot = gr.Chatbot(
        label="Chat",
        value=[
            {
                "role": "assistant",
                "content": "Hi, I am ready to help answer questions from the resume data. I will only use the PDFs indexed from the data folder.",
            }
        ],
        height=520,
    )
    message = gr.Textbox(
        label="Ask a question",
        placeholder="Example: What are my strongest technical skills?",
        lines=2,
    )

    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        rebuild_button = gr.Button("Rebuild index")
        refresh_button = gr.Button("Refresh status")

    message.submit(chat, inputs=[message, chatbot], outputs=[chatbot, message])
    send_button.click(chat, inputs=[message, chatbot], outputs=[chatbot, message])
    rebuild_button.click(rebuild_index, outputs=status_box)
    refresh_button.click(status_text, outputs=status_box)
    timer.tick(status_text, outputs=status_box)


def launch_app() -> None:
    start_background_indexing()
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "8000"))
    demo.launch(server_name="0.0.0.0", server_port=server_port, ssr_mode=False)


if __name__ == "__main__":
    launch_app()
