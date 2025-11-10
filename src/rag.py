from typing import List, Dict, Any
import requests
from src.config import load_config
from src.search import SemanticSearch
from src.utils.logger import get_logger


SYSTEM_PROMPT = (
    "Anda adalah asisten layanan pelanggan perusahaan. Jawab dengan sopan, ringkas, dan akurat. "
    "Gunakan konteks yang diberikan. Jika informasi tidak tersedia, katakan dengan jujur dan sarankan langkah lanjutan."
)


def build_context_chunks(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        lines.append(r["content"])
    return "\n".join(lines)


class RAGService:
    def __init__(self):
        self.cfg = load_config()
        self.search = SemanticSearch()
        self.logger = get_logger("RAGService")

    def answer(
        self,
        question: str,
        category: str | None = None,
        top_k: int | None = None,
        previous_answer: str | None = None,
        previous_question: str | None = None,
    ) -> Dict[str, Any]:
        retrieved = self.search.query(question, top_k=top_k, category=category)
        context = build_context_chunks(retrieved)

        if not self.cfg["GROQ_API_KEY"]:
            return {
                "answer": "Groq API belum dikonfigurasi. Isi GROQ_API_KEY di .env untuk mengaktifkan jawaban LLM.",
                "sources": retrieved,
            }

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Konteks:\n{context}"},
                # Sertakan riwayat percakapan agar jawaban konsisten
                *([
                    {"role": "user", "content": f"Pertanyaan sebelumnya: {previous_question}"}
                ] if previous_question else []),
                *([
                    {"role": "assistant", "content": f"Jawaban sebelumnya: {previous_answer}"}
                ] if previous_answer else []),
                {"role": "user", "content": f"Pertanyaan: {question}. Mohon lanjutkan secara konsisten dengan jawaban sebelumnya jika relevan."},
            ],
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer, "sources": retrieved}
        except Exception as e:
            self.logger.error(f"Groq API error: {e}")
            return {
                "answer": f"Terjadi error saat memanggil Groq API: {e}",
                "sources": retrieved,
            }


def answer_question(
    question: str,
    category: str | None = None,
    top_k: int | None = None,
    previous_answer: str | None = None,
    previous_question: str | None = None,
) -> Dict[str, Any]:
    service = RAGService()
    return service.answer(
        question,
        category=category,
        top_k=top_k,
        previous_answer=previous_answer,
        previous_question=previous_question,
    )