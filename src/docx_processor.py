from typing import List
from docx import Document as DocxDocument

class DocxProcessor:
    def __init__(self):
        pass

    def extract_text_from_docx(self, docx_path: str) -> str:
        doc = DocxDocument(docx_path)
        parts: List[str] = []

        for p in doc.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                parts.append(txt)

        for table in doc.tables:
            for row in table.rows:
                cells = [(c.text or "").strip() for c in row.cells]
                row_text = " | ".join([c for c in cells if c])
                if row_text:
                    parts.append(row_text)

        return "\n".join(parts).strip()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size deve ser > 0")
        if overlap < 0:
            raise ValueError("overlap deve ser >= 0")
        if overlap >= chunk_size:
            overlap = chunk_size // 3

        words = text.split()
        chunks: List[str] = []
        i = 0
        step = max(1, chunk_size - overlap)

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            i += step

        return chunks

if __name__ == "__main__":
    processor = DocxProcessor()
    path = "training.docx"
    full_text = processor.extract_text_from_docx(path)
    chunks = processor.chunk_text(full_text, chunk_size=50, overlap=10)

    print("Texto (primeiras 400 chars):")
    print(full_text[:400], "...\n")
    print("Chunks:")
    for i, ch in enumerate(chunks, 1):
        print(f"Chunk {i}: {ch[:100]}...")