import re
from PyPDF2 import PdfReader

def load_pdf(file_path: str) -> str:
    """Read PDF and return cleaned text."""
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks
