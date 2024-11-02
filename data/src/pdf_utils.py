import requests
import pymupdf
from typing import List

# Fetch PDF
def get_pdf(url: str, dump_path: str) -> None:
    pdf_response = requests.get(url)
    with open(dump_path, 'wb') as f:
        f.write(pdf_response.content)

# Chunk text into manageable segments
def chunk_pdf(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    chunks, start = [], 0
    while start < len(text):

        end = min(start + chunk_size, len(text))
        last_period = text.rfind('.', start, end)
        last_newline = text.rfind('\n', start, end)
        end = max(last_period, last_newline, end)
        
        chunks.append(text[start:end])
        start = end + overlap

    return chunks

# Extract text from PDF using pymupdf
def extract_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text
