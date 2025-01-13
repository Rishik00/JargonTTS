import os
import pymupdf
import requests
from groq import Groq
from dotenv import load_dotenv
from typing import List, Tuple

from db import SQLiteDB

load_dotenv()

# Load the system prompt
with open('system_prompt.txt', 'r') as prompt_file:
    system_prompt = prompt_file.read()


def get_pdf_from_url(link: str, out_path: str) -> None:
    """Download a PDF file from a URL and save it locally."""
    response = requests.get(link)
    if response.status_code == 200:
        with open(out_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF saved to {out_path}")
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text("text")
    return text


def chunk_pdf(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split the text into chunks of specified size with overlap."""
    chunks, start = [], 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        last_period = text.rfind('.', start, end)
        last_newline = text.rfind('\n', start, end)
        chunk_end = max(last_period, last_newline, end)
        chunks.append(text[start:chunk_end])
        start = chunk_end + overlap

    return chunks


def parse_with_groq(text: str, system_prompt: str) -> List[Tuple[str, str]]:
    """Send text to Groq API to extract technical jargon."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract technical jargon from the following text:\n\n{text}"}
        ],
        model="mixtral-8x7b-32768",
        max_tokens=1000,
        temperature=0.2,
    )
    jargon_response = chat_completion.choices[0].message.content
    # print(jargon_response, type(jargon_response), jargon_response.split(','))

    jargon_pairs = list(set(jargon_response.strip().split(',')))
    print(f'checking here: {jargon_pairs}')

    return jargon_pairs

def main(db_path: str, url: str, output_file: str):
    """Main function to process the PDF and extract jargon."""
    db = SQLiteDB(path=db_path, db_name='jargon')

    if url:
        print(f"Downloading PDF from {url} to {output_file}...")
        get_pdf_from_url(link=url, out_path=output_file)

    print(f"Processing file: {output_file}")
    full_text = extract_pdf_text(output_file)
    text_chunks = chunk_pdf(full_text, chunk_size=2500, overlap=200)

    # Process each chunk for jargon
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}...")
        jargon_pairs = parse_with_groq(chunk, system_prompt)
        print(jargon_pairs)

        # terms = [pair[0].strip()  if isinstance(pair, tuple) else pair.strip() for pair in jargon_pairs]
        db.add(jargon_pairs)

    print("Final fetch from database:")
    print(db.fetch(limit=5))  # Fetch and display a few rows for verification
    db.close()


if __name__ == "__main__":
    # paper_link: str = "https://arxiv.org/pdf/2410.05258#page=13&zoom=100,144,445"
    paper_link: str = "https://arxiv.org/pdf/1402.4283"

    db_relative_path = r'..\audio_files\jargon_db.db'  # Relative path from target_dir
    db_path = os.path.abspath(db_relative_path)  # Convert to absolute path
    output_file = "difftrans.pdf"

    print("Database path:", db_path)
    main(db_path=db_path, url=paper_link, output_file=output_file)