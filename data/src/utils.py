import os
import pymupdf
import argparse
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

    jargon_pairs = list(set(jargon_response.strip().split(',')))
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
        db.add(jargon_pairs)
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI Application")
    parser.add_argument("--path", type=str, required=True, help="Path to the model (e.g., 'microsoft/speecht5_tts')")
    parser.add_argument("--url", type=str, required=True, help="Path to the text file for TTS synthesis")
    parser.add_argument("--output-file", type=str, default="output.wav", help="Path to save the generated audio file")
    parser.add_argument("--db-path", type=str, default="../audio_files/jargon_db.db", help="Path to the speaker embedding database")
    parser.add_argument("--db-name", type=str, default="default", help="Name of the speaker embedding to use")
    args = parser.parse_args()

    # Validate paths
    db_path_absolute = os.path.abspath(args.db_path)
    print("Database path (absolute):", db_path_absolute)

    # Call the main function
    main(db_path=db_path_absolute, url=args.url, output_file=args.output_file)