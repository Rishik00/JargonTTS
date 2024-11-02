from groq import Groq
from dotenv import load_dotenv
from typing import List, Tuple

from JargonTTS.data.src.pdf_utils import get_pdf, chunk_pdf, extract_pdf_text
from JargonTTS.data.src.dbs import SQLiteDBStore


GROQ_API_KEY = load_dotenv()

system_prompt = """
You are an advanced AI designed to parse research papers and extract all technical jargon. The text provided will 
contain technical language, including machine learning terminology, network-related terms, and other domain-specific 
jargon. Your task is to identify and list all of the technical jargon present in the text. Please provide the output 
as a comma-separated list of terms. Do not include common words or phrases that are not specific technical jargon.
Do not include author names and citations.

And when it comes to numbers I want you to represent them in their english format. Few examples might be:
10 x 10 should be ten by 10
12 should be twelve

And do not include decimal numbers and mathematical notations written in latex.
"""


# Parse jargon with Groq
def parse_jargon_with_groq(text: str, system_prompt: str, groq_api_key: str) -> List[Tuple[str, str]]:
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract technical jargon from the following text:\n\n{text}"}
        ],
        model="mixtral-8x7b-32768",
        max_tokens=1000,
        temperature=0.2,
    )
    jargon_response = chat_completion.choices[0].message.content.strip().split('\n')
    jargon_pairs = [tuple(line.split(':')) for line in jargon_response if ':' in line]
    return jargon_pairs

# Main function
def main(db_path: str, groq_api_key: str, system_prompt: str, urls: List[str] = [], pdf_paths: List[str] = []):
    db = SQLiteDBStore(db_path)

    # Download PDFs from URLs
    for url in urls:
        filename = url.split('/')[-1]
        get_pdf(url, filename)
        pdf_paths.append(filename)

    # Process each PDF
    for path in pdf_paths:
        print(f"Processing file: {path}")
        full_text = extract_pdf_text(path)
        text_chunks = chunk_pdf(full_text, chunk_size=1500)

        # Process each chunk for jargon
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}...")
            jargon_pairs = parse_jargon_with_groq(chunk, system_prompt, groq_api_key)
            db.add_to_db(jargon_pairs)

    db.close()

main(
    db_path='jargon_db.db',
    groq_api_key=GROQ_API_KEY,
    system_prompt=system_prompt,
    pdf_paths=['/content/Matrix Calculus for Deep Learning (13.pdf']
)