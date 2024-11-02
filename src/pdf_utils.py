import requests

def get_pdf(url: str, dump_path: str) -> None:

    pdf_response = requests.get(url)
    with open(dump_path, 'wb') as f:
        f.write(pdf_response.content)

def chunk_pdf(text: str, chunk_size: int, stop_ch = '\n', window_size: int = 1, overlap: int = 200):
    ## Chunk through the text for upto k words
    ## Upto window_size words, take the phrases/words 
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end > len(text):
            end = len(text)
            
        else:
            # Find the last period or newline within the overlap
            last_period = text.rfind('.', end - overlap, end)
            last_newline = text.rfind('\n', end - overlap, end)

            end = max(last_period, last_newline)
            if end == -1 or end < start:
                end = start + chunk_size

        chunks.append(text[start:end])
        start = end

    return chunks

def parse_and_store_keywords(text: str):
    ## Remove duplicates
    ## Convert into bytes streams
    ## Send to the db
    pass