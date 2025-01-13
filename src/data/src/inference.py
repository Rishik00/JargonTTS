import gc
import torch
import time
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
from contextlib import contextmanager

@contextmanager
def manage_memory():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def load_model(model_name: str = "microsoft/speecht5_tts", device: str = "cuda"):
    print(f"Loading {model_name} on {device}...")
    model = pipeline("text-to-speech", model=model_name, device=device)
    print("Model loaded for inference!")
    return model

def load_speaker_embeddings(name: str = "Matthijs/cmu-arctic-xvectors", idx: int = 7000, device: str = "cuda"):
    print(f"Loading speaker embeddings from {name}...")
    speaker_embeddings_dataset = load_dataset(name, split="validation")
    speaker_embedding = torch.tensor(
        speaker_embeddings_dataset[idx]["xvector"], device=device
    ).unsqueeze(0)
    print("Speaker embeddings loaded successfully!")
    return speaker_embedding

def take_inference(num: int, model, text: str, speaker_embedding, output_dir: str = "."):
    output_path = f"{output_dir}/speech_{num}.wav"
    print(f"Generating speech for text: {text}")
    
    speech = model(
        text,
        forward_params = {"speaker_embeddings": speaker_embedding}
    )

    if speech:
        sf.write(output_path, speech["audio"], speech["sampling_rate"])
        print(f"Speech generated successfully! File saved at {output_path}")
        print(f"Audio length: {len(speech['audio']) / speech['sampling_rate']:.2f} seconds")
    else:
        print("Failed to generate speech.")

    print("Total memory used:")
    get_memory_usage()

if __name__ == "__main__":
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initializing the TTS pipeline...")
    model_name = "microsoft/speecht5_tts"
    model = load_model(model_name=model_name, device=device)

    print("Loading speaker embeddings...")
    speaker_embeds = load_speaker_embeddings(device=device)

    # Example texts for inference
    texts = [
        "neural network",
        "Hello world"
    ]
    
    print("Starting inference...")
    for i, text in enumerate(texts, start=1):
        with manage_memory():
            take_inference(num=i, model=model, text=text, speaker_embedding=speaker_embeds)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
