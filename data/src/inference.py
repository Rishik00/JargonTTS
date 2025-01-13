import gc
import torch
import argparse
import time
import os
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

def load_model(model_name: str, device: str):
    print(f"Loading {model_name} on {device}...")
    model = pipeline("text-to-speech", model=model_name, device=0 if device == "cuda" else -1)
    print("Model loaded for inference!")
    return model

def load_speaker_embeddings(name: str, idx: int, device: str):
    print(f"Loading speaker embeddings from {name}...")
    speaker_embeddings_dataset = load_dataset(name, split="validation")
    speaker_embedding = torch.tensor(
        speaker_embeddings_dataset[idx]["xvector"], device=device
    ).unsqueeze(0)
    print("Speaker embeddings loaded successfully!")
    return speaker_embedding

def take_inference(num: int, model, text: str, speaker_embedding, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"speech_{num}.wav")
    print(f"Generating speech for text: {text}")
    
    speech = model(
        text,
        forward_params={"speaker_embeddings": speaker_embedding}
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
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI Application")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'microsoft/speecht5_tts')")
    parser.add_argument("--text-file", type=str, required=True, help="Path to the text file for TTS synthesis")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the generated audio file")
    parser.add_argument("--speaker-index", type=int, default=0, help="Index of the speaker embedding to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on ('cuda' or 'cpu')")
    args = parser.parse_args()

    start_time = time.time()

    # Optional: Add a check if the file exists
    if not os.path.isfile(args.text_file):
        print(f"Error: The file {args.text_file} does not exist.")
        exit(1)

    print("Initializing the TTS pipeline...")
    model = load_model(model_name=args.model, device=args.device)

    print("Loading speaker embeddings...")
    speaker_embeds = load_speaker_embeddings(name="Matthijs/cmu-arctic-xvectors", idx=args.speaker_index, device=args.device)

    print("Starting inference...")
    with manage_memory():
        with open(args.text_file, 'r') as input_file:
            input_text = input_file.readlines()

            # Process each line, stripping any extra spaces or newlines
            input_text = [line.strip() for line in input_text if line.strip()]  # Ignore empty lines

            for idx, text in enumerate(input_text):
                print(f"Processing line {idx + 1}: {text}")
                # Uncomment the following line to run TTS once model and embeddings are loaded
                take_inference(num=idx + 1, model=model, text=text, speaker_embedding=speaker_embeds, output_dir=args.output_dir)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
