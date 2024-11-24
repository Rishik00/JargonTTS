import torch
from datasets import load_dataset
from typing import List

from model_utils import memory_management_fn, get_detailed_memory_usage
from inference_model import TTSModelForAudioInference
from dbs import SQLiteDBStore


def get_audio_samples(texts: List[str], batch_size: int = 1):

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            with memory_management_fn():
                batch_texts = texts[i:i + batch_size]
                
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                print("Memory at batch start:")
                get_detailed_memory_usage()
                
                # Load speaker embedding for this batch
                speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embedding = torch.tensor(
                    speaker_embeddings_dataset[7000]["xvector"], 
                    device=device
                ).unsqueeze(0)
                
                # Initialize model for this batch
                tts_model = TTSModelForAudioInference(
                    model_name="microsoft/speecht5_tts",
                    speaker_embedding=speaker_embedding,
                    device=device
                )
                
                # Process each text in batch
                for idx, text in enumerate(batch_texts):
                    global_idx = i + idx
                    print(f"\nProcessing text {global_idx + 1}/{len(texts)}")
                    tts_model.take_inference(text, global_idx)
                
                # Cleanup after batch
                del tts_model
                torch.cuda.empty_cache()
                
                print("\nMemory after batch:")
                get_detailed_memory_usage()
            
        print("\nFinished processing all texts")
        
    except Exception as e:
        print(f"Error in get_audio_samples: {str(e)}")

def main():
    pass