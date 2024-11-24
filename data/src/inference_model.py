import torch
from contextlib import contextmanager

from transformers import pipeline
from datasets import load_dataset
import time
import soundfile as sf
from typing import List, Union


from model_utils import memory_management_fn, get_detailed_memory_usage, get_audio_samples


class TTSModelForAudioInference:
    
    def __init__(
            self, 
            model_name: str, 
            speaker_embedding: torch.Tensor, 
            device: str
    ):

        self.model_name = model_name
        self.speaker_embedding = speaker_embedding
        self.device = device
        self.model = None
        
        with memory_management_fn():
            self.load_model()

    def load_model(self):
        """Load the TTS model"""
        try:
            print(f"Loading {self.model_name} to {self.device}")
            self.model = pipeline("text-to-speech", self.model_name, device=self.device)
            print("Loaded model for inference!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def take_inference(
            self, 
            text: str, 
            num: int
    ) -> dict:
    
        try:
            with memory_management_fn(), torch.inference_mode():
                start_time = time.time()
                
                output_path = f"speech_{num}.wav"
                speech = self.model(
                    text,
                    forward_params={"speaker_embeddings": self.speaker_embedding}
                )

                if speech:
                    # Save the audio file
                    sf.write(output_path, speech["audio"], speech["sampling_rate"])
                    
                    duration = time.time() - start_time
                    audio_length = len(speech["audio"]) / speech["sampling_rate"]
                    
                    print(f"Generated speech for: {text}")
                    print(f"Output saved to: {output_path}")
                    print(f"Audio length: {audio_length:.2f} seconds")
                    print(f"Generation time: {duration:.2f} seconds")
                    print(f"Real-time factor: {duration/audio_length:.2f}x")
                    
                    print('\nCurrent memory usage:')
                    
                    return speech
                    
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None