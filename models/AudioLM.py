from abc import ABC, abstractmethod
import torch
from transformers import pipeline
import os
from models.TTS_Google import GoogleTTS
import soundfile as sf

class AudioLM(ABC):
    """
    Abstract base class for audio language models
    """
    def __init__(self, **model_kwargs):
        """
        Initialize audio language model
        
        Args:
            **model_kwargs: Model-specific keyword arguments
                e.g., hf_token for DiVA, max_length for Qwen
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.model_kwargs = model_kwargs
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def process_audio(self, audio_path: str, prompt=None, **process_kwargs) -> str:
        """
        Abstract method to process audio
        
        Args:
            audio_path (str): URL or path to audio file
            
        Returns:
            str: Processing result
        """
        pass

    def process_text(self, text: str, tts_model="google_api", prompt=None, cache_path=None, **process_kwargs) -> str:
        """
        Abstract method to process audio
        
        Args:
            text (str): text to process
            tts_model (str): text to speech model
                                e.g. facebook/mms-tts-eng
                                     facebook/mms-tts

            
        Returns:
            str: Processing result
        """
        if cache_path != None and os.path.exists(cache_path):
            result = self.process_audio(cache_path, prompt=prompt, **process_kwargs)
        else:
            if cache_path == None:
                os.makedirs("./temp", exist_ok=True)
                path = "./temp/output_audio.wav"
            else:
                path = cache_path
            if tts_model == "google_api":
                GoogleTTS().text_to_speech(text, path, process_kwargs.get("target_lang", "en-US"))
            else:
                pipe = pipeline("text-to-speech", model=tts_model, device=self.device)
                audio_data = pipe(text)
                sf.write(path, audio_data["audio"].squeeze(), audio_data["sampling_rate"])
            result = self.process_audio(path, prompt=prompt, **process_kwargs)
        return result
        