from transformers import AutoModel
import librosa
from .AudioLM import AudioLM
from huggingface_hub import login

class DiVA(AudioLM):
    def __init__(self, hf_token, config = {}):
        """
        Initialize DiVA model
        Args:
            model_name: HuggingFace model name
            hf_token: HuggingFace access token
        """
        super().__init__()
        self.hf_token = hf_token
        self.max_new_tokens = config.get('max_new_tokens', 512)
        self.temperature = config.get('temperature', 0.5)
        self.top_p = config.get('top_p', 1.0)
        self.num_beams = config.get('num_beams', 1)
        self.load_model()
    
    def load_model(self):
        if self.hf_token:
            login(token=self.hf_token)
            
        self.model = AutoModel.from_pretrained(
            "WillHeld/DiVA-llama-3-v0-8b", 
            trust_remote_code=True
        )
        self.model.config.max_new_tokens = self.max_new_tokens
        self.model.config.temperature = self.temperature
        self.model.config.top_p = self.top_p
        self.model.config.num_beams = self.num_beams
    
    def process_audio(self, audio_path: str, prompt=None, addtional_system_prompt=None, **kwargs) -> str:
        if not audio_path and prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        
        if not audio_path and prompt:
            responses = self.model.generate([0], [((addtional_system_prompt + "\n") if addtional_system_prompt else "") +  prompt], max_new_tokens=self.max_new_tokens)
        elif audio_path and prompt:
            speech_data, _ = librosa.load(audio_path, sr=16_000)
            responses = self.model.generate([speech_data], [((addtional_system_prompt + "\n") if addtional_system_prompt else "") +  prompt], max_new_tokens=self.max_new_tokens)
        elif audio_path and not prompt:
            speech_data, _ = librosa.load(audio_path, sr=16_000)
            if addtional_system_prompt:
                responses = self.model.generate([speech_data], [addtional_system_prompt], max_new_tokens=self.max_new_tokens)
            else:
                responses = self.model.generate([speech_data], max_new_tokens=self.max_new_tokens)
        return responses[0]
