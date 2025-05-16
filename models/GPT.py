import base64
from openai import AzureOpenAI
from .AudioLM import AudioLM
import numpy as np
import soundfile as sf
import os

class GPT(AudioLM):
    def __init__(self, endpoint: str, api_key: str, deployment: str, config = {}):
        super().__init__()
        self.top_p = config.get('top_p', 1.0)
        self.temperature = config.get('temperature', 0.5)
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.load_model()

    def load_model(self):
        self.client = AzureOpenAI(
            api_version="2025-01-01-preview",  
            api_key=self.api_key, 
            azure_endpoint=self.endpoint
        )

    def process_audio(self, audio_path: str, prompt=None, addtional_system_prompt=None, **kwargs) -> str:
        # Read and encode audio file
        content = []
        if prompt is None and audio_path is None:
            raise ValueError('At least one of prompt or audio_path must be provided')
        if prompt is not None:
            content.append({ 
                "type": "text", 
                "text": prompt
            })
        if audio_path==None:
            # Create a 1-second silent audio with a sample rate of 16000
            sample_rate = 16000
            silent_audio = np.zeros(sample_rate * 1)
            os.makedirs('temp', exist_ok=True)
            audio_path = 'temp/none.wav'
            sf.write(audio_path, silent_audio, sample_rate)
        with open(audio_path, 'rb') as wav_reader: 
            encoded_string = base64.b64encode(wav_reader.read()).decode('utf-8') 
            content.append({
                "type": "input_audio", 
                "input_audio": { 
                    "data": encoded_string, 
                    "format": "wav" 
                } 
            })

        output_audio_path = kwargs.get("output_audio_path", None)
        if addtional_system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": addtional_system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        if output_audio_path:
            completion = self.client.chat.completions.create( 
                model=self.deployment, 
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p
            )
            wav_bytes=base64.b64decode(completion.choices[0].message.audio.data)
            with open(kwargs.get("output_audio"), "wb") as f:
                f.write(wav_bytes)
            return output_audio_path
        else:
            completion = self.client.chat.completions.create( 
                model=self.deployment, 
                modalities=["text"],
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return completion.choices[0].message.content