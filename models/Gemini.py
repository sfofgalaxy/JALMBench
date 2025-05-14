import google.generativeai as genai
from .AudioLM import AudioLM
import time
from google.generativeai.types import BlockedPromptException

class Gemini(AudioLM):
    def __init__(self, api_key: str, model_name: str, config = {}):
        super().__init__()
        self.generation_config=genai.types.GenerationConfig(
            max_output_tokens=config.get('max_new_tokens', 512),
            temperature=config.get('temperature', 0.5),
        )
        self.max_length = config.get('max_new_tokens', 512)
        self.api_key = api_key
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def process_audio(self, audio_path: str, prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        content = []
        if audio_path:
            uploaded_file = genai.upload_file(audio_path)
            content.append(uploaded_file)
        if prompt:
            content.append(prompt)
        
        result = self.model.generate_content(content, generation_config=self.generation_config)
        if uploaded_file:
            uploaded_file.delete()
        # avoid quota limit rate in process_batch
        time.sleep(2)
        if result.prompt_feedback.block_reason:
            print(result)
            print(result.prompt_feedback.block_reason)
            return f"[[Error]] block_reason: {result.prompt_feedback.block_reason}"
        return result.text

