import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from .AudioLM import AudioLM

class Qwen(AudioLM):
    def __init__(self, config = {}):
        super().__init__()
        self.max_length = config.get('max_new_tokens', 512)
        self.load_model()
    
    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", 
            device_map=self.device
        )
    
    def process_audio(self, audio_path: str, prompt=None, addtional_system_prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        content = []
        if audio_path:
            content.append({"type": "audio", "audio_url": audio_path})
        if prompt:
            content.append({"type": "text", "text": prompt})
        if addtional_system_prompt:
            conversation = [{
                "role": "system", 
                "content": addtional_system_prompt
            }]
        else:
            conversation = []
        conversation.append({
                "role": "user", 
                "content": content
            })
        text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_data = librosa.load(
                            ele['audio_url'], 
                            sr=self.processor.feature_extractor.sampling_rate
                        )[0]
                        audios.append(audio_data)

        if audios == []:
            audios = None
        
        inputs = self.processor(
            text=text, 
            audios=audios, 
            return_tensors="pt", 
            padding=True
        )
        # inputs.input_ids = inputs.input_ids.to(self.device)
        inputs = inputs.to(self.device)
        
        generate_ids = self.model.generate(**inputs, max_length=self.max_length)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
