import torch
import torchaudio
from .AudioLM import AudioLM
from .spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType
from transformers import GenerationConfig

class SpiritLM(AudioLM):
    def __init__(self, model_name="spirit-lm-base-7b", model_path: str = None, speech_encoder_path: str = None, config = {}):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.speech_encoder_path = speech_encoder_path
        self.temperature = config.get('temperature', 0.5)
        self.top_p = config.get('top_p', 0.9)
        self.max_new_tokens = 512
        self.load_model()

    def load_model(self):
        self.model = Spiritlm(self.model_name, self.model_path, self.speech_encoder_path)

    def process_audio(self, audio_path: str, prompt=None, addtional_system_prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        inputs = []
        if addtional_system_prompt:
            inputs.append(GenerationInput(
                    content=addtional_system_prompt,
                    content_type=ContentType.TEXT,
                ))
        if audio_path:
            inputs.append(GenerationInput(
                        content=audio_path,
                        content_type=ContentType.SPEECH,
                    ))
        if prompt:
            inputs.append(GenerationInput(
                    content=prompt,
                    content_type=ContentType.TEXT,
                ))
        outputs = self.model.generate(
            output_modality=OutputModality.TEXT,
            interleaved_inputs=inputs,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            ),
        )
        
        text_response = outputs[0].content
        
        output_audio_path = kwargs.get("output_audio_path")
        if output_audio_path:
            audio_outputs = self.model.generate(
                output_modality=OutputModality.SPEECH,
                interleaved_inputs=inputs,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                ),
            )
            
            audio_tensor = torch.from_numpy(audio_outputs[0].content)
            torchaudio.save(output_audio_path, audio_tensor.unsqueeze(0), 16000)

        return text_response