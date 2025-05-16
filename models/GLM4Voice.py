from transformers import AutoModel, WhisperFeatureExtractor, AutoTokenizer
import torch
from .AudioLM import AudioLM
from .GLM.flow_inference import AudioDecoder
from .GLM.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .GLM.speech_tokenizer.utils import extract_speech_token
import uuid
import torchaudio

class GLM4Voice(AudioLM):
    def __init__(self, decoder_path: str, config = {}):
        super().__init__()
        self.temperature = config.get('temperature', 0.5)
        self.top_p = config.get('top_p', 1.0)
        self.max_new_tokens = config.get('max_new_tokens', 2048)
        self.load_model(decoder_path)

    def load_model(self, decoder_path: str):
        model_path = "THUDM/glm-4-voice-9b"
        tokenizer_path = "THUDM/glm-4-voice-tokenizer" 
        flow_path = decoder_path
        
        flow_config = f"{flow_path}/config.yaml"
        flow_checkpoint = f"{flow_path}/flow.pt"
        hift_checkpoint = f"{flow_path}/hift.pt"

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=None,
            device_map={"": 0} 
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=self.device
        )
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    @torch.inference_mode()
    def process_audio(self, audio_path: str, prompt=None, addtional_system_prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        if audio_path:
            audio_tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path]
            )[0]
            assert len(audio_tokens) != 0, "No audio tokens extracted"

            audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " + (addtional_system_prompt if addtional_system_prompt else "")
            user_input = audio_tokens
        else:
            user_input = prompt
            system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. " + (addtional_system_prompt if addtional_system_prompt else "")
        inputs = f"<|system|>\n{system_prompt}"
        inputs += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        inputs = self.glm_tokenizer([inputs], return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output_sequences = self.glm_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            generated_sequence = output_sequences[0]

            text_tokens = []
            audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
            for token_id in generated_sequence:
                if token_id < audio_offset:
                    text_tokens.append(token_id)
                else:
                    audio_tokens.append(token_id - audio_offset)
            if len(audio_tokens) > 4096:
                audio_tokens = audio_tokens[-4097:-1]
            if audio_tokens:
                tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
                tts_speech, _ = self.audio_decoder.token2wav(
                    tts_token,
                    uuid=str(uuid.uuid4()),
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                    prompt_feat=torch.zeros(1, 0, 80).to(self.device),
                    finalize=True
                )
                output_audio_path = kwargs.get("output_audio_path", None)
                if output_audio_path:
                    torchaudio.save(output_audio_path, tts_speech.cpu(), 22050, format="wav")

            complete_text = self.glm_tokenizer.decode(text_tokens, skip_special_tokens=False)
            text = complete_text.split("<|assistant|> streaming_transcription\n")[1].split("<|user|>")[0]
            return text