from .AudioLM import AudioLM
import torch
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import json
import os
import re
import soundfile as sf
from peft import PeftModel
from .speechgpt.utils.speech2unit.speech2unit import Speech2Unit
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import sys
from typing import List
from tqdm import tqdm

def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response

class SpeechGPT(AudioLM):
    def __init__(self, model_path: str, lora_weights: str, s2u_path: str, vocoder_path: str, config = {}):
        super().__init__()
        self.meta_instruction = """You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n"""
        self.template = "[Human]: {question} <eoh>. [SpeechGPT]: "
        self.template_jailbreak_1 = "[Human]: {question}"
        self.template_jailbreak_2 = "<eoh>. [SpeechGPT]: "

        # Load configuration
        self.model_path = model_path
        self.lora_weights = lora_weights
        self.s2u_dir = s2u_path
        self.vocoder_dir = vocoder_path
        
        # Generation parameters
        self.generate_kwargs = {
            "max_new_tokens": config.get('max_new_tokens', 512),
            "min_new_tokens": config.get('min_new_tokens', 10),
            "temperature": config.get('temperature', 0.5),
            "do_sample": config.get('do_sample', True),
            "top_k": config.get('top_k', 60),
            "top_p": config.get('top_p', 0.8),
        }
        
        self.load_model()

    def load_model(self):
        """Initialize SpeechGPT model components"""
        # Initialize speech2unit
        self.s2u = Speech2Unit(ckpt_dir=self.s2u_dir)

        # Load main model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Load LoRA weights if specified
        if self.lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # Initialize tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        # Initialize vocoder
        vocoder_path = os.path.join(self.vocoder_dir, "vocoder.pt")
        vocoder_cfg_path = os.path.join(self.vocoder_dir, "config.json")
        with open(vocoder_cfg_path) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder_path, vocoder_cfg).to(self.device)

    def process_audio(self, audio_path: str, prompt=None, **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        
        with torch.no_grad():
            # Preprocess
            if audio_path and prompt:
                preprocessed_prompts = [self._preprocess(audio_path), self._preprocess(prompt)]
            elif audio_path:
                preprocessed_prompts = [self._preprocess(audio_path)]
            else:
                preprocessed_prompts = [self._preprocess(prompt)]
            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            input_ids = input_ids.to(self.device)

            # Generate
            generation_config = GenerationConfig(**self.generate_kwargs)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            # Postprocess
            response = self._postprocess(responses[0])
            # Generate audio if needed
            output_audio_path = kwargs.get('output_audio_path', None)
            
            if output_audio_path and response["answer"] and '<sosp>' in response["answer"]:
                unit = [int(num) for num in re.findall(r'<(\d+)>', response["answer"])]
                x = {"code": torch.LongTensor(unit).view(1, -1).to(self.device)}
                wav = self.vocoder(x, True)
                sf.write(output_audio_path, wav.detach().cpu().numpy(), 16000)

            return response["answer"]

    def _preprocess(self, raw_text: str):
        processed_parts = []
        for part in raw_text.split("is input:"):
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp3", ".mp4"]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)
        prompt_seq = f"{self.meta_instruction}\n" + self.template.format(question=processed_text)
        return prompt_seq
    
    
    def _postprocess(self, response: str):
        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[ta]", tag2="[ua]") if "[ua]" in response and '[ta]' in response else ""
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question":question, "answer":answer, "unitAnswer":ua}

