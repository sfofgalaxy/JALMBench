# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import WhisperFeatureExtractor
from .AudioLM import AudioLM
from .salmonn.config import Config
from .salmonn.utils import prepare_one_sample
from .salmonn.models.salmonn import SALMONN
import argparse

class SalmonnModel(AudioLM):
    def __init__(self, llama_path: str, whisper_path: str, beats_path: str, salmonn_ckpt_path: str, config = {}):
        super().__init__()
        args = {"cfg_path": "./models/salmonn/decode_config.yaml", "device": "cuda" if torch.cuda.is_available() else "cpu"}
        args = argparse.Namespace(**args)
        cfg = Config(args)
        cfg.config.model.llama_path = llama_path
        cfg.config.model.whisper_path = whisper_path
        cfg.config.model.beats_path = beats_path
        cfg.config.model.ckpt = salmonn_ckpt_path
        cfg.config.generate.max_new_tokens = config.get('max_new_tokens', 512)
        cfg.config.generate.num_beams = config.get('num_beams', 1)
        cfg.config.generate.temperature = config.get('temperature', 0.5)
        cfg.config.generate.top_p = config.get('top_p', 1.0)

        self.config = cfg.config
        self.load_model()
        
    def load_model(self):
        self.model = SALMONN.from_config(self.config.model)
        self.model.to(self.device)
        self.model.eval()
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.config.model.whisper_path)
        
    def process_audio(self, audio_path: str, prompt: str = "", **kwargs) -> str:
        if not audio_path and not prompt:
            raise ValueError("Either audio_path or prompt must be provided")
        samples = prepare_one_sample(audio_path, self.wav_processor)
        formatted_prompt = [
            self.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + (prompt.strip() if prompt!=None and prompt!="" else "Please answer the speaker's question in detail."))
        ]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = self.model.generate(
                samples, 
                self.config.generate,
                prompts=formatted_prompt
            )[0]
            
        return output.strip()