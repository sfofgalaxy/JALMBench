# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import whisper
from .omni_speech_llama import OmniSpeechLlamaForCausalLM
from .omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from .speech_encoder.speech_encoder import build_speech_encoder


def load_pretrained_model(model_cache, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM

    speech_encoder_path = f"{model_cache}/speech_encoder/large-v3.pt"
    # Load OmniSpeech model
    tokenizer = AutoTokenizer.from_pretrained(model_cache, use_fast=False)
    pretrained_config = AutoConfig.from_pretrained(model_cache)
    pretrained_config.speech_encoder = speech_encoder_path
    model = model_cls.from_pretrained(
        model_cache,
        low_cpu_mem_usage=False,
        config=pretrained_config,
        **kwargs
    )
    model = model.to(device=device)

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
