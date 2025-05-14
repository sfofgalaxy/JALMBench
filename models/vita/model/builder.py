import warnings

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging

from ..model import *

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_pretrained_model(
    model_path,
    model_type,
    mm_audio_encoder,
    mm_vision_tower,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    if model_type not in {"mixtral-8x7b"}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if model_type == "mixtral-8x7b":
        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 1,
            "model.layers.13": 1,
            "model.layers.14": 1,
            "model.layers.15": 1,
            "model.layers.16": 1,
            "model.layers.17": 1,
            "model.layers.18": 1,
            "model.layers.19": 1,
            "model.layers.20": 1,
            "model.layers.21": 1,
            "model.layers.22": 1,
            "model.layers.23": 1,
            "model.layers.24": 2,
            "model.layers.25": 2,
            "model.layers.26": 2,
            "model.layers.27": 2,
            "model.layers.28": 2,
            "model.layers.29": 2,
            "model.layers.30": 2,
            "model.layers.31": 2,
            "model.norm": 2,
            "model.vision_tower": 2,
            "model.mm_projector": 2,
            "model.audio_encoder": 2,
            "lm_head": 2,
        }
        device_map["model.audio_encoder"] = 0
        kwargs.update(device_map=device_map)
        config = AutoConfig.from_pretrained(model_path)
        config.mm_vision_tower = mm_vision_tower
        config.mm_audio_encoder = mm_audio_encoder
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, config=config)
        model = VITAMixtralForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, config=config, **kwargs
        )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))


    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if model_type == "phi-3":
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model, image_processor, context_len
