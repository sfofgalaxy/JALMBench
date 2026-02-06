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
    device_map="cuda:0",
    device="cuda",
    **kwargs,
):
    if model_type not in {"mixtral-8x7b", "nemo", "qwen2p5_instruct", "qwen2p5_fo_instruct"}:
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

    if model_type == "nemo":
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VITAMistralForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif model_type == "qwen2p5_instruct":
        config = AutoConfig.from_pretrained(model_path)
        config.mm_vision_tower = mm_vision_tower
        config.mm_audio_encoder = mm_audio_encoder
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, config=config)
        model = VITAQwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, config=config, **kwargs
        )
    elif model_type == "qwen2p5_fo_instruct":
        # import pdb; pdb.set_trace()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VITAFOQwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
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

