# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import math
from typing import Union

import pandas as pd
import torch
import torchaudio
from .stsp_constants import STSP_DATA_ROOT, STSP_MANIFEST_ROOT
from ...model.spiritlm_model import Spiritlm

FEW_SHOT_MANIFEST_DIR = STSP_MANIFEST_ROOT / "few_shot"
FEW_SHOT_TEMPLATE = "{prompt}{generation}"


def wav_prompt(spiritlm_model: Spiritlm, wav: Union[str, torch.Tensor]) -> str:
    return spiritlm_model.SPEECH_PROMPT_PREFIX + spiritlm_model.speech_tokenizer(wav)


def text_prompt(spiritlm_model: Spiritlm, text: str) -> str:
    return spiritlm_model.TEXT_PROMPT_PREFIX + text


def _load_half_wav(wav_path: str, load_first_half: bool) -> torch.Tensor:
    wav_path = STSP_DATA_ROOT / wav_path
    wav = torchaudio.load(wav_path)[0].squeeze(0)
    size = wav.size()[0]
    half_size = size // 2
    if load_first_half:
        wav = wav[:half_size]
    else:
        wav = wav[half_size:]
    return wav


def build_few_shot_prompt(
    spiritlm_model: Spiritlm,
    input_output: str,
    n_shots: int = 3,
) -> str:
    """
    Build the few-shot prompt by simply concatenating a set of examples.

    E.g., a 3-shots T->S prompt would like this:
    "[Text]text1[Speech]speech_tokens1\n[Text]text2[Speech]speech_tokens2\n[Text]text3[Speech]speech_tokens3\n"
    """
    manifset_file_mapping = {
        "text_text": "t2t",
        "speech_text": "s2t",
        "text_speech": "t2s",
        "speech_speech": "s2s",
    }
    manifest_path = (
        FEW_SHOT_MANIFEST_DIR / f"{manifset_file_mapping[input_output]}.jsonl"
    )
    df = pd.read_json(manifest_path, lines=True)
    assert n_shots <= len(df)

    # ensure a balanced sampels for each sentiment
    nb_samples_per_sentiment = math.ceil(n_shots / 3)
    df = df.groupby("sentiment").sample(n=nb_samples_per_sentiment)

    prompts = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        generation = row["generation"]
        if input_output == "text_text":
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=text_prompt(spiritlm_model, prompt),
                generation=text_prompt(spiritlm_model, generation),
            )
        elif input_output == "text_speech":
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=text_prompt(spiritlm_model, prompt),
                generation=wav_prompt(
                    spiritlm_model, _load_half_wav(generation, load_first_half=False)
                ),
            )
        elif input_output == "speech_text":
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=wav_prompt(
                    spiritlm_model, _load_half_wav(prompt, load_first_half=True)
                ),
                generation=text_prompt(spiritlm_model, generation),
            )
        elif input_output == "speech_speech":
            prompt = FEW_SHOT_TEMPLATE.format(
                prompt=wav_prompt(
                    spiritlm_model, _load_half_wav(prompt, load_first_half=True)
                ),
                generation=wav_prompt(
                    spiritlm_model, _load_half_wav(generation, load_first_half=False)
                ),
            )
        prompts.append(prompt)
    print(f"prompts: {prompts}")
    return "\n".join(prompts) + "\n"
