# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import torch

from .hubert_tokenizer import HubertTokenizer

CURRENT_DEVICE = (
    torch.device(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def spiritlm_hubert(speech_encoder_path):
    return HubertTokenizer(
        hubert_ckpt=speech_encoder_path / "hubert_25hz/mhubert_base_25hz.pt",
        hubert_layer=11,
        quantizer_ckpt=speech_encoder_path / "hubert_25hz/L11_quantizer_500.pt",
        is_linear_quantizer=True,
    ).to(CURRENT_DEVICE)
