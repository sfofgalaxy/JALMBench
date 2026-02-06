# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
import json
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import whisper
import torch.nn.functional as F

from cog import BasePredictor, Input, Path, BaseModel
from fairseq import utils as fairseq_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from .omni_speech.builder import load_pretrained_model
from .omni_speech.utils import disable_torch_init, ctc_postprocess
from .omni_speech.conversation import conv_templates
from .omni_speech.preprocess import tokenizer_speech_token


# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HOME"] = MODEL_CACHE
# os.environ["TORCH_HOME"] = MODEL_CACHE
# os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
# os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
# os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE



def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["curl", "-L", "-o", f"{dest}/models.tar", url], close_fds=False)
    subprocess.check_call(["tar", "-xf", f"{dest}/models.tar", "-C", dest], close_fds=False)
    # subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def collate_fn(batch):
    input_ids, speech_tensors, speech_lengths = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return input_ids, speech_tensors, speech_lengths

# DataLoader
def create_data_loader(questions, tokenizer, model_config, input_type, mel_size, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config, input_type, mel_size, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config, input_type, mel_size, conv_mode):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')

        return input_ids, speech, torch.LongTensor([speech.shape[0]])

    def __len__(self):
        return len(self.questions)

class Predictor(BasePredictor):
    def setup(self, model_cache: str, vocoder_path: str) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # if not os.path.exists(model_cache):
        #     os.makedirs(model_cache)
        #     download_weights(f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/models.tar", model_cache)

        # Model
        disable_torch_init()
        self.tokenizer, self.model, _ = load_pretrained_model(
            f"{model_cache}", s2s=True, device=self.device
        )

        with open(f"{vocoder_path}/config.json") as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(
            f"{vocoder_path}/g_00500000", vocoder_cfg
        ).to(self.device)


    def predict(
        self,
        input_audio: Path = Input(description="Input audio"),
        prompt: str = Input(
            default=None
        ),
        temperature: float = Input(
            description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        top_p: float = Input(
            description="Controls diversity of the output. Valid when temperature > 0. Lower values make the output more focused, higher values make it more diverse.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate", default=256, ge=1
        ),
        output_audio_path: str = Input(
            description="Output audio path", default=None
        ),
    ) -> str:
        """Run a single prediction on the model"""
        if prompt:
            value = f"<speech>\n{prompt}"
        else:
            value = "<speech>"
        questions = [
            {
                "speech": str(input_audio),
                "conversations": [{"from": "human", "value": value}],
            }
        ]

        data_loader = create_data_loader(
            questions,
            self.tokenizer,
            self.model.config,
            input_type="mel",
            mel_size=128,
            conv_mode="llama_3",
        )

        (input_ids, speech_tensor, speech_length) = next(iter(data_loader))

        input_ids = input_ids.to(device=self.device, non_blocking=True)
        speech_tensor = speech_tensor.to(
            dtype=torch.float16, device=self.device, non_blocking=True
        )
        speech_length = speech_length.to(device=self.device, non_blocking=True)

        with torch.inference_mode():
            output_ids, output_units = self.model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p if temperature > 0 else None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=128004,
            )

        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[
            0
        ].strip()

        output_units = ctc_postprocess(
            output_units, blank=self.model.config.unit_vocab_size
        )

        output_units = [(list(map(int, output_units.strip().split())))]

        x = {
            "code": torch.LongTensor(output_units[0]).view(1, -1),
        }

        x = fairseq_utils.move_to_cuda(x)
        
        # Add padding to the input tensor to ensure it is large enough for the kernel size
        x["code"] = F.pad(x["code"], (1, 1), "constant", 0)
        
        wav = self.vocoder(x, True)
        if output_audio_path:
            sf.write(
                output_audio_path,
                wav.detach().cpu().numpy(),
                16000,
            )
        return prediction
