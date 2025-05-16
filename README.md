# JALMBench
JALMBench is a modular benchmark framework designed to evaluate jailbreak attacks and defenses against ALMs. Currently, \Bench supports 12 ALMs, 8 jailbreak attacks (4 text-transferred methods and 4 audio-originated methods), and 5 defense methods.  It is highly extensible.

## Disclaimer
This project is for research and educational purposes only. It is not intended for commercial use. 
<span style="color:red;">This repository contains examples of harmful language.</span>.


---

## Supported Models

1. 💬 **SpeechGPT**
2. ⭐ **Spirit LM**
3. 🗣️ **GLM-4-Voice**
4. 🐟 **SALMONN**
5. 🎙️ **Qwen2-Audio-7B-Instruct**
6. 🦙 **LLaMA-Omni**
7. 🔊 **DiVA**
8. ❄️ **Freeze-Omni**
9. 🧠 **VITA-1.0**
10. ⚡ **VITA-1.5**
11. 🤖 **GPT-4o-Audio (OpenAI)**
12. 🌟 **Gemini-2.0-Flash**

---

### Notes
- Models are listed in alphabetical order.
- For detailed information on each model, please refer to their respective documentation or official websites.

## Requirements
<!-- pip install sox
pip install git+https://github.com/facebookresearch/WavAugment.git -->
insall cuda-12.4

## Usage
### 1. Create Environment
#### `Docker` (recommend)
The CUDA version on the host device is 12.4. Mounting CUDA is mandatory, while mounting cached models, Hugging Face, and Transformer cache are optional. To clone the repository, use the following command:

git clone https://github.com/sfofgalaxy/JALMBench.git

```bash
docker run -it \
-v /path/to/cuda-12.4:/usr/local/cuda \
-v /path/to/cached_models:/home/cached_models \
-v /data_sda/zf/cache:/home/cache \
-v path/to/xxx/:/home/xxx/ \
-e CUDA_HOME="/usr/local/cuda" \
-e TRANSFORMERS_CACHE="/data_sda/zf/cache/transformers" \
-e HF_HOME="/data_sda/zf/cache/huggingface" \
--gpus all \
--name benchmark \
ziffer99/audio-benchmark:0.1

conda activate bm

cd /home/JALMBench
```

#### `Conda`
1. install `g++`
2. set environment variable CUDA_HOME
3. execute:
create the environment with `conda`
```bash
conda env create -n bm -f environment.yml
conda activate bm
```

#### `pip`
1. install `g++`
2. set environment variable CUDA_HOME
3. execute:
```bash
conda create -n bm python=3.10.16
conda activate bm
pip install pip==24.0
pip install -r requirements.txt
```

### 2. Download Models

Make sure you have git-lfs installed (https://git-lfs.com)

Firstly, you may create a folder to restore these models. For example:
```bash
mkdir cached_models/
```

### 💬 SpeechGPT
```bash
# Download SpeechGPT Models
mkdir cached_models/SpeechGPT
cd cached_models/SpeechGPT
git clone https://huggingface.co/fnlp/SpeechGPT-7B-cm
git clone https://huggingface.co/fnlp/SpeechGPT-7B-com

# Download speech2unit model
mkdir cached_models/speech2unit
cd cached_models/speech2unit
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin

# Download vocoder model
mkdir cached_models/vocoder
cd cached_models/vocoder
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O config.json
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O vocoder.pt
``` 

### ⭐ Spirit LM
Please follow the instrcutions in [Spirit LM Checkpoints](https://github.com/facebookresearch/spiritlm/blob/main/checkpoints/README.md) to download the checkpoints of speech encoder and spirit LM model.

### 🗣️ GLM4Voice
```bash
cd cached_models/
git clone https://huggingface.co/THUDM/glm-4-voice-decoder
```
### 🐟 SALMONN
```bash
mkdir cached_models/SALMONN
cd cached_models/SALMONN

# Download whisper-large-v2
git clone https://huggingface.co/openai/whisper-large-v2

# Download vicuna-13b-v1.1
git clone https://huggingface.co/lmsys/vicuna-13b-v1.1

# Download BEATs
curl -L https://djf19a.dm.files.1drv.com/y4mwa3IjEyKMl5kK57BQ65XVHjIuMPD428GnN1PY9qkBfoWBAT1PURVSnl7Mq5fIRM00MWe0_IwM8F-PjWimiXCiCs_sZf6jy3-LaJtsnKzfTvQ5ogwyJvEnOdhWBnAizxkLS5EzrutX2U6GbQaMejUK6CN8DcJsjPXGGbBZfTWZN1kR7icLDiaujL5_4zrb0CBAZc30DPXWNmYsiTNqVHfKA -o BEATs.pt

# Download salmoon v1 ckpt
curl -L https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth -o salmonn_v1.pth
```

### 🦙 LLaMA-Omni
```bash
mkdir cached_models/LLaMA-Omni
cd cached_models/LLaMA-Omni
git clone https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni
# vocoder
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P vocoder/
```
### ❄️ FreezeOmni
```bash
mkdir cached_models/
cd cached_models/
git clone https://huggingface.co/VITA-MLLM/Freeze-Omni
```

### 🧠 VITA & ⚡ VITA-1.5
```bash
# may need to change the device map in the load_pretrained_model() function in vita/model/builder.py according to the GPU memory
mkdir cached_models/VITA
cd cached_models/VITA
git clone https://huggingface.co/VITA-MLLM/VITA
git clone https://huggingface.co/VITA-MLLM/VITA-1.5
git clone https://huggingface.co/OpenGVLab/InternViT-300M-448px
```
---
Note: Other models will be downloaded during usage, you can also download them in advance.

### 3. Configuration
Please refer to the `.env` file in the root directory.

## Quick Start

```bash
conda activate bm
```

### 1. Generate from audio
```bash
python process_single.py \
--model qwen \
--audio 0.wav \
--output_text ./test/0.txt \
```

### 2. Generate from text with TTS model
```bash
python process_single.py \
--model qwen \
--text "What is the court size of a standard indoor volleyball game?" \
--output_text ./test/0.txt \
--output_audio ./test/0.wav
--tts_model google_api
```

### 3. Generate from audio (or text with TTS model) with prompt
```bash
python process_single.py \
--model qwen \
--audio 0.wav \
--prompt "Answer this audio." \
--output_text ./test/0.txt \
--output_audio ./test/0.wav
```

**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. (e.g., `qwen`, `diva`).
- `--audio`: Input audio path.
- `--prompt`: Prompt, for example: "Answer the speacker's question of this audio." 
- `--output_text`: The output text path.
- `--output_audio`: The output audio path (if applicable).

The short for model selection in the brackets.
1. 💬 **SpeechGPT** (speechgpt)
2. ⭐ **Spirit LM** (spirit)
3. 🗣️ **GLM4Voice** (glm)
4. 🐟 **SALMONN** (salmonn)
5. 🎙️ **Qwen2-Audio** (qwen)
6. 🦙 **LLaMA-Omni** (lo)
7. 🔊 **DiVA** (diva)
8. ❄️ **FreezeOmni** (fo)
9. 🧠 **VITA-1.0** (vita_1)
10. ⚡ **VITA-1.5** (vita_1_5)
11. 🤖 **OpenAI GPT** (gpt)
12. 🌟 **Gemini-2.0-Flash** (gemini)


## Evaluation
The following steps are used to evaluate the datasets in our paper, which can be accessed from this [link](https://huggingface.co/datasets/AnonymousUser000/JALMBench).

### Step 1
Get the ALM's Response with the above commands or the following Commands.
```shell
python main.py --model qwen --data aharm --modality audio
```
**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. (e.g., `qwen`, `diva`).
- `--data`: Selects the subset of the dataset. Replace `aharm` with other subsets like `tharm` (text only), `pap`, etc., depending on your evaluation needs.
- `--modality`: Use `audio` for spoken modality input, `text` for text modality input.
This will generate the output and save it to a file named `qwen-aharm-audio.jsonl` in the root folder.
- `--defense`: Defense methods used, default: None (i.e. no defense). (e.g., `AdaShield`, `LLaMAGuard`)

Please refer to `main.py` for more details.

### Step 2

For datasets `alpacaeval`, `commoneval`, `wildvoice`, and `sd-qa`, we use `gpt-4o-mini` to evaluate the responses. Run the following command to get the GPT score:
```shell
python evaluation/evaluator.py --file qwen-aharm-audio.jsonl
```
The GPT evaluation scores will be saved to `result-qwen-aharm-audio.jsonl`.

### Step3
To generate the final attack success rate (ASR), run:
```shell
python evaluation/get_result.py --file result-qwen-aharm-audio.jsonl
```


## Discussion & Communication

Please feel free to contact us via email: `zpengao@connect.hkust-gz.edu.cn` or directly discuss on [Github Issues](https://github.com/sfofgalaxy/JALMBench/issues).

## Acknowledgements

This project is built upon the following repositories:
- [SpeechGPT](https://github.com/0nutation/SpeechGPT)
- [SpiritLM](https://github.com/facebookresearch/spiritlm)
- [GLM4Voice](https://github.com/THUDM/GLM-4-Voice)
- [SALMONN](https://github.com/bytedance/SALMONN)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni)
- [DiVA](https://diva-audio.github.io/)
- [FreezeOmni](https://freeze-omni.github.io/)
- [VITA-1.0](https://github.com/VITA-MLLM/VITA/tree/vita-1.0)
- [VITA-1.5](https://github.com/VITA-MLLM/VITA)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-quickstart?pivots=ai-foundry-portal) and [OpenAI GPT](https://platform.openai.com/docs/guides/realtime)
- [Gemini](https://gemini.google.com/)

---

<!-- ## Citation
If you use the JALMBench in your research, please cite the following paper:
```
@article{
}
``` -->