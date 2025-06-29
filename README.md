Official Repo for [JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models
](https://www.arxiv.org/abs/2505.17568)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/AnonymousUser000/JALMBench)
[![arXiv](https://img.shields.io/badge/arXiv-2505.17568-b31b1b.svg)](https://arxiv.org/abs/2505.17568)


![Teaser Preview](teaser.png)

<span style="color:red; font-size:16px; font-weight:bold;">Disclaimer:</span>
<span style="color:red; font-size:16px;">This project is for research and educational purposes only. It is not intended for commercial use. </span>
<span style="color:red; font-size:16px;">This repository contains examples of harmful language. </span>

---

# JALMBench

JALMBench is a modular benchmark framework designed to evaluate jailbreak attacks and defenses against ALMs. Currently, JALMBench supports 12 ALMs, 8 jailbreak attacks (4 text-transferred methods and 4 audio-originated methods), and 5 defense methods.  It is highly extensible.

---

## Usage

### 1. Download Models

We recommend you download models on your own server or computer. For example, download all required models in the folder `cached_models`. You may download the models you want to evaluate.

---

Make sure you have git-lfs installed (https://git-lfs.com)

Firstly, you may create a folder to restore these models. For example:
```bash
mkdir cached_models/
```

#### 💬 SpeechGPT
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

#### ⭐ Spirit LM
Please follow the instrcutions in [Spirit LM Checkpoints](https://github.com/facebookresearch/spiritlm/blob/main/checkpoints/README.md) to download the checkpoints of speech encoder and spirit LM model.

#### 🗣️ GLM4Voice
```bash
cd cached_models/
git clone https://huggingface.co/THUDM/glm-4-voice-decoder
```
#### 🐟 SALMONN
```bash
mkdir cached_models/SALMONN
cd cached_models/SALMONN

# Download whisper-large-v2
git clone https://huggingface.co/openai/whisper-large-v2

# Download vicuna-13b-v1.1
git clone https://huggingface.co/lmsys/vicuna-13b-v1.1

# Download BEATs from https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea
# Please Refer to https://github.com/bytedance/SALMONN?tab=readme-ov-file
curl -L <BEATS_DOWNLOAD_URL> -o BEATs.pt

# Download salmoon v1 ckpt
curl -L https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth -o salmonn_v1.pth
```

#### 🦙 LLaMA-Omni
```bash
mkdir cached_models/LLaMA-Omni
cd cached_models/LLaMA-Omni
git clone https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni
# vocoder
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P vocoder/
```
#### ❄️ FreezeOmni
```bash
mkdir cached_models/
cd cached_models/
git clone https://huggingface.co/VITA-MLLM/Freeze-Omni
```

#### 🧠 VITA & ⚡ VITA-1.5
```bash
# may need to change the device map in the load_pretrained_model() function in vita/model/builder.py according to the GPU memory
mkdir cached_models/VITA
cd cached_models/VITA
git clone https://huggingface.co/VITA-MLLM/VITA
git clone https://huggingface.co/VITA-MLLM/VITA-1.5
git clone https://huggingface.co/OpenGVLab/InternViT-300M-448px
```

Note: Other models will be downloaded during usage, you can also download them in advance.

---
### 2. Create Environment
Pre-requirement:
<!-- pip install sox
pip install git+https://github.com/facebookresearch/WavAugment.git -->
You should insall cuda-12.4 on your own server or computer.

---

#### `Docker` (recommend)
The CUDA version on the host device is 12.4. Mounting CUDA is mandatory, while mounting cached models, Hugging Face, and Transformer cache are optional. To clone the repository, use the following command:

git clone https://github.com/sfofgalaxy/JALMBench.git

```bash
git clone https://github.com/sfofgalaxy/JALMBench.git

docker run -it \
-v /path/to/cuda-12.4:/usr/local/cuda \
-v /path/to/cached_models:/home/cached_models \
-v /path/to/cache:/home/cache \
-v path/to/JALMBench/:/home/JALMBench/ \
-e CUDA_HOME="/usr/local/cuda" \
-e TRANSFORMERS_CACHE="/home/cache/transformers" \
-e HF_HOME="/home/cache/huggingface" \
--gpus all \
--name benchmark \
ziffer99/audio-benchmark:0.1

conda activate bm

cd /home/JALMBench
```
Note that `TRANSFORMERS_CACHE`, `HF_HOME`, and `-v /path/to/cache:/home/cache` are optional, you may choose based on your own path.


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

### 3. Configuration
Please refer to the `.env` file in the root directory.

## Quick Start

```bash
conda activate bm
```

### 1. Generate from Audio
```bash
python process_single.py \
--model qwen \
--audio 0.wav \
--output_text 0.txt
```

### 2. Generate from Text with TTS Model
```bash
python process_single.py \
--model qwen \
--text "What is the court size of a standard indoor volleyball game?" \
--output_text 0.txt \
--output_audio 0.wav
--tts_model google_api
```

### 3. Generate from Audio (or Text with TTS Model) with Prompt
```bash
python process_single.py \
--model qwen \
--audio 0.wav \
--prompt "Answer this audio." \
--output_text 0.txt \
--output_audio 0.wav
```

**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. (e.g., `qwen`, `diva`).
- `--audio`: Input audio path.
- `--prompt`: Prompt, for example: "Answer the speacker's question of this audio." 
- `--output_text`: The output text path.
- `--output_audio`: The output audio path (if applicable).

---
##### Supported Models:

For detailed information on each model, please refer to their respective documentation or official websites.
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
11. 🤖 **OpenAI GPT-4o-Audio** (gpt)
12. 🌟 **Gemini-2.0-Flash** (gemini)

## Evaluation
The following steps are used to evaluate the datasets in our paper, which can be accessed from this [link](https://huggingface.co/datasets/AnonymousUser000/JALMBench).

### Step 1
Get the ALM's Response with the above commands or the following Commands.
```shell
python main.py --model qwen --data AHarm --modality audio
```
**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. (e.g., `qwen`, `diva`).
- `--data`: Selects the subset of the dataset. Replace `AHarm` with other subsets like `THarm` (text only), `PAP`, etc., depending on your evaluation needs.
- `--modality`: Use `audio` for spoken modality input, `text` for text modality input. This will generate the output and save it to a file named `qwen-aharm-audio.jsonl` in the root folder.
- `--defense`: Defense methods used, default: None (i.e. no defense). (e.g., `AdaShield`, `LLaMAGuard`)
- `--language`: (Optional) Filter by language, e.g. `en`.
- `--gender`: (Optional) Filter by gender, e.g. `male`, `female`, `Neutral`.
- `--accent`: (Optional) Filter by accent, e.g. `US`, `GB`, `AU`, `IN`.

Example with filtering:
```shell
python main.py --model qwen --data ADiv --modality audio --language en --gender female --accent US
```
This command will only process samples in the ADiv subset where language is English, gender is female, and accent is US.

---
This will generate the output and save it to a file named `qwen-AHarm-audio.jsonl` in the root folder. Please refer to `main.py` for more details.

### Step 2
Before evaluating the results, you need to configure several parameters in `evaluation/evaluator.py` by replacing `"xxx"` in it if you use Azure OpenAI API, you can easily change: `deployment_name`, `api_key`, `endpoint`. (if you are using API provided by OpenAI, you should use `evaluation/evaluator_openai.py` accordingly)
For datasets `alpacaeval`, `commoneval`, `wildvoice`, and `sd-qa`, we use `gpt-4o-mini` to evaluate the responses. Run the following command to get the GPT score:
```shell
# or evaluation/evaluator_openai.py if you are using API provided by OpenAI
python evaluation/evaluator.py --file qwen-AHarm-audio.jsonl
```
The GPT evaluation scores will be saved to `result-qwen-AHarm-audio.jsonl`.

### Step3
To generate the final attack success rate (ASR), run:
```shell
python evaluation/get_result.py --file result-qwen-AHarm-audio.jsonl
```

For attack methods like AMSE, BoN, PAP, AdvWave where multiple attack attempts share the same ID and any successful attack counts as success for that ID, use the `by_id` method:
```shell
python evaluation/get_result.py --file result-qwen-AHarm-audio.jsonl --method by_id
```

**Supported Arguments:**
- `--file`: Path to the result jsonl file
- `--method`: ASR calculation method. Options:
  - `simple` (default): Calculate ASR based on individual samples
  - `by_id`: Group by ID and consider an ID successful if any sample with that ID succeeds (for AMSE/BoN/PAP/AdvWave attacks)

**NOTE**
If you are evaluating the `AdvWave` method performance, you should combine results from both `AHarm` and `AdvWave` since the `AdvWave` method only optimizes samples that originally failed in the `AHarm` attack, requiring the combined evaluation of both datasets for the final assessment.

## OpenBookQA Evaluation Utility

For evaluating models on the OpenBookQA dataset, we provide a streamlined 3-step workflow in the `utility/` directory:

### Step 1: Generate Model Predictions
```bash
python utility/generate.py --model qwen --defense no_defense
```
This script will:
- Check if OpenBookQA audio files exist, download them if not
- Evaluate the specified model on the audio dataset
- Save prediction results to `{model_name}-openbookqa-utility-{defense_method}-generate.json`

### Step 2: Extract Answer Choices
```bash
python utility/extract_answers.py --model qwen --defense no_defense
```
This script will:
- Process prediction files and extract multiple choice answers (A/B/C/D)
- Use regex pattern matching first for fast extraction
- Fall back to LLM-based extraction when regex fails
- Save processed results to `{model_name}-openbookqa-utility-{defense_method}-answer.json`

### Step 3: Calculate Accuracy
```bash
python utility/calculate_accuracy.py --model qwen --defense no_defense
```
This script will:
- Calculate accuracy for the specified model and defense method
- Display accuracy percentage results

---

## Project Structure Description

```bash
├─audio_edit_tools
├─defense
│  ├─output_filter
│  └─prompts
├─evaluation
├─matcha
├─models
└─utility
```
- `models`: This contains classes related to ALMs (Audio Language Models). You can directly implement the `AudioLM.py` class to add new ALMs. Modifying the `process_text(·)` function in `AudioLM.py` allows you to change the TTS tool.

- `matcha`: Related to models, includes utilities for handling models.

- `defense`: Contains various defense methods.

- `evaluation`: Used to evaluate the models.

- `audio_edit_tools`: A set of tools for audio editing (preprocessing), including speed adjustment, volume control, adding white noise, adding background sounds, pitch modification, smoothing, codecs (compression), audio filters (high/low frequency filtering), echo, quantization (bit reduction), and audio concatenation (e.g., appending harmful queries before or after an audio clip).

- `utility`: To evaluate the utility on the openbookqa dataset.

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

## Citation
If you use the JALMBench in your research, please cite the following paper:
```
@misc{peng2025jalmbenchbenchmarkingjailbreakvulnerabilities,
      title={JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models}, 
      author={Zifan Peng and Yule Liu and Zhen Sun and Mingchen Li and Zeren Luo and Jingyi Zheng and Wenhan Dong and Xinlei He and Xuechao Wang and Yingjie Xue and Shengmin Xu and Xinyi Huang},
      year={2025},
      eprint={2505.17568},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.17568}, 
}
```