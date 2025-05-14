# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

"""
Usage example:

cd {SPIRITLM ROOT FOLDER}
export PYTHONPATH=.

# Speech to Text
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_path data/examples/ref.jsonl --eval --write_pred ./pred_s_t.jsonl --input_output speech_text
# Text to Text
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_path data/examples/ref.jsonl --eval --write_pred ./pred_t_t.jsonl --input_output text_text
# Text to Speech#
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_path data/examples/ref.jsonl --eval --write_pred ./pred._t_s.jsonl --input_output text_speech
# Speech to Speech
torchrun --nnodes 1 --nproc-per-node 1 spiritlm/eval/stsp/predict_stsp.py --eval_manifest_path data/examples/ref.jsonl --eval --write_pred ./pred_s_s.jsonl --input_output speech_speech

"""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Union

import torch
import torch.distributed as dist
import torchaudio
from ..eval_stsp import eval
from ..load_data import SpeechData, TextData
from ..stsp.few_shot_prompt import build_few_shot_prompt
from ..stsp.sentiment_classifiers import (
    get_text_sentiment_prediction,
    load_sentiment_classifier,
)
from ..stsp.stsp_constants import STSP_DATA_ROOT, STSP_MODEL_ROOT
from ..stsp.utils import (
    ExpressoEmotionClassifier,
    load_emotion_classifier,
    wav2emotion_and_sentiment,
)
from ...model.spiritlm_model import (
    ContentType,
    GenerationInput,
    InterleavedOutputs,
    OutputModality,
    Spiritlm,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GenerationConfig, set_seed

SPEECH_CLASSIFIER = STSP_MODEL_ROOT / "speech_classifier"
TEXT_CLASSIFIER = STSP_MODEL_ROOT / "text_classifier"

NB_RETRIES = 3


def get_eval_classifier(args):
    if args.input_output.endswith("speech"):
        return load_emotion_classifier(str(SPEECH_CLASSIFIER))
    elif args.input_output.endswith("text"):
        return load_sentiment_classifier(str(TEXT_CLASSIFIER))
    else:
        raise (Exception(f"{args.input_output} not supported"))


def get_sentiment(
    input_output,
    generation,
    classifer: Union[AutoModelForSequenceClassification, ExpressoEmotionClassifier],
):
    if input_output.endswith("speech"):
        _, pred_sentiment = wav2emotion_and_sentiment(generation, classifer)
    elif input_output.endswith("text"):
        _, pred_sentiment = get_text_sentiment_prediction(generation, classifer)
    return pred_sentiment


def write_jsonl(dir: str, predictions: dict):
    Path(dir).parent.mkdir(exist_ok=True, parents=True)
    with open(dir, "w") as f:
        for id, result_dict in predictions.items():
            record = {"id": id, **result_dict}
            json_string = json.dumps(record)
            f.write(json_string + "\n")  # Add a newline to separate JSON objects
    print(f"{dir} written")


def write_wav(
    wav,
    save_dir: Path,
    sample_rate: int = 16_000,
) -> str:
    """Save wav under `save_dir` with a random name and return the full path."""
    save_dir.mkdir(exist_ok=True, parents=True)
    random_path = save_dir / (str(uuid.uuid4()) + ".wav")
    torchaudio.save(
        random_path, torch.from_numpy(wav).unsqueeze(0), sample_rate=sample_rate
    )
    return str(random_path)


def run(args):
    world_size = int(os.environ["WORLD_SIZE"])
    world_rank = int(os.environ["RANK"])
    print(
        f"Running distributed inference with world_size: {world_size}, world_rank: {world_rank}"
    )
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    set_seed(args.seed)
    spiritlm_model = Spiritlm(args.model)
    evaluation_classifier = get_eval_classifier(args)
    input_output = args.input_output
    eval_manifest_path = args.eval_manifest_path
    write_wav_output = args.write_wav_output

    if args.few_shot > 0:
        prompt = build_few_shot_prompt(
            spiritlm_model=spiritlm_model,
            input_output=args.input_output,
            n_shots=args.few_shot,
        )
    else:
        prompt = None

    # load
    if input_output.startswith("speech"):
        eval_dataset = SpeechData(eval_manifest_path, root_dir=STSP_DATA_ROOT)
    elif input_output.startswith("text"):
        eval_dataset = TextData(eval_manifest_path, root_dir=STSP_DATA_ROOT)

    sampler = DistributedSampler(dataset=eval_dataset)
    loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,  # large batch size is not supported yet
        sampler=sampler,
        num_workers=4,
    )
    predictions = {}
    if input_output.endswith("speech"):
        output_modality = OutputModality.SPEECH
        max_new_tokens = 300
    else:
        output_modality = OutputModality.TEXT
        max_new_tokens = 50
    for _, data in tqdm(
        enumerate(loader),
        desc=f"Predict {eval_manifest_path}",
        total=eval_dataset.__len__() // world_size,
    ):
        # retry the generation multiple times because sometime it does not generate hubert tokens
        for i in range(NB_RETRIES):
            try:
                out: InterleavedOutputs = spiritlm_model.generate(
                    output_modality=output_modality,
                    interleaved_inputs=[
                        GenerationInput(
                            content=(
                                data["wav"][0]
                                if input_output.startswith("speech")
                                else data["text"][0]
                            ),  # 0 because of batch size 1
                            content_type=(
                                ContentType.SPEECH
                                if input_output.startswith("speech")
                                else ContentType.TEXT
                            ),
                        )
                    ],
                    generation_config=GenerationConfig(
                        temperature=0.8,
                        top_p=0.95,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                    ),
                    prompt=prompt,
                )
            except Exception as e:
                print(f"Got an exception when generating: {e}")
                if i == NB_RETRIES - 1:
                    raise Exception(f"Failed to generate after {NB_RETRIES}")
            else:
                break
        assert len(out) == 1
        generated_output = out[0].content
        detected_sentiment = get_sentiment(
            input_output, generated_output, evaluation_classifier
        )
        if output_modality == OutputModality.TEXT:
            generation = generated_output
        elif write_wav_output and output_modality == OutputModality.SPEECH:
            generation = write_wav(generated_output, Path(write_wav_output))
        else:
            generation = None
        result_dict = {"pred": detected_sentiment}
        if generation is not None:
            result_dict["generation"] = generation
        predictions[str(data["id"][0])] = result_dict

    if args.eval:
        gathered_predictions = [None for _ in range(world_size)]
        dist.gather_object(
            predictions, gathered_predictions if world_rank == 0 else None, dst=0
        )
        if world_rank == 0:
            all_predictions = {k: v for d in gathered_predictions for k, v in d.items()}
            eval(
                eval_manifest_path,
                {k: v["pred"] for k, v in all_predictions.items()},
                info_data=f"{eval_manifest_path}, input-output {input_output}",
                label="sentiment",
            )

    if args.write_pred is not None and world_rank == 0:
        write_jsonl(args.write_pred, all_predictions)


def setup_env():
    os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_manifest_path",  # data/examples/ref.jsonl
        type=str,
        help="Path to reference record",
        required=True,
    )

    parser.add_argument(
        "--data_root_dir",  # data/stsp_data
        type=str,
        help=f"Path to root data folder, default to {str(STSP_DATA_ROOT)}",
        default=str(STSP_DATA_ROOT),
        required=False,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="spirit-lm-expressive-7b",
        help="Model name (spirit-lm-base-7b or spirit-lm-expressive-7b) or path to model",
        required=False,
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="Number of few shot examples, 3/6/9",
        required=False,
    )
    parser.add_argument(
        "--input_output",
        type=str,
        default="speech_speech",
        help="speech_speech speech_text text_speech text_text",
        required=False,
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="emotion",
        required=False,
    )
    parser.add_argument(
        "--write_pred",
        type=str,
        default=None,
        help="Path to save the predictions output",
        required=False,
    )
    parser.add_argument(
        "--write_wav_output",
        type=str,
        default=None,
        help="Path to save the generated audio if the output is speech",
        required=False,
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )

    args = parser.parse_args()
    setup_env()
    run(args)
