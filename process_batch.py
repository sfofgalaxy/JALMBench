from InitModel import get_model
import argparse
import os
import json
from pathlib import Path
import time
def parse_args():
    parser = argparse.ArgumentParser(description='Audio language model process tool')
    parser.add_argument('--type', type=str, required=True, help='batch text, audio or prompt only query')
    parser.add_argument('--input_path', type=str, required=True, help='input audio dir path or input json file path')
    parser.add_argument('--cache_audio_dir', type=str, help='input audio dir when using json (cache)')
    parser.add_argument('--prompt', type=str, help='audio\'s specific text prompt (optional)')
    parser.add_argument('--output_dir', type=str, default="./output", help='output audio file path')
    parser.add_argument('--model', type=str, default='qwen')
    parser.add_argument('--continue_index', type=str, help='start index for processing text or audio (optional)')
    parser.add_argument('--target_lang', type=str, default='en-US', help='target audio language for processing text')
    return parser.parse_args()

def process_batch_audio(audios : list, prompt: str, output_dir: Path, model, output_name: str, continue_index: str):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    start_index = 0
    if continue_index is not None:
        for i, audio in enumerate(audios):
            if os.path.basename(audio) == continue_index:
                start_index = i
                break
    print(f"Processing {len(audios)} audios: {audios}")
    time_start = time.time()
    for audio in audios[start_index:]:
        output_audio = output_dir / (audio.stem + '.wav')
        try: 
            output = model.process_audio(str(audio), prompt=prompt, output_audio_path=output_audio)
            results.append({"id": int(audio.stem), "response": output })
        except Exception as e:
            raise
        with open(output_dir / output_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    average_time = (time_end - time_start)/len(audios[start_index:])
    print(f"Time taken per audio: {average_time} seconds")
    return average_time

def process_batch_text(texts : list, prompt: str, output_dir: Path, model, output_name: str, continue_index: str, cache_audio_dir, target_lang="en-US"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / output_name
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
    start_index = 0
    if continue_index is not None:
        for i, audio in enumerate(texts):
            if os.path.basename(audio) == continue_index:
                start_index = i
                break

    start_time = time.time()
    for text in texts[start_index:]:
        # output_audio = output_dir / (str(text['id']) + '.wav')
        # if audio_path is not none and file already exsisted, no need to invoke tts in google
        if "response" in text and not text["response"].startswith("[[Error]]"):
            continue
        if cache_audio_dir != None:
            audio_path = os.path.join(cache_audio_dir, (str(text['id']) + '.wav'))
        else:
            audio_path = None
        try:
            output = model.process_text(text['query'], prompt=prompt, cache_path=audio_path, target_lang=target_lang)
            text["response"] = output
        except Exception as e:
            text["response"] = f"[[Error]]: {e}"
            raise
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=4)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    average_time = (end_time - start_time)/len(texts[start_index:])
    print(f"Time taken per audio: {average_time} seconds")
    return average_time

def process_batch_prompt(prompts, output_dir, model, output_name, continue_index):
    start_index = 0
    if continue_index is not None:
        for i, audio in enumerate(prompts):
            if os.path.basename(audio) == continue_index:
                start_index = i
                break
    
    for prompt in prompts[start_index:]:
        output_audio = output_dir / (str(prompt['id']) + '.wav')
        try:
            output = model.process_audio(None, prompt=prompt['text'], output_audio=output_audio)
            prompt["response"] = output
        except Exception as e:
            prompt["response"] = f"[[Error]]: {e}"
            # raise
        with open(output_dir / output_name, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)
    return

def main():
    args = parse_args()
    model_name = args.model.lower()
    model = get_model(model_name)
    input_path = args.input_path
    output_name = f"{model_name}.json"
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # json content must be the format of [{'id': str, 'query': str, ... other fields}, ...]
    if args.type == 'audio':
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path must be a directory for audio type: {input_path}")
        audios = os.listdir(input_path)
        input_dir = Path(input_path)
        audios = [input_dir / x for x in audios if x.endswith('.wav') or x.endswith('.mp3')]
        audios = sorted(audios, key=lambda x: int(x.stem))
        process_batch_audio(audios, args.prompt, output_dir, model, output_name, args.continue_index)
    elif args.type == 'text':
        if not os.path.isdir(input_path) and not input_path.endswith('.json'):
            raise ValueError(f"Input path must be a directory or json file for text type: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        process_batch_text(texts, args.prompt, output_dir, model, output_name, args.continue_index, args.cache_audio_dir, args.target_lang)
    elif args.type == 'prompt':
        if not os.path.isdir(input_path) and not input_path.endswith('.json'):
            raise ValueError(f"Input path must be a directory or json file for text type: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        process_batch_prompt(prompts, output_dir, model, output_name, args.continue_index)
    else:
        raise ValueError(f"Invalid input type: {args.type}")

if __name__ == '__main__':
    main()