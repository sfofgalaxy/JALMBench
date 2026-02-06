import argparse
from InitModel import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Audio language model process tool')
    parser.add_argument('--audio', type=str, help='input audio path')
    parser.add_argument('--text', type=str, help='input text (will be converted to audio with tts_model)')
    parser.add_argument('--prompt', type=str, help='audio\'s specific text prompt (optional)')
    parser.add_argument('--output_text', type=str, required=True, help='output text file path')
    parser.add_argument('--output_audio', type=str, default=None, help='output audio file path')
    parser.add_argument('--model', type=str, default='qwen')
    parser.add_argument('--tts_model', type=str, default='google_api (only for text input)')
    return parser.parse_args()

def main():
    args = parse_args()
 
    if args.text and args.audio:
        raise ValueError('Cannot process both audio and text, if you mean prompt, please use --prompt')
    
    model = get_model(args.model.lower())

    if args.text:
        result = model.process_text(args.text, tts_model=args.tts_model, prompt=args.prompt, output_audio=args.output_audio)
    elif args.audio:
        result = model.process_audio(args.audio, prompt=args.prompt, output_audio=args.output_audio)
    # only use text modality (prompt)
    else:
        result = model.process_audio(None, prompt=args.prompt, output_audio=args.output_audio)
    
    print(result)
    with open(args.output_text, 'w', encoding='utf-8') as f:
        f.write(result)
    print(result, f'\nSave result to: {args.output_text} and {args.output_audio} if output audio is supported and not none')

if __name__ == '__main__':
    main()