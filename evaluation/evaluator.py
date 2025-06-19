from openai import AzureOpenAI
import json
import re
import argparse
from pathlib import Path

def evaluate_from_json(jsonl_file_path) -> list:
    client = init_client()
    # Read jsonl file
    responses = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line))
    
    # Evaluate each response
    for response in responses:
        if "score" in response and response["score"] != 0:
            continue
        if response["response"] == "" or response["response"].startswith("[[Error]]"):
            response["evaluation"] = "The response is empty or contains an error. #thescore: 1"
            response["score"] = 1
        else:
            evaluation = evaluate_single_response(client, response['original_text'], response['response'])
            response["evaluation"] = evaluation
            response["score"] = extract_score(evaluation)
    
    # Save results to new jsonl file
    output_path = f"result-{Path(jsonl_file_path).name}"
    with open(output_path, 'w', encoding='utf-8') as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')
    
    return responses

def evaluate_single_response(client, prompt, response) -> str:
    evaluation_path = "./evaluation/eval_prompt.md"
    with open(evaluation_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    if response == None or response == "" or response.startswith("[[Error]]"):
        return "The response is empty or contains an error. #thescore: 1"
    content = [
        {"role": "user", "content": system_prompt.format(prompt, response)}
    ]
    deployment_name="xxx"
    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=content,
            temperature=0,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[[Error]] {e}"

def init_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client"""
    api_key="xxx"
    endpoint="xxx"
    api_version="2025-01-01-preview"
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key, 
        api_version=api_version
    )

def extract_score(evaluation: str) -> int:
    if evaluation == None:
        return 0
    """Extract the score from the evaluation string using regex"""
    match = re.search(r"#thescore:\s*(\d+)", evaluation)
    if match:
        return int(match.group(1))
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model responses')
    parser.add_argument('--file', type=str, required=True, help='Path to the jsonl file containing model responses')
    args = parser.parse_args()
    evaluate_from_json(args.file)
