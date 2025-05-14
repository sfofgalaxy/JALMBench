from openai import AzureOpenAI
import json
import re

# evaluate single json file
def evaluate_from_json(json_file_path) -> list:
    client = init_client()
    if type(json_file_path) == str:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
    else:
        responses = json_file_path

    for response in responses:
        if "score" in response and response["score"] != 0:
            continue
        if response["response"] == "" or response["response"].startswith("[[Error]]"):
            response["evaluation"] = "The response is empty or contains an error. #thescore: 1"
            response["score"] = 1
        else:
            evaluation = evaluate_single_response(client, response['original_query'], response['response'])
            response["evaluation"] = evaluation
            response["score"] = extract_score(evaluation)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)
    
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
    deployment_name="gpt-4o"
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
