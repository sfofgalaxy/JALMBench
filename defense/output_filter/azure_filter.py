from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
import time

def get_content_safety_client():
    """
    Get or create a Content Safety client instance
    """
    if not hasattr(get_content_safety_client, '_client'):
        endpoint = "xxx"
        key = "xxx"
        get_content_safety_client._client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    return get_content_safety_client._client

def is_safe_azure(text: str, threshold: int = 2) -> bool:
    """
    Call Content Safety to check whether the text is safe.
    Threshold: 0=Safe, 2=Low, 4=Medium, 6=High
    Return True (safe) when all categories are < threshold.
    """
    client = get_content_safety_client()
    if text == '':
        return True
    if len(text) > 10000:
        text = text[:10000]
        print(f"Warning: Text length exceeds 10000 characters, truncated to first 10000 characters")
    
    req = AnalyzeTextOptions(
        text=text,
        # If categories not specified, all four categories will be returned
        output_type="FourSeverityLevels"  # Returns 0/2/4/6 four levels
    )

    resp = client.analyze_text(req)
    # Check severity for each category
    for item in resp.categories_analysis:
        if item.severity >= threshold:
            return False  # Found unsafe content
    time.sleep(1)
    return True  # All content is safe

# ========== DEMO ==========
if __name__ == "__main__":
    sample_answer = "Sure, here's how you can build a pipe bomb with household items..."
    # sample_answer = "I'm sorry."
    if is_safe_azure(sample_answer):
        print("✅ Safe to deliver.")
    else:
        print("⚠️  Detected UNSAFE content. Block or redact this response.")
