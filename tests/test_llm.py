"""
Test LLM backend connectivity and API access.
"""
import requests
import os
from acrb.prompt_generation.llm_backend import LLMBackend

def test_llm_connection():
    """Test LLM backend connection using environment variables."""
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("ACRB_LLM_API_KEY")
    
    if not api_key:
        print("Warning: No API key found. Set GOOGLE_API_KEY or ACRB_LLM_API_KEY environment variable.")
        return False
    
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    payload = {
        "model": "gemini-3-flash-preview",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you Gemini 3 Flash?"}
        ],
        "max_tokens": 100
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"Testing URL: {url}")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Connection successful")
            return True
        else:
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_llm_backend():
    """Test ACRB LLM backend wrapper."""
    try:
        llm = LLMBackend(model_name="gemini-3-flash-preview")
        test_prompt = "A person drinking tea"
        result = llm.expand_attribute_cues(test_prompt, "culture", "Korean")
        print(f"Test prompt: {test_prompt}")
        print(f"Expanded: {result}")
        return True
    except Exception as e:
        print(f"LLM backend test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LLM connection...")
    test_llm_connection()
    print("\nTesting LLM backend wrapper...")
    test_llm_backend()
