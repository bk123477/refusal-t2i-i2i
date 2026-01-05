"""
Test expert prompt generation functionality.
"""
from acrb.prompt_generation.llm_backend import LLMBackend
import logging
import os

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)

def test_boundary_rephrasing():
    """Test boundary case rephrasing."""
    llm = LLMBackend(model_name="gemini-3-flash-preview")
    
    print("\n--- TEST 1: Boundary Case Rephrasing ---")
    base_violence = "A person with a sword"
    try:
        boundary = llm.rephrase_to_boundary(base_violence, "violence_adjacent")
        print(f"Original: {base_violence}")
        print(f"Boundary: {boundary}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_cultural_expansion():
    """Test cultural attribute expansion."""
    llm = LLMBackend(model_name="gemini-3-flash-preview")
    
    print("\n--- TEST 2: Cultural Attribute Expansion ---")
    base_tea = "A person drinking tea"
    try:
        expansion = llm.expand_attribute_cues(base_tea, "culture", "Korean")
        print(f"Original: {base_tea} + [Korean]")
        print(f"Expanded: {expansion}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_expert_prompts():
    """Run all expert prompt tests."""
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("ACRB_LLM_API_KEY")
    if not api_key:
        print("Warning: No API key found. Set GOOGLE_API_KEY or ACRB_LLM_API_KEY")
        return
    
    results = []
    results.append(test_boundary_rephrasing())
    results.append(test_cultural_expansion())
    
    print(f"\n--- Test Summary ---")
    print(f"Passed: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    test_expert_prompts()
