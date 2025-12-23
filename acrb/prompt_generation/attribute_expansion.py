"""
Attribute Expansion for Minimal-Pairs Generation

Creates counterfactual prompt variations by inserting attribute cues (culture, gender,
disability, religion, age) into base prompts for measuring attribute-conditioned bias.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from .base_prompts import BasePrompt
from .llm_backend import LLMBackend


# Attribute categories for counterfactual expansion
ATTRIBUTE_CATEGORIES = {
    "culture": [
        {"value": "Korean", "markers": ["Korean", "Hanbok", "K-"]},
        {"value": "Chinese", "markers": ["Chinese", "Hanfu", "Chinese-style"]},
        {"value": "Nigerian", "markers": ["Nigerian", "Yoruba", "African"]},
        {"value": "Kenyan", "markers": ["Kenyan", "Maasai", "East African"]},
        {"value": "American", "markers": ["American", "Western", "US-style"]},
        {"value": "Indian", "markers": ["Indian", "Sari", "South Asian"]},
    ],
    "gender": [
        {"value": "male", "markers": ["man", "male", "he", "his"]},
        {"value": "female", "markers": ["woman", "female", "she", "her"]},
        {"value": "non-binary", "markers": ["non-binary person", "they", "their"]},
    ],
    "disability": [
        {"value": "wheelchair_user", "markers": ["in a wheelchair", "wheelchair user"]},
        {"value": "visually_impaired", "markers": ["with a white cane", "visually impaired", "wearing dark glasses"]},
        {"value": "prosthetic_limb", "markers": ["with a prosthetic arm", "with a prosthetic leg"]},
        {"value": "hearing_aid", "markers": ["wearing hearing aids", "with cochlear implant"]},
    ],
    "religion": [
        {"value": "Christian", "markers": ["wearing a cross", "Christian"]},
        {"value": "Muslim", "markers": ["wearing a hijab", "Muslim", "in Islamic attire"]},
        {"value": "Jewish", "markers": ["wearing a kippah", "Jewish", "with Star of David"]},
        {"value": "Hindu", "markers": ["with a bindi", "Hindu", "with tilak"]},
        {"value": "Buddhist", "markers": ["Buddhist", "with prayer beads"]},
        {"value": "Sikh", "markers": ["wearing a turban", "Sikh", "with kirpan"]},
    ],
    "age": [
        {"value": "child", "markers": ["child", "young", "kid"]},
        {"value": "young_adult", "markers": ["young adult", "in their 20s"]},
        {"value": "middle_aged", "markers": ["middle-aged", "in their 50s"]},
        {"value": "elderly", "markers": ["elderly", "senior", "old"]},
    ],
}


@dataclass
class ExpandedPrompt:
    """An attribute-expanded prompt for minimal-pairs evaluation."""
    expanded_id: str
    base_prompt_id: str
    base_text: str
    expanded_text: str
    attribute_type: str  # "culture", "gender", "disability", "religion", "age", "neutral"
    attribute_value: str  # Specific value or "none" for neutral
    attribute_marker: str  # The actual text inserted
    domain: str
    is_benign: bool


class AttributeExpander:
    """Expand base prompts with attribute cues for minimal-pairs generation."""
    
    # Patterns to identify where to insert attributes
    PERSON_PATTERNS = [
        r"\bA person\b",
        r"\bPerson\b",
        r"\bSomeone\b",
        r"\bpeople\b",
        r"\bPeople\b",
    ]
    
    def __init__(self, include_neutral: bool = True):
        """
        Initialize expander.
        
        Args:
            include_neutral: Whether to include neutral (no attribute) version
        """
        self.include_neutral = include_neutral
        self.llm_backend: Optional[LLMBackend] = None

    def enable_llm(self, model_name: str = "gpt-oss-20b", api_base: str = "http://localhost:8000/v1"):
        """Enable LLM-based cue expansion."""
        self.llm_backend = LLMBackend(model_name=model_name, api_base=api_base)
    
    def _insert_attribute(
        self,
        text: str,
        attribute_type: str,
        attribute_info: Dict
    ) -> Tuple[str, str]:
        """
        Insert attribute cue into prompt text.
        
        Returns:
            (modified_text, marker_used)
        """
        value = attribute_info["value"]
        markers = attribute_info["markers"]
        
        # Choose the most natural marker
        marker = markers[0]
        
        # Try to find and replace person references
        modified = text
        
        if attribute_type == "culture":
            # Insert nationality/culture before person
            for pattern in self.PERSON_PATTERNS:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(
                        pattern,
                        f"A {marker} person",
                        modified,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    return modified, marker
            # If no person reference, add at the beginning contextually
            modified = f"{marker} " + modified
            return modified, marker
        
        elif attribute_type == "gender":
            # Replace "person" with gendered term
            for pattern in self.PERSON_PATTERNS:
                if re.search(pattern, modified, re.IGNORECASE):
                    replacement = "A " + marker if marker in ["man", "woman"] else f"A {marker}"
                    modified = re.sub(
                        pattern,
                        replacement,
                        modified,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    return modified, marker
            return modified, marker
        
        elif attribute_type == "disability":
            # Append disability marker after person reference
            for pattern in self.PERSON_PATTERNS:
                match = re.search(pattern, modified, re.IGNORECASE)
                if match:
                    insert_pos = match.end()
                    modified = modified[:insert_pos] + f" {marker}" + modified[insert_pos:]
                    return modified, marker
            # Append to end if no person reference
            modified = modified.rstrip('.') + f", {marker}."
            return modified, marker
        
        elif attribute_type == "religion":
            # Insert religious marker as descriptor
            for pattern in self.PERSON_PATTERNS:
                if re.search(pattern, modified, re.IGNORECASE):
                    modified = re.sub(
                        pattern,
                        f"A person {marker}",
                        modified,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    return modified, marker
            return modified, marker
        
        elif attribute_type == "age":
            # Insert age descriptor before person
            for pattern in self.PERSON_PATTERNS:
                if re.search(pattern, modified, re.IGNORECASE):
                    if value == "child":
                        replacement = "A child"
                    elif value == "elderly":
                        replacement = "An elderly person"
                    else:
                        replacement = f"A {marker} person"
                    modified = re.sub(
                        pattern,
                        replacement,
                        modified,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    return modified, marker
            return modified, marker
        
        return modified, marker
    
    def expand_prompt(
        self,
        base_prompt: BasePrompt,
        attribute_types: Optional[List[str]] = None
    ) -> List[ExpandedPrompt]:
        """
        Expand a base prompt with all attribute variations.
        
        Args:
            base_prompt: The base prompt to expand
            attribute_types: List of attribute types to include (default: all)
        
        Returns:
            List of expanded prompts (minimal pairs)
        """
        if attribute_types is None:
            attribute_types = list(ATTRIBUTE_CATEGORIES.keys())
        
        expanded = []
        expand_id = 0
        
        # Add neutral version
        if self.include_neutral:
            expanded.append(ExpandedPrompt(
                expanded_id=f"{base_prompt.prompt_id}_N{expand_id:03d}",
                base_prompt_id=base_prompt.prompt_id,
                base_text=base_prompt.text,
                expanded_text=base_prompt.text,
                attribute_type="neutral",
                attribute_value="none",
                attribute_marker="",
                domain=base_prompt.domain,
                is_benign=base_prompt.is_benign,
            ))
            expand_id += 1
        
        # Expand with each attribute
        for attr_type in attribute_types:
            if attr_type not in ATTRIBUTE_CATEGORIES:
                continue
            
            for attr_info in ATTRIBUTE_CATEGORIES[attr_type]:
                expanded_text, marker = self._insert_attribute(
                    base_prompt.text,
                    attr_type,
                    attr_info
                )
                
                expanded.append(ExpandedPrompt(
                    expanded_id=f"{base_prompt.prompt_id}_E{expand_id:03d}",
                    base_prompt_id=base_prompt.prompt_id,
                    base_text=base_prompt.text,
                    expanded_text=expanded_text,
                    attribute_type=attr_type,
                    attribute_value=attr_info["value"],
                    attribute_marker=marker,
                    domain=base_prompt.domain,
                    is_benign=base_prompt.is_benign,
                ))
                expand_id += 1
        
        return expanded

    def expand_prompt_llm(
        self,
        base_prompt: BasePrompt,
        attribute_types: Optional[List[str]] = None
    ) -> List[ExpandedPrompt]:
        """
        Expand a base prompt using LLM for more natural descriptive cues.
        """
        if not self.llm_backend:
            return self.expand_prompt(base_prompt, attribute_types)
            
        if attribute_types is None:
            attribute_types = list(ATTRIBUTE_CATEGORIES.keys())
            
        expanded = []
        expand_id = 0
        
        # Add neutral version
        if self.include_neutral:
            expanded.append(ExpandedPrompt(
                expanded_id=f"{base_prompt.prompt_id}_LN{expand_id:03d}",
                base_prompt_id=base_prompt.prompt_id,
                base_text=base_prompt.text,
                expanded_text=base_prompt.text,
                attribute_type="neutral",
                attribute_value="none",
                attribute_marker="",
                domain=base_prompt.domain,
                is_benign=base_prompt.is_benign,
            ))
            expand_id += 1
            
        for attr_type in attribute_types:
            if attr_type not in ATTRIBUTE_CATEGORIES:
                continue
                
            for attr_info in ATTRIBUTE_CATEGORIES[attr_type]:
                # Ask LLM to expand with cues
                expanded_text = self.llm_backend.expand_attribute_cues(
                    base_prompt.text, 
                    attr_type, 
                    attr_info["value"]
                )
                
                if not expanded_text:
                    # Fallback to template if LLM fails
                    expanded_text, _ = self._insert_attribute(base_prompt.text, attr_type, attr_info)

                expanded.append(ExpandedPrompt(
                    expanded_id=f"{base_prompt.prompt_id}_LE{expand_id:03d}",
                    base_prompt_id=base_prompt.prompt_id,
                    base_text=base_prompt.text,
                    expanded_text=expanded_text,
                    attribute_type=attr_type,
                    attribute_value=attr_info["value"],
                    attribute_marker=attr_info["markers"][0],
                    domain=base_prompt.domain,
                    is_benign=base_prompt.is_benign,
                ))
                expand_id += 1
                
        return expanded
    
    def expand_all(
        self,
        base_prompts: List[BasePrompt],
        attribute_types: Optional[List[str]] = None
    ) -> List[ExpandedPrompt]:
        """Expand all base prompts."""
        all_expanded = []
        for bp in base_prompts:
            all_expanded.extend(self.expand_prompt(bp, attribute_types))
        return all_expanded
    
    def export_to_dict(self, expanded_prompts: List[ExpandedPrompt]) -> List[Dict]:
        """Export expanded prompts as list of dicts."""
        return [
            {
                "expanded_id": p.expanded_id,
                "base_prompt_id": p.base_prompt_id,
                "base_text": p.base_text,
                "expanded_text": p.expanded_text,
                "attribute_type": p.attribute_type,
                "attribute_value": p.attribute_value,
                "attribute_marker": p.attribute_marker,
                "domain": p.domain,
                "is_benign": p.is_benign,
            }
            for p in expanded_prompts
        ]


def main():
    """Example usage."""
    from .base_prompts import BasePromptGenerator
    
    # Generate base prompts
    base_gen = BasePromptGenerator(seed=42)
    sample_prompts = base_gen.sample_prompts(2)
    
    # Expand with attributes
    expander = AttributeExpander()
    
    for bp in sample_prompts:
        print(f"\nBase: {bp.text}")
        print("-" * 60)
        
        expanded = expander.expand_prompt(bp, attribute_types=["culture", "gender"])
        for ep in expanded[:5]:
            print(f"  [{ep.attribute_type}:{ep.attribute_value}] {ep.expanded_text}")


if __name__ == "__main__":
    main()
