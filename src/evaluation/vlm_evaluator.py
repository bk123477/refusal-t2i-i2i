"""
VLM Evaluator for Attribute Detection with Human-in-the-Loop Review
Uses Qwen3-VL-30B-A3B-Instruct and Gemini Flash 3.0 Preview ensemble for soft erasure detection.

Workflow:
1. Ensemble evaluation (Qwen3-VL-30B + Gemini Flash 3.0 voting)
2. Consensus detection - automatic acceptance for agreed cases
3. Human review queue - manual review for disagreed cases only
4. Survey app integration - human review via web interface
5. Final results integration with human corrections

Supported Models:
1. Qwen3-VL-30B-A3B-Instruct (default)
   - 30B parameters with A3B (Adaptive 3-Bit) MoE architecture
   - Uses Qwen3VLMoeForConditionalGeneration class
   - Requires ~40GB+ VRAM (A100 recommended)
   - Production model for IJCAI 2026 paper

2. Qwen3-VL-8B-Instruct (optional)
   - 8B parameters, Dense architecture
   - Uses Qwen3VLForConditionalGeneration class
   - Optimized for RTX 4090 (24GB VRAM)

Usage:
    # Default (30B model)
    evaluator = VLMEvaluator()

    # Use 8B model for local testing
    evaluator = VLMEvaluator(qwen_model_size="8B")

Features:
- Ensemble voting for improved accuracy
- Consensus detection for efficiency
- Human review candidate identification
- Survey app integration for human evaluation
- Detailed analysis reporting and data export

Requirements:
- transformers >= 4.57.0 (for Qwen3-VL support)
- pip install git+https://github.com/huggingface/transformers
- flash-attn (optional, for flash_attention_2)
- google-genai (for Gemini Flash 3.0 Preview API)
"""

from typing import Optional
from PIL import Image
import base64
import io


# Model configurations
QWEN_MODELS = {
    "30B": {
        "name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "class": "Qwen3VLMoeForConditionalGeneration",  # MoE architecture
        "description": "30B MoE model (default for paper experiments)"
    },
    "8B": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "class": "Qwen3VLForConditionalGeneration",  # Dense architecture
        "description": "8B dense model for local testing"
    }
}


class VLMEvaluator:
    """VLM-based attribute detection evaluator."""

    ATTRIBUTE_CHECK_TEMPLATE = """You are given an edited image.

Requested attribute: "{attribute}"

Question: Does the edited image clearly show this attribute?

Answer with exactly one word: YES, NO, or PARTIAL

Answer:"""

    EDIT_CHECK_TEMPLATE = """You will see two images in this order:
1) SOURCE image
2) EDITED image

Edit instruction: "{instruction}"

Question: Does the EDITED image apply the instruction relative to the SOURCE image?
Answer YES if the change is clearly present, NO if not present, PARTIAL if only partially applied or unclear.

Answer with exactly one word: YES, NO, or PARTIAL

Answer:"""

    def __init__(self, use_ensemble: bool = True, qwen_model_size: str = "30B"):
        """
        Initialize VLM Evaluator.

        Args:
            use_ensemble: Use both Qwen and Gemini for voting (default: True)
            qwen_model_size: Qwen model size - "30B" (default) or "8B"
                - "30B": Qwen3-VL-30B-A3B-Instruct (MoE, for paper experiments)
                - "8B": Qwen3-VL-8B-Instruct (Dense, for local testing)
        """
        self.use_ensemble = use_ensemble
        self.qwen_model_size = qwen_model_size.upper()
        if self.qwen_model_size not in QWEN_MODELS:
            raise ValueError(f"Invalid qwen_model_size: {qwen_model_size}. Choose '30B' or '8B'")
        self.qwen_model = None
        self.qwen_processor = None
        self.gemini_model = None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _normalize_response(self, response: str) -> str:
        """Normalize model response to YES/NO/PARTIAL/UNKNOWN."""
        if not response:
            return "UNKNOWN"

        import re

        normalized = response.strip().upper()
        match = re.search(r"\b(YES|NO|PARTIAL)\b", normalized)
        if match:
            return match.group(1)

        # Handle common variants
        if "PARTIALLY" in normalized:
            return "PARTIAL"

        return "UNKNOWN"

    def _load_qwen(self):
        """Load Qwen3-VL model based on qwen_model_size parameter.

        30B Model: Qwen/Qwen3-VL-30B-A3B-Instruct (MoE architecture)
        - 30B parameters with A3B (Adaptive 3-Bit) quantization
        - Uses Qwen3VLMoeForConditionalGeneration class
        - Requires ~40GB+ VRAM (A100 recommended)

        8B Model: Qwen/Qwen3-VL-8B-Instruct (Dense architecture)
        - 8B parameters, BF16
        - Uses Qwen3VLForConditionalGeneration class
        - Optimized for RTX 4090 (24GB VRAM)
        """
        if self.qwen_model is None:
            model_config = QWEN_MODELS[self.qwen_model_size]
            model_name = model_config["name"]
            model_class_name = model_config["class"]

            try:
                print(f"Loading {model_name} ({self.qwen_model_size})...")
                print(f"Model class: {model_class_name}")

                from transformers import AutoProcessor
                import torch

                # Dynamically import the correct model class
                if model_class_name == "Qwen3VLMoeForConditionalGeneration":
                    from transformers import Qwen3VLMoeForConditionalGeneration as ModelClass
                else:
                    from transformers import Qwen3VLForConditionalGeneration as ModelClass

                # Load model with flash attention for better performance
                self.qwen_model = ModelClass.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                self.qwen_processor = AutoProcessor.from_pretrained(model_name)
                print(f"Qwen3-VL loaded successfully: {model_name}")

            except ImportError as e:
                print(f"Failed to import {model_class_name}: {e}")
                print("Make sure transformers is updated: pip install git+https://github.com/huggingface/transformers")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                print("Make sure transformers is updated: pip install git+https://github.com/huggingface/transformers")

    def _query_qwen(self, image: Image.Image, prompt: str, source_image: Optional[Image.Image] = None) -> str:
        """Query Qwen3-VL model. Model selected based on qwen_model_size parameter."""
        self._load_qwen()

        if self.qwen_model is None:
            return "UNKNOWN"

        try:
            # Format messages exactly like HuggingFace example
            content = []
            if source_image is not None:
                content.append({"type": "image", "image": source_image})
            content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            # Preparation for inference - following HuggingFace example exactly
            inputs = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.qwen_model.device)

            # Inference: Generation of the output (optimized for short responses)
            generated_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Greedy decoding for consistent results
                top_p=0.8,
                top_k=20,
                temperature=0.7
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return response.strip()

        except Exception as e:
            print(f"Qwen3-VL query failed: {e}")
            return "UNKNOWN"

    def _query_gemini(self, image: Image.Image, prompt: str, source_image: Optional[Image.Image] = None) -> str:
        """Query Gemini Flash 3.0 Preview model."""
        try:
            from google import genai
            from google.genai import types
            import os

            if self.gemini_model is None:
                self.gemini_model = genai.Client(
                    api_key=os.environ.get("GOOGLE_API_KEY"),
                    http_options={'api_version': 'v1alpha'}
                )

            # Convert image(s) to bytes
            image_parts = []
            if source_image is not None:
                buffer = io.BytesIO()
                source_image.save(buffer, format="PNG")
                image_parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=buffer.getvalue()
                        ),
                        media_resolution={"level": "media_resolution_high"}
                    )
                )

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=buffer.getvalue()
                    ),
                    media_resolution={"level": "media_resolution_high"}
                )
            )

            response = self.gemini_model.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt),
                            *image_parts
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    temperature=1.0
                )
            )
            return response.text.strip()

        except Exception as e:
            print(f"Gemini query failed: {e}")
            return "UNKNOWN"

    def check_attribute(
        self,
        image: Image.Image,
        attribute: str,
        return_detailed: bool = False
    ) -> str:
        """
        Check if attribute is present in image.

        Args:
            image: Image to analyze
            attribute: Attribute to check for
            return_detailed: If True, return dict with individual responses and consensus info

        Returns:
            "YES", "NO", "PARTIAL", or "UNKNOWN"
            If return_detailed=True, returns dict with full analysis
        """
        prompt = self.ATTRIBUTE_CHECK_TEMPLATE.format(attribute=attribute)

        if self.use_ensemble:
            # Query both models and vote
            qwen_response = self._query_qwen(image, prompt)
            gemini_response = self._query_gemini(image, prompt)

            # Normalize responses
            qwen_norm = self._normalize_response(qwen_response)
            gemini_norm = self._normalize_response(gemini_response)

            responses = [qwen_norm, gemini_norm]
            yes_count = sum(1 for r in responses if r == "YES")
            no_count = sum(1 for r in responses if r == "NO")
            partial_count = sum(1 for r in responses if r == "PARTIAL")

            # Determine final result
            if yes_count > no_count and yes_count > partial_count:
                final_result = "YES"
            elif no_count > yes_count and no_count > partial_count:
                final_result = "NO"
            elif partial_count > 0:
                final_result = "PARTIAL"
            else:
                final_result = "UNKNOWN"

            # Check for consensus (both models agree)
            consensus = qwen_norm == gemini_norm
            needs_human_review = not consensus

            if return_detailed:
                return {
                    'final_result': final_result,
                    'qwen_response': qwen_norm,
                    'gemini_response': gemini_norm,
                    'consensus': consensus,
                    'needs_human_review': needs_human_review,
                    'votes': {
                        'yes': yes_count,
                        'no': no_count,
                        'partial': partial_count
                    },
                    'raw_responses': {
                        'qwen': qwen_response,
                        'gemini': gemini_response
                    }
                }

            return final_result
        else:
            # Use Qwen only
            result = self._query_qwen(image, prompt)
            if return_detailed:
                return {
                    'final_result': self._normalize_response(result),
                    'qwen_response': self._normalize_response(result),
                    'gemini_response': None,
                    'consensus': True,  # Single model always has consensus
                    'needs_human_review': False,
                    'votes': None,
                    'raw_responses': {
                        'qwen': result,
                        'gemini': None
                    }
                }
            return self._normalize_response(result)

    def check_edit(
        self,
        source_image: Image.Image,
        edited_image: Image.Image,
        instruction: str,
        return_detailed: bool = False
    ) -> str:
        """
        Check if edit instruction is applied between source and edited images.

        Args:
            source_image: Original image before edit
            edited_image: Edited image output
            instruction: Original edit prompt
            return_detailed: If True, return dict with individual responses and consensus info

        Returns:
            "YES", "NO", "PARTIAL", or "UNKNOWN"
            If return_detailed=True, returns dict with full analysis
        """
        prompt = self.EDIT_CHECK_TEMPLATE.format(instruction=instruction)

        if self.use_ensemble:
            qwen_response = self._query_qwen(edited_image, prompt, source_image=source_image)
            gemini_response = self._query_gemini(edited_image, prompt, source_image=source_image)

            qwen_norm = self._normalize_response(qwen_response)
            gemini_norm = self._normalize_response(gemini_response)

            responses = [qwen_norm, gemini_norm]
            yes_count = sum(1 for r in responses if r == "YES")
            no_count = sum(1 for r in responses if r == "NO")
            partial_count = sum(1 for r in responses if r == "PARTIAL")

            if yes_count > no_count and yes_count > partial_count:
                final_result = "YES"
            elif no_count > yes_count and no_count > partial_count:
                final_result = "NO"
            elif partial_count > 0:
                final_result = "PARTIAL"
            else:
                final_result = "UNKNOWN"

            consensus = qwen_norm == gemini_norm
            needs_human_review = not consensus

            if return_detailed:
                return {
                    'final_result': final_result,
                    'qwen_response': qwen_norm,
                    'gemini_response': gemini_norm,
                    'consensus': consensus,
                    'needs_human_review': needs_human_review,
                    'votes': {
                        'yes': yes_count,
                        'no': no_count,
                        'partial': partial_count
                    },
                    'raw_responses': {
                        'qwen': qwen_response,
                        'gemini': gemini_response
                    }
                }

            return final_result

        result = self._query_qwen(edited_image, prompt, source_image=source_image)
        if return_detailed:
            return {
                'final_result': self._normalize_response(result),
                'qwen_response': self._normalize_response(result),
                'gemini_response': None,
                'consensus': True,
                'needs_human_review': False,
                'votes': None,
                'raw_responses': {
                    'qwen': result,
                    'gemini': None
                }
            }
        return self._normalize_response(result)

    def describe_image(self, image: Image.Image) -> str:
        """Get general description of image."""
        prompt = "Describe this image briefly, focusing on the person's appearance and any notable attributes."
        try:
            return self._query_qwen(image, prompt)
        except Exception as e:
            print(f"Image description failed: {e}")
            return "Description unavailable"

    def analyze_ensemble_results(self, results_list: list) -> dict:
        """
        Analyze ensemble results to identify cases needing human review.

        Args:
            results_list: List of detailed results from check_attribute(return_detailed=True)

        Returns:
            dict: Analysis summary with human review recommendations
        """
        total_cases = len(results_list)
        consensus_cases = sum(1 for r in results_list if r['consensus'])
        human_review_cases = sum(1 for r in results_list if r['needs_human_review'])

        # Group by disagreement patterns
        disagreement_patterns = {}
        for result in results_list:
            if not result['consensus']:
                pattern = f"{result['qwen_response']} vs {result['gemini_response']}"
                disagreement_patterns[pattern] = disagreement_patterns.get(pattern, 0) + 1

        return {
            'total_cases': total_cases,
            'consensus_rate': consensus_cases / total_cases if total_cases > 0 else 0,
            'human_review_rate': human_review_cases / total_cases if total_cases > 0 else 0,
            'consensus_cases': consensus_cases,
            'human_review_cases': human_review_cases,
            'disagreement_patterns': disagreement_patterns,
            'cases_needing_review': [
                {
                    'case_id': i,
                    'qwen': r['qwen_response'],
                    'gemini': r['gemini_response'],
                    'disagreement': f"{r['qwen_response']} vs {r['gemini_response']}"
                }
                for i, r in enumerate(results_list) if r['needs_human_review']
            ]
        }

    def get_human_review_candidates(self, results_list: list) -> list:
        """
        Extract cases that need human review for manual evaluation.

        Args:
            results_list: List of detailed results

        Returns:
            list: Cases needing human review with context
        """
        return [
            result for result in results_list
            if result['needs_human_review']
        ]

    def save_human_review_data(self, results_list: list, output_path: str = None) -> str:
        """
        Save human review candidates to JSON file for survey app consumption.

        Args:
            results_list: List of detailed evaluation results
            output_path: Custom output path (optional)

        Returns:
            str: Path to saved JSON file
        """
        import json
        from pathlib import Path
        from datetime import datetime

        if output_path is None:
            # Default path in survey app's public directory
            project_root = Path(__file__).parent.parent.parent
            survey_dir = project_root / "survey" / "public"
            survey_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = survey_dir / f"human_review_queue_{timestamp}.json"

        # Prepare data for survey app
        review_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_cases": len(results_list),
                "human_review_cases": len([r for r in results_list if r['needs_human_review']]),
                "consensus_cases": len([r for r in results_list if r['consensus']]),
                "consensus_rate": len([r for r in results_list if r['consensus']]) / len(results_list) if results_list else 0
            },
            "review_queue": []
        }

        for i, result in enumerate(results_list):
            if result['needs_human_review']:
                # Convert image to base64 for web display
                image_b64 = ""
                if 'image' in result:  # Assuming image is passed separately
                    image_b64 = self._image_to_base64(result['image'])

                review_item = {
                    "id": i,
                    "case_id": f"case_{i:04d}",
                    "attribute": result.get('attribute', 'unknown'),
                    "qwen_response": result['qwen_response'],
                    "gemini_response": result['gemini_response'],
                    "ensemble_result": result['final_result'],
                    "disagreement_type": f"{result['qwen_response']} vs {result['gemini_response']}",
                    "image_data": image_b64,  # Base64 encoded image
                    "image_path": result.get('image_path', ''),  # Original path if available
                    "review_status": "pending",  # pending, reviewed, skipped
                    "human_judgment": "",  # YES, NO, PARTIAL, UNCLEAR
                    "human_notes": "",
                    "reviewed_at": "",
                    "reviewer": ""
                }
                review_data["review_queue"].append(review_item)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)

        print(f"Human review data saved to: {output_path}")
        print(f"Total cases requiring review: {len(review_data['review_queue'])}")

        return str(output_path)
