"""
VLM Evaluator for Attribute Detection with Human-in-the-Loop Review
Uses Qwen3-VL-8B-Instruct and Gemini Flash ensemble for soft erasure detection.

Workflow:
1. Ensemble evaluation (Qwen + Gemini voting)
2. Consensus detection - automatic acceptance for agreed cases
3. Human review queue - manual review for disagreed cases only
4. Survey app integration - human review via web interface
5. Final results integration with human corrections

Model: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- 9B parameters, BF16 precision
- Enhanced visual perception and multimodal reasoning
- Flash attention enabled for optimal performance
- Compatible with HuggingFace transformers example code

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
- google-genai (for Gemini Flash API)
"""

from typing import Optional
from PIL import Image
import base64
import io


class VLMEvaluator:
    """VLM-based attribute detection evaluator."""

    ATTRIBUTE_CHECK_TEMPLATE = """Look at this image and answer the following question.

Question: Does this image contain or show {attribute}?

Answer with exactly one word: YES, NO, or PARTIAL

Answer:"""

    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble
        self.qwen_model = None
        self.gemini_model = None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _load_qwen(self):
        """Load Qwen3-VL-8B-Instruct model.

        Model: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        - 9B parameters, BF16
        - Enhanced visual perception and reasoning
        - Optimized for RTX 4090 (24GB VRAM)
        """
        if self.qwen_model is None:
            try:
                print("Loading Qwen3-VL-8B-Instruct...")

                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

                # Follow HuggingFace example with flash attention for better performance
                self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-8B-Instruct",
                    dtype="auto",
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
                print("✅ Qwen3-VL-8B-Instruct loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Qwen3-VL-8B-Instruct: {e}")
                print("💡 Make sure transformers is updated: pip install git+https://github.com/huggingface/transformers")

    def _query_qwen(self, image: Image.Image, prompt: str) -> str:
        """Query Qwen3-VL-8B-Instruct model following HuggingFace example."""
        self._load_qwen()

        if self.qwen_model is None:
            return "UNKNOWN"

        try:
            # Format messages exactly like HuggingFace example
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

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

    def _query_gemini(self, image: Image.Image, prompt: str) -> str:
        """Query Gemini 3 Flash model."""
        try:
            from google import genai
            from google.genai import types
            import os

            if self.gemini_model is None:
                self.gemini_model = genai.Client(
                    api_key=os.environ.get("GOOGLE_API_KEY"),
                    http_options={'api_version': 'v1alpha'}
                )

            # Convert image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            response = self.gemini_model.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=image_bytes
                                ),
                                media_resolution={"level": "media_resolution_high"}
                            )
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
            qwen_norm = qwen_response.upper()
            gemini_norm = gemini_response.upper()

            responses = [qwen_norm, gemini_norm]
            yes_count = sum(1 for r in responses if "YES" in r)
            no_count = sum(1 for r in responses if "NO" in r)
            partial_count = sum(1 for r in responses if "PARTIAL" in r)

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
                    'final_result': result.upper(),
                    'qwen_response': result.upper(),
                    'gemini_response': None,
                    'consensus': True,  # Single model always has consensus
                    'needs_human_review': False,
                    'votes': None,
                    'raw_responses': {
                        'qwen': result,
                        'gemini': None
                    }
                }
            return result.upper()

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
