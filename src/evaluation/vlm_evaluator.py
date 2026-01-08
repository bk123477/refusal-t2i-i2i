"""
VLM Evaluator for Attribute Detection
Uses Qwen3-VL-8B-Instruct and Gemini Flash ensemble for soft erasure detection.

Model: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- 9B parameters, BF16 precision
- Enhanced visual perception and multimodal reasoning
- Optimized for RTX 4090 (24GB VRAM)

Requirements:
- transformers >= 4.57.0 (for Qwen3-VL support)
- pip install git+https://github.com/huggingface/transformers
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

                self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-8B-Instruct",
                    torch_dtype="bfloat16",
                    device_map="auto",
                    trust_remote_code=True
                )
                self.qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen3-VL-8B-Instruct",
                    trust_remote_code=True
                )
                print("✅ Qwen3-VL-8B-Instruct loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Qwen3-VL-8B-Instruct: {e}")
                print("💡 Make sure transformers is updated: pip install git+https://github.com/huggingface/transformers")

    def _query_qwen(self, image: Image.Image, prompt: str) -> str:
        """Query Qwen3-VL-8B-Instruct model."""
        self._load_qwen()

        if self.qwen_model is None:
            return "UNKNOWN"

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Preparation for inference
            inputs = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.qwen_model.device)

            # Inference: Generation of the output
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=10)
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
        attribute: str
    ) -> str:
        """
        Check if attribute is present in image.

        Args:
            image: Image to analyze
            attribute: Attribute to check for

        Returns:
            "YES", "NO", "PARTIAL", or "UNKNOWN"
        """
        prompt = self.ATTRIBUTE_CHECK_TEMPLATE.format(attribute=attribute)

        if self.use_ensemble:
            # Query both models and vote
            qwen_response = self._query_qwen(image, prompt)
            gemini_response = self._query_gemini(image, prompt)

            # Simple voting
            responses = [qwen_response.upper(), gemini_response.upper()]
            yes_count = sum(1 for r in responses if "YES" in r)
            no_count = sum(1 for r in responses if "NO" in r)
            partial_count = sum(1 for r in responses if "PARTIAL" in r)

            if yes_count > no_count and yes_count > partial_count:
                return "YES"
            elif no_count > yes_count and no_count > partial_count:
                return "NO"
            elif partial_count > 0:
                return "PARTIAL"
            else:
                return "UNKNOWN"
        else:
            # Use Qwen only
            return self._query_qwen(image, prompt)

    def describe_image(self, image: Image.Image) -> str:
        """Get general description of image."""
        prompt = "Describe this image briefly, focusing on the person's appearance and any notable attributes."
        try:
            return self._query_qwen(image, prompt)
        except Exception as e:
            print(f"Image description failed: {e}")
            return "Description unavailable"
