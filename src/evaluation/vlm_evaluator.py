"""
VLM Evaluator for Attribute Detection
Uses Qwen-VL and Gemini Flash ensemble for soft erasure detection.
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
        """Load Qwen-VL model."""
        if self.qwen_model is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct"
                )
                print("Qwen-VL loaded successfully")
            except Exception as e:
                print(f"Failed to load Qwen-VL: {e}")

    def _query_qwen(self, image: Image.Image, prompt: str) -> str:
        """Query Qwen-VL model."""
        self._load_qwen()

        if self.qwen_model is None:
            return "UNKNOWN"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.qwen_processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.qwen_model.device)

        output_ids = self.qwen_model.generate(**inputs, max_new_tokens=10)
        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return response.strip()

    def _query_gemini(self, image: Image.Image, prompt: str) -> str:
        """Query Gemini Flash model."""
        try:
            import google.generativeai as genai
            import os

            if self.gemini_model is None:
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

            response = self.gemini_model.generate_content([prompt, image])
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
        return self._query_qwen(image, prompt)
