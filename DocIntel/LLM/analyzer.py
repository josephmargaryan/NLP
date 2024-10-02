from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image


class AdvancedLayoutAnalyzer:
    """Uses LayoutLMv3 for advanced layout and text understanding."""

    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

    def analyze_document(self, image_path: str) -> dict:
        """Analyzes the layout of a document using LayoutLMv3."""
        image = Image.open(image_path).convert("RGB")
        encoded_inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**encoded_inputs)
        predictions = outputs.logits.argmax(-1)
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            encoded_inputs["input_ids"].squeeze().tolist()
        )
        return {"tokens": tokens, "predictions": predictions}
