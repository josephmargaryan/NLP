from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Seq2SeqTextCorrector:
    def __init__(self, model_name: str = "oliverguhr/spelling-correction-english-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def correct_text(self, text: str) -> str:
        """
        Correct OCR text using a sequence-to-sequence model (like T5).
        Automatically handles texts that exceed the token limit by splitting into chunks.
        """
        tokenized_input = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_length = tokenized_input.input_ids.shape[1]

        if input_length > 512:  # Assuming 512 tokens is the limit
            # Split the text into chunks
            chunks = [text[i : i + 512] for i in range(0, len(text), 512)]
            corrected_chunks = []

            for chunk in chunks:
                inputs = self.tokenizer(
                    f"correct: {chunk}",
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )
                outputs = self.model.generate(
                    **inputs, max_new_tokens=50
                )  # Only use max_new_tokens
                corrected_chunk = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                corrected_chunks.append(corrected_chunk)

            return " ".join(corrected_chunks)

        # For short text
        inputs = self.tokenizer(
            f"correct: {text}", return_tensors="pt", truncation=True, padding=True
        )
        outputs = self.model.generate(
            **inputs, max_new_tokens=50
        )  # Only use max_new_tokens
        corrected_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return corrected_text


# Example usage
if __name__ == "__main__":
    corrector = Seq2SeqTextCorrector()
    original_text = "Ths is an exmple of corcting OCR txt using a langug model."
    corrected_text = corrector.correct_text(original_text)
    print("Original Text:", original_text)
    print("Corrected Text:", corrected_text)
