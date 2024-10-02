import os
import subprocess
import fitz  # PyMuPDF
from io import BytesIO
from typing import Union
from PIL import Image
import camelot
from pytesseract import image_to_string, Output
from google.cloud import vision
from openai import OpenAI

client = OpenAI(api_key="your-openai-api-key")
from docx import Document


class DocumentConverter:
    """Enhanced class to handle conversion of various file formats to images or text."""

    def __init__(self, file: Union[str, BytesIO], output_dir: str = "./output"):
        """
        Parameters
        ----------
        file : Union[str, BytesIO]
            The input file path or binary stream to be converted.
        output_dir : str
            Directory to store the converted files.
        """
        self.file = file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def convert_pdf_to_images(self) -> list:
        """Converts a PDF file to images (one image per page)."""
        if isinstance(self.file, str) and os.path.exists(self.file):
            pdf_doc = fitz.open(self.file)
        else:
            pdf_doc = fitz.open(stream=self.file, filetype="pdf")

        image_paths = []
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap()  # Render page to an image
            output_image_path = os.path.join(
                self.output_dir, f"page_{page_num + 1}.png"
            )
            pix.save(output_image_path)
            image_paths.append(output_image_path)

        pdf_doc.close()
        return image_paths

    def convert_image_to_text(
        self, image_path: str, ocr_service: str = "tesseract"
    ) -> str:
        """Uses OCR to extract text from an image using Tesseract or Google Vision."""
        img = Image.open(image_path)
        if ocr_service == "tesseract":
            return image_to_string(img)
        elif ocr_service == "google_vision":
            return self._google_vision_ocr(img)

    def _google_vision_ocr(self, img: Image) -> str:
        """Extract text using Google Vision OCR."""
        client = vision.ImageAnnotatorClient()
        content = img.tobytes()  # Convert image to bytes
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        return texts[0].description if texts else ""

    def extract_tables_from_pdf(self) -> list:
        """Extracts tables from PDF using Camelot."""
        tables = camelot.read_pdf(self.file, pages="all")
        return [table.df for table in tables]

    def extract_metadata(self) -> dict:
        """Extracts metadata from PDF."""
        pdf_doc = fitz.open(self.file)
        metadata = pdf_doc.metadata
        pdf_doc.close()
        return metadata

    def convert_docx_to_text(self) -> str:
        """Converts a DOCX file to plain text using python-docx."""
        doc = Document(self.file)
        return "\n".join([p.text for p in doc.paragraphs])

    def correct_text_with_gpt(self, text: str) -> str:
        "Uses GPT for spell-checking and post-processing of OCR text."
        response = client.completions.create(
            model="gpt-4",
            prompt=f"Correct the following OCR text: {text}",
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].text.strip()

    def convert(self, ocr_service: str = "tesseract", use_gpt: bool = False) -> dict:
        """Main function to handle the conversion and text extraction."""
        converted = {}

        if self.file.endswith(".pdf"):
            images = self.convert_pdf_to_images()
            converted["images"] = images
            converted["text"] = [
                self.convert_image_to_text(img, ocr_service) for img in images
            ]
            if use_gpt:
                # Apply GPT correction only if use_gpt is True
                converted["text"] = [
                    self.correct_text_with_gpt(text) for text in converted["text"]
                ]

            converted["metadata"] = self.extract_metadata()
            converted["tables"] = self.extract_tables_from_pdf()

        elif self.file.endswith(".docx"):
            text = self.convert_docx_to_text()
            if use_gpt:
                text = self.correct_text_with_gpt(text)
            converted["text"] = text

        else:
            raise ValueError("Unsupported file format")

        return converted


if __name__ == "__main__":
    converter = DocumentConverter("/home/josephmargaryan/DocIntel/data/sample.pdf")
    converted_data = converter.convert(ocr_service="tesseract")
    print("Extracted Text:", converted_data["text"])
    print("Metadata:", converted_data["metadata"])
    print("Tables:", converted_data["tables"])
