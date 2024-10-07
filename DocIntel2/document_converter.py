import fitz  # PyMuPDF
from io import BytesIO, BufferedReader
from typing import List, Union
import os


class DocumentConverter:
    def __init__(self, file: Union[str, bytes], output_dir: str = "./output"):
        self.file = file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def convert_pdf_to_images(self) -> List[str]:
        """Converts a PDF file to images (one image per page)."""
        pdf_doc = fitz.open(self.file)
        image_paths = []
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap()
            output_image_path = os.path.join(
                self.output_dir, f"page_{page_num + 1}.png"
            )
            pix.save(output_image_path)
            image_paths.append(output_image_path)
        pdf_doc.close()
        return image_paths

    def extract_metadata(self) -> dict:
        """Extracts metadata from PDF."""
        pdf_doc = fitz.open(self.file)
        metadata = pdf_doc.metadata
        pdf_doc.close()
        return metadata

    def extract_text(self) -> str:
        """Extracts structured text from PDF using PyMuPDF."""
        pdf_doc = fitz.open(self.file)
        text = ""
        for page in pdf_doc:
            text += page.get_text("text")  # Extracts plain text
        pdf_doc.close()
        return text

    def extract_structured_text(self) -> dict:
        """Extracts structured text from PDF."""
        pdf_doc = fitz.open(self.file)
        structured_data = []
        for page in pdf_doc:
            blocks = page.get_text("dict")["blocks"]  # Extract blocks of text
            for block in blocks:
                if block["type"] == 0:  # Text block
                    structured_data.append(block)
        pdf_doc.close()
        return structured_data
