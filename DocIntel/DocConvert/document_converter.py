import os
import subprocess
import fitz  # PyMuPDF
from io import BytesIO
from typing import Union
from PIL import Image

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
            output_image_path = os.path.join(self.output_dir, f"page_{page_num + 1}.png")
            pix.save(output_image_path)
            image_paths.append(output_image_path)

        pdf_doc.close()
        return image_paths

    def convert_image_to_text(self, image_path: str) -> str:
        """Uses Tesseract OCR to extract text from an image."""
        from pytesseract import image_to_string
        img = Image.open(image_path)
        return image_to_string(img)

    def convert_docx_to_text(self) -> str:
        """Converts a DOCX file to plain text using python-docx."""
        from docx import Document
        doc = Document(self.file)
        return '\n'.join([p.text for p in doc.paragraphs])

    def convert(self) -> dict:
        """Main function to handle the conversion and text extraction."""
        converted = {}
        if self.file.endswith(".pdf"):
            images = self.convert_pdf_to_images()
            converted['images'] = images
            converted['text'] = [self.convert_image_to_text(img) for img in images]
        elif self.file.endswith(".docx"):
            converted['text'] = self.convert_docx_to_text()
        else:
            raise ValueError("Unsupported file format")
        
        return converted
