import os
from typing import List, Union
import fitz  # PyMuPDF


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


if __name__ == "__main__":
    converter = DocumentConverter("/home/josephmargaryan/DocIntel/data/sample.pdf")
    images = converter.convert_pdf_to_images()
    print(images)  # Should print paths to the saved images
