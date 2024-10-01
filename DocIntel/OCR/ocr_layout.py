import pytesseract
from PIL import Image
import fitz  # PyMuPDF

class OCRLayoutAnalyzer:
    """Enhanced OCR class with layout analysis."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def extract_text_with_layout(self) -> dict:
        """
        Extracts text and its layout information (bounding boxes) from a PDF or image file.
        Returns
        -------
        dict
            A dictionary containing the text and layout metadata.
        """
        if self.file_path.endswith('.pdf'):
            return self._extract_from_pdf()
        else:
            return self._extract_from_image()
    
    def _extract_from_image(self) -> dict:
        """Extracts text with bounding box info from an image using Tesseract."""
        img = Image.open(self.file_path)
        # Use tesseract to extract text along with bounding boxes
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        return data

    def _extract_from_pdf(self) -> dict:
        """Extracts text with layout info from PDF using PyMuPDF and Tesseract."""
        doc = fitz.open(self.file_path)
        layout_data = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            layout_data.append(data)
        doc.close()
        return layout_data

if __name__ == "__main__":
    analyzer = OCRLayoutAnalyzer("/home/josephmargaryan/DocIntel/data/sample.pdf")
    layout_data = analyzer.extract_text_with_layout()
    print(layout_data)