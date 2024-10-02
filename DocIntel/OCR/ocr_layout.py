import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pytesseract import image_to_string, Output
from google.cloud import vision


class OCRLayoutAnalyzer:
    """Enhanced OCR class with layout analysis."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text_with_layout(self, ocr_service: str = "tesseract") -> dict:
        """
        Extracts text and its layout information (bounding boxes) from a PDF or image file.
        Returns
        -------
        dict
            A dictionary containing the text and layout metadata.
        """
        if self.file_path.endswith(".pdf"):
            return self._extract_from_pdf(ocr_service)
        else:
            return self._extract_from_image(ocr_service)

    def _extract_from_image(self, ocr_service: str = "tesseract") -> dict:
        """Extracts text with bounding box info from an image."""
        img = Image.open(self.file_path)
        data = (
            pytesseract.image_to_data(img, output_type=Output.DICT)
            if ocr_service == "tesseract"
            else self._google_vision_layout(img)
        )
        return data

    def _extract_from_pdf(self, ocr_service: str = "tesseract") -> dict:
        """Extracts text with layout info from PDF using PyMuPDF and Tesseract."""
        doc = fitz.open(self.file_path)
        layout_data = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data = (
                pytesseract.image_to_data(img, output_type=Output.DICT)
                if ocr_service == "tesseract"
                else self._google_vision_layout(img)
            )
            layout_data.append(data)
        doc.close()
        return layout_data

    def _google_vision_layout(self, img: Image) -> dict:
        """Extract layout information using Google Vision."""
        client = vision.ImageAnnotatorClient()
        content = img.tobytes()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        text_data = {
            "text": response.full_text_annotation.text,
            "bounding_boxes": [v.bounding_poly for v in response.text_annotations],
        }
        return text_data


if __name__ == "__main__":
    analyzer = OCRLayoutAnalyzer("/home/josephmargaryan/DocIntel/data/sample.pdf")
    layout_data = analyzer.extract_text_with_layout()
    print(layout_data)
