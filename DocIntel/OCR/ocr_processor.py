import cv2
import pytesseract
import easyocr
from PIL import Image
from paddleocr import PaddleOCR
from kraken.lib import models
from kraken import pageseg
import ocrmypdf
import os


class OCRProcessor:
    def __init__(
        self,
        ocr_service: str = "tesseract",
        use_preprocessing: bool = False,
        preprocessing_methods: list = None,
    ):
        """
        Initialize the OCR processor with a choice of OCR service and optional preprocessing.

        Parameters:
        ocr_service (str): The OCR engine to use. Options include 'tesseract', 'easyocr', 'paddleocr', 'kraken', 'ocrmypdf'.
        use_preprocessing (bool): Whether to apply preprocessing to the image.
        preprocessing_methods (list): List of preprocessing methods to apply. Available options: 'adaptive_threshold', 'denoise', 'edge_detection'.
        """
        self.ocr_service = ocr_service
        self.use_preprocessing = use_preprocessing
        self.preprocessing_methods = (
            preprocessing_methods if preprocessing_methods else []
        )

    def preprocess_image(self, image_path: str) -> str:
        """Preprocesses an image dynamically based on the selected preprocessing methods."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply adaptive thresholding
        if "adaptive_threshold" in self.preprocessing_methods:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

        # Apply Gaussian blur for denoising
        if "denoise" in self.preprocessing_methods:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply edge detection
        if "edge_detection" in self.preprocessing_methods:
            img = cv2.Canny(img, 100, 200)

        # Save the preprocessed image in the same directory with '_preprocessed'
        base, ext = os.path.splitext(image_path)
        preprocessed_image_path = f"{base}_preprocessed{ext}"
        cv2.imwrite(preprocessed_image_path, img)
        return preprocessed_image_path

    def convert_image_to_text(self, image_path: str) -> str:
        """Extracts text from an image using the selected OCR engine with optional preprocessing."""
        try:
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                image_path = self.preprocess_image(image_path)

            img = Image.open(image_path)

            # Tesseract OCR
            if self.ocr_service == "tesseract":
                config = "--oem 3 --psm 6"
                return pytesseract.image_to_string(img, config=config)

            # EasyOCR
            elif self.ocr_service == "easyocr":
                reader = easyocr.Reader(["en"])
                result = reader.readtext(image_path, detail=0)
                return " ".join(result)

            # PaddleOCR extraction needs to handle nested lists
            elif self.ocr_service == "paddleocr":
                ocr = PaddleOCR(use_angle_cls=True, lang="en")
                result = ocr.ocr(image_path)

                # Extract text from nested list structure
                extracted_text = []
                for line in result:
                    for word_info in line:
                        # word_info is a list where [1][0] is the actual text
                        extracted_text.append(word_info[1][0])

                return " ".join(extracted_text)  # Join all text into a single string

            # Kraken OCR
            elif self.ocr_service == "kraken":
                model = models.load_any("default")
                segments = pageseg.segment(img)
                result = model.recognize(segments)
                return result["text"]

            # OCRmyPDF (for PDF files only)
            elif self.ocr_service == "ocrmypdf":
                pdf_output = "output_ocr.pdf"
                ocrmypdf.ocr(image_path, pdf_output, deskew=True)
                return f"OCR'd PDF saved to: {pdf_output}"

        except Exception as e:
            print(f"Error during image to text conversion: {e}")
            return ""

    def convert_pdf_to_text(self, pdf_path: str) -> str:
        """Use OCRmyPDF specifically for PDF OCR processing."""
        try:
            pdf_output = "output_ocr.pdf"
            ocrmypdf.ocr(pdf_path, pdf_output, deskew=True)
            return f"OCR'd PDF saved to: {pdf_output}"
        except Exception as e:
            print(f"Error during PDF to text conversion: {e}")
            return ""


if __name__ == "__main__":
    # Test the OCRProcessor with different engines and preprocessing methods
    # ocr_processor = OCRProcessor(ocr_service='tesseract', use_preprocessing=True, preprocessing_methods=['adaptive_threshold', 'denoise'])
    ocr_processor = OCRProcessor(
        ocr_service="paddleocr"
    )  # paddleocr is best with no preprocessing so far, 'kraken', 'ocrmypdf' doesnt work that well as of right now!
    text = ocr_processor.convert_image_to_text(
        "/home/josephmargaryan/DocIntel/output/page_1.png"
    )
    print(text)
