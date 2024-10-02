from DocConvert.document_converter import DocumentConverter
from OCR.ocr_processor import OCRProcessor
from PostProcessing.text_post_processor import Seq2SeqTextCorrector
from TableExtraction.table_extractor import TableExtractor


class DocumentProcessingPipeline:
    """
    A class to process documents (PDFs, DOCX, images), performing OCR, extracting tables, correcting text,
    and extracting metadata.
    """

    def __init__(
        self,
        file_path: str,
        output_dir: str = "./output",
        ocr_service: str = "tesseract",
        use_correction: bool = False,
    ):
        """
        Initialize the pipeline with the necessary file path and configurations.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.ocr_service = ocr_service
        self.use_correction = use_correction
        self.converter = DocumentConverter(file_path, output_dir)
        self.ocr = OCRProcessor(ocr_service)
        self.table_extractor = TableExtractor()
        self.text_corrector = Seq2SeqTextCorrector() if use_correction else None

    def process_document(self, file_type: str = "pdf"):
        """
        Main function to process the document based on the file type (PDF, DOCX, image).
        """
        # Initialize results dictionary
        results = {"text": "", "metadata": {}, "tables": []}

        # Handle different file types
        if file_type == "pdf":
            # Extract metadata
            results["metadata"] = self.converter.extract_metadata()

            # First, try extracting tables directly from the PDF
            results["tables"] = self.table_extractor.extract_tables(
                self.file_path, method="lattice"
            )

            # Convert PDF to images for OCR
            image_paths = self.converter.convert_pdf_to_images()

            # Perform OCR on each page and correct text if required
            extracted_texts = []
            for img_path in image_paths:
                text = self.ocr.convert_image_to_text(img_path)
                if self.use_correction:
                    text = self.text_corrector.correct_text(text)
                extracted_texts.append(text)
            results["text"] = " ".join(extracted_texts)

        elif file_type == "docx":
            # Extract text directly from DOCX
            results["text"] = self.converter.convert_docx_to_text()

        elif file_type in ["png", "jpg", "jpeg"]:
            # Handle image files directly
            text = self.ocr.convert_image_to_text(self.file_path)
            if self.use_correction:
                text = self.text_corrector.correct_text(text)
            results["text"] = text

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return results


if __name__ == "__main__":
    # Specify the file path and output directory
    file_path = "/home/josephmargaryan/DocIntel/data/sample_with_table.pdf"
    output_dir = "./output"

    # Initialize the Document Processing Pipeline
    # Set use_correction to True if you want to enable text correction
    pipeline = DocumentProcessingPipeline(
        file_path, output_dir, ocr_service="paddleocr", use_correction=True
    )

    # Process the document (specify the file type as needed)
    try:
        results = pipeline.process_document(file_type="pdf")

        # Display the extracted metadata
        print("Extracted Metadata:")
        for key, value in results["metadata"].items():
            print(f"{key}: {value}")

        # Display the extracted text
        print("\nExtracted Text:")
        print(results["text"])

        # Display the extracted tables
        print("\nExtracted Tables:")
        for idx, table in enumerate(results["tables"]):
            print(f"Table {idx + 1}:")
            print(table)

    except Exception as e:
        print(f"An error occurred during document processing: {e}")
