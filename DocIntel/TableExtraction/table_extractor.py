import camelot
import tabula
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output


class TableExtractor:
    def __init__(self):
        # Load the LayoutLMv3 model and tokenizer for table detection
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.model.eval()

    def extract_tables_camelot(self, file: str, method: str = "lattice") -> list:
        """Extracts tables from PDF using Camelot's lattice or stream method."""
        tables = camelot.read_pdf(file, pages="all", flavor=method)
        return [table.df for table in tables]

    def extract_tables_tabula(self, file: str) -> list:
        """Extracts tables from PDF using Tabula."""
        tables = tabula.read_pdf(file, pages="all", multiple_tables=True)
        return tables

    def extract_tables_deep_learning(self, file: str) -> list:
        """Extract tables from PDF using deep learning (LayoutLMv3)."""
        # Convert PDF to images
        images = convert_from_path(file)
        extracted_tables = []

        for image in images:
            # Use pytesseract to get text from the image
            ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
            words = ocr_data["text"]

            # Filter out empty words
            words = [word for word in words if word.strip()]

            # Create dummy bounding boxes for the words (arbitrary coordinates)
            boxes = [[0, 0, 50, 50]] * len(
                words
            )  # Create the same arbitrary box for each word

            if len(words) == 0:
                continue  # Skip if no words were extracted

            # Tokenize the text and dummy layout
            inputs = self.tokenizer(
                text=words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract tables from model predictions (this is a simplified step)
            predicted_labels = outputs.logits.argmax(-1).squeeze().tolist()
            extracted_tables.append(
                predicted_labels
            )  # Placeholder for actual table extraction logic

        return extracted_tables

    def extract_tables(
        self, file: str, method: str = "lattice", use_deep_learning: bool = False
    ) -> list:
        """
        Main method to extract tables from a PDF using Camelot, Tabula, or Deep Learning.

        Parameters:
        - method: 'lattice' or 'stream' for Camelot; ignored if using deep learning.
        - use_deep_learning: Whether to use a deep learning model (LayoutLMv3) for table extraction.
        """
        if use_deep_learning:
            return self.extract_tables_deep_learning(file)

        # Fallback to Camelot or Tabula based on PDF structure
        if method in ["lattice", "stream"]:
            return self.extract_tables_camelot(file, method=method)
        else:
            return self.extract_tables_tabula(file)


if __name__ == "__main__":
    # Specify the PDF file that contains tables for testing
    file_path = "/home/josephmargaryan/DocIntel/data/sample_with_table.pdf"

    # Create an instance of the TableExtractor class
    table_extractor = TableExtractor()

    # Test Camelot's lattice method
    print("Testing Camelot's lattice method:")
    camelot_lattice_tables = table_extractor.extract_tables(file_path, method="lattice")
    for idx, table in enumerate(camelot_lattice_tables):
        print(f"Table {idx + 1} (Camelot Lattice):")
        print(table)

    # Test Camelot's stream method
    print("\nTesting Camelot's stream method:")
    camelot_stream_tables = table_extractor.extract_tables(file_path, method="stream")
    for idx, table in enumerate(camelot_stream_tables):
        print(f"Table {idx + 1} (Camelot Stream):")
        print(table)

    # Test Tabula's table extraction
    print("\nTesting Tabula method:")
    tabula_tables = table_extractor.extract_tables(file_path, method="tabula")
    for idx, table in enumerate(tabula_tables):
        print(f"Table {idx + 1} (Tabula):")
        print(table)

    # Test Deep Learning-based table extraction (LayoutLMv3)
    print("\nTesting Deep Learning method (LayoutLMv3):")
    deep_learning_tables = table_extractor.extract_tables(
        file_path, use_deep_learning=True
    )
    for idx, table in enumerate(deep_learning_tables):
        print(f"Table {idx + 1} (Deep Learning):")
        print(table)
