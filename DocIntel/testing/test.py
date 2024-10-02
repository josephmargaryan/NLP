from DocConvert.document_converter import DocumentConverter
from OCR.ocr_layout import OCRLayoutAnalyzer
from BoundBox.bounding_box import calculate_bounding_box
from LLM.analyzer import AdvancedLayoutAnalyzer

# Convert PDF to Images and Extract Tables, Metadata
converter = DocumentConverter("/home/josephmargaryan/DocIntel/data/sample.pdf")
converted_data = converter.convert(ocr_service="tesseract")
print("Extracted Text:", converted_data["text"])
print("Metadata:", converted_data["metadata"])
print("Tables:", converted_data["tables"])

# Perform OCR Layout Analysis
ocr_analyzer = OCRLayoutAnalyzer("/home/josephmargaryan/DocIntel/data/sample.pdf")
layout_data = ocr_analyzer.extract_text_with_layout(ocr_service="tesseract")
print("Layout Data:", layout_data)

# Analyze Document Layout with LayoutLMv3
layout_analyzer = AdvancedLayoutAnalyzer()
for image_path in converted_data["images"]:
    layout_info = layout_analyzer.analyze_document(image_path)
    print("Layout Analysis for Image:", layout_info)
