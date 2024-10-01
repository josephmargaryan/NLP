# **DocIntel**

### **Advanced Document Conversion, OCR, and Layout Analysis**

**DocIntel** is a comprehensive document processing library that combines cutting-edge OCR (Optical Character Recognition), document conversion, and layout analysis technologies. With support for a variety of file formats like PDFs, images, and DOCX, DocIntel provides high-performance tools for text extraction and semantic understanding of document layouts. Using state-of-the-art models like LayoutLMv3 and traditional methods like Tesseract, DocIntel goes beyond simple OCR by preserving the structure and layout of documents for more intelligent analysis.

## **Features**

- **Multi-format Document Conversion**:
    - Convert PDFs into images for further processing.
    - Extract text from PDFs, DOCX, and images.
  
- **Optical Character Recognition (OCR)**:
    - Extract text from images and scanned PDFs using Tesseract.
    - Advanced layout-based OCR that provides bounding boxes for text blocks.
  
- **Layout Analysis**:
    - Calculate and extract bounding box coordinates to maintain document structure.
    - Use state-of-the-art models like LayoutLMv3 for semantic document understanding.
  
- **State-of-the-Art NLP Integration**:
    - Use Hugging Face’s NLP models for text summarization, question answering, and text generation.
  
- **Scalable and Extendable**:
    - Easily extendable to support additional document formats and OCR engines.
    - Capable of handling large-scale document processing workflows.

## **Getting Started**

### **Prerequisites**

- Python 3.7 or later
- Install dependencies via `pip`:

```bash
pip install -r requirements.txt

### **Prerequisites**
git clone https://github.com/yourusername/DocIntel.git
cd DocIntel

### **Dependencies**
The following Python packages are required to run DocIntel:
- **PyMuPDF (fitz) for PDF processing**
- **Pillow for image processing**
- **pytesseract for OCR (Tesseract OCR)**
- **transformers for NLP tasks (LayoutLMv3, Text2Text, Question Answering)**
- **python-docx for DOCX file handling**

Install these dependencies using:
pip install pymupdf pillow pytesseract transformers python-docx

### **Tesseract Setup**
To enable OCR capabilities, you will need to install Tesseract OCR. On most Linux-based systems:
sudo apt-get install tesseract-ocr

on mac(using homebrew)
brew install tesseract

## **Usage**
- **PDF to Images and Text Extraction**
Here are some basic usage examples to get started:
from docintel.document_converter import DocumentConverter

converter = DocumentConverter("sample.pdf")
converted_data = converter.convert()
print(converted_data['text'])  # Extracted text from PDF

- **Image OCR with Layout Information**
from docintel.ocr_layout import OCRLayoutAnalyzer

analyzer = OCRLayoutAnalyzer("sample_image.png")
layout_data = analyzer.extract_text_with_layout()
print(layout_data)

- **Advanced Layout Analysis using LayoutLMv3**
from docintel.advanced_layout import AdvancedLayoutAnalyzer

layout_analyzer = AdvancedLayoutAnalyzer()
layout_info = layout_analyzer.analyze_document("sample_image.png")
print(layout_info)

### **Configuration**
DocIntel allows you to configure various components such as OCR engines, output directories, and document formats. You can extend or customize this configuration as per your requirements.

### **Architecture Overview**
- **Document Conversion: Converts PDFs and other file formats to images, enabling OCR and further processing**
- **OCR and Layout Analysis: Extracts text from images and PDFs, along with bounding boxes for layout analysis.**
- **ML and NLP Pipelines: Uses Hugging Face’s transformer models for advanced text analysis like summarization, question answering, and text generation.**

### **Roadmap**
- **Extend support for .pptx, .xlsx, and other document formats.**
- **Integrate with cloud-based OCR APIs (Google Cloud Vision, AWS Textract).**
- **Improve scalability with parallel processing for large document batches.**
- **Add more advanced NLP models for document classification and semantic understanding.**

### **Contributing**
We welcome contributions! Please fork the repository and submit a pull request for any new features, bug fixes, or improvements.

### **Contact**
For any inquiries or support, please reach out to ____