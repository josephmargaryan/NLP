# **DocIntel (Document Intelligence)**

### **Advanced Document Conversion, OCR, and Layout Analysis**

**DocIntel** is a comprehensive document processing library that combines cutting-edge OCR (Optical Character Recognition), document conversion, and layout analysis technologies. With support for a variety of file formats like PDFs, images, and DOCX, DocIntel provides high-performance tools for text extraction and semantic understanding of document layouts. Utilizing state-of-the-art models like LayoutLMv3 alongside traditional methods like Tesseract, DocIntel goes beyond simple OCR by preserving the structure and layout of documents for more intelligent analysis.

## **Features**

- **Multi-format Document Conversion**:
    - Convert PDFs into images for further processing.
    - Extract text from PDFs, DOCX, and images.
  
- **Optical Character Recognition (OCR)**:
    - Extract text from images and scanned PDFs using Tesseract, EasyOCR, and PaddleOCR.
    - Advanced layout-based OCR that provides bounding boxes for text blocks, enhancing text extraction accuracy.

- **Layout Analysis**:
    - Calculate and extract bounding box coordinates to maintain document structure.
    - Use state-of-the-art models like LayoutLMv3 for semantic document understanding, enabling deeper insights into document content.
  
- **Table Extraction**:
    - Extract structured data from tables in PDFs using Camelot and Tabula.
    - Integrate deep learning approaches for improved table detection and extraction.

- **Text Correction**:
    - Utilize advanced NLP models to correct OCR errors and enhance the readability of extracted text.
  
- **Scalable and Extendable**:
    - Easily extendable to support additional document formats and OCR engines.
    - Capable of handling large-scale document processing workflows.


## **Getting Started**

### **Prerequisites**

- Python 3.7 or later
- Install dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### **Tesseract Setup**
To enable OCR capabilities, you will need to install Tesseract OCR. On most Linux-based systems:
```
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install ghostscript
sudo apt install default-jdk
sudo apt install poppler-utils
```

on mac(using homebrew)
```
brew install tesseract
brew install ghostscript
brew install java
brew install poppler
```

### **Contact**
For any inquiries or support, please reach out to ____
