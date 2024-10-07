

def main():
    file_path = "/path/to/your/pdf"
    output_dir = "./output"

    # Initialize the Document Processing Pipeline
    pipeline = DocumentProcessingPipeline(
        file_path, output_dir, ocr_service="paddleocr", use_correction=True
    )

    # Process the document
    results = pipeline.process_document(file_type="pdf")
    print(results["text"])

if __name__ == "__main__":
    main()

