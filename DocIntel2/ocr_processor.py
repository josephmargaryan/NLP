class OCRProcessor:
    def __init__(
        self,
        ocr_service: str = "tesseract",
        use_preprocessing: bool = False,
        preprocessing_methods: list = None,
    ):
        self.ocr_service = ocr_service
        self.use_preprocessing = use_preprocessing
        self.preprocessing_methods = (
            preprocessing_methods if preprocessing_methods else []
        )

    def preprocess_image(self, image_path: str) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if "adaptive_threshold" in self.preprocessing_methods:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        if "denoise" in self.preprocessing_methods:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if "edge_detection" in self.preprocessing_methods:
            img = cv2.Canny(img, 100, 200)

        base, ext = os.path.splitext(image_path)
        preprocessed_image_path = f"{base}_preprocessed{ext}"
        cv2.imwrite(preprocessed_image_path, img)
        return preprocessed_image_path

    def convert_image_to_text(self, image_path: str) -> str:
        try:
            if self.use_preprocessing:
                image_path = self.preprocess_image(image_path)

            img = Image.open(image_path)
            if self.ocr_service == "tesseract":
                config = "--oem 3 --psm 6"
                return pytesseract.image_to_string(img, config=config)

        except Exception as e:
            print(f"Error during image to text conversion: {e}")
            return ""

    def extract_pdf_text_or_ocr(self, pdf_path: str) -> str:
        converter = DocumentConverter(pdf_path)
        # Try extracting text first
        text = converter.extract_text()
        if not text.strip():  # If text is empty, fallback to OCR
            image_paths = converter.convert_pdf_to_images()
            text = ""
            for img_path in image_paths:
                text += self.convert_image_to_text(img_path)
        return text
