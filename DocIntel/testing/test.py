converter = DocumentConverter("sample.pdf")
converted_data = converter.convert()
print(converted_data['text'])

ocr_analyzer = OCRLayoutAnalyzer("sample.pdf")
layout_data = ocr_analyzer.extract_text_with_layout()
bounding_boxes = calculate_bounding_box(layout_data[0])
print(bounding_boxes)

layout_analyzer = AdvancedLayoutAnalyzer()
layout_info = layout_analyzer.analyze_document("sample.png")
print(layout_info)

