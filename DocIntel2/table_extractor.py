class TableExtractor:
    def __init__(self):
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    def extract_tables(self, file: str, method: str = "lattice", use_deep_learning: bool = False) -> list:
        if use_deep_learning:
            return self.extract_tables_deep_learning(file)
        if method in ["lattice", "stream"]:
            return self.extract_tables_camelot(file, method=method)
        else:
            return self.extract_tables_tabula(file)
