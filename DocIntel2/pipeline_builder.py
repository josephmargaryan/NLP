import yaml

class PipelineBuilder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def build_pipeline(self):
        steps = self.config["pipeline"]
        pipeline = []
        for step in steps:
            if step["task"] == "ocr":
                processor = OCRProcessor(ocr_service=step["ocr_service"])
                pipeline.append(processor)
            elif step["task"] == "ml":
                ml_pipeline = MLPipeline(model_name=step["model_name"], task=step["task_type"])
                pipeline.append(ml_pipeline)
            # Add more tasks dynamically here
        return pipeline
