from transformers import pipeline

class MLPipeline:
    def __init__(self, model_name: str, task: str = "question-answering"):
        self.task = task
        self.model = pipeline(task, model=model_name)
    
    def answer_question(self, question: str, context: str) -> str:
        result = self.model(question=question, context=context)
        return result["answer"]

    def generate_text(self, prompt: str) -> str:
        result = self.model(prompt)
        return result[0]["generated_text"]
