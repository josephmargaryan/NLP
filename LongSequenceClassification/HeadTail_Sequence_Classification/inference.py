import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm

class HeadTailInference:
    def __init__(self, model_path, tokenizer_name, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def preprocess_data(self, df):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
            text = re.sub(r'\s+', ' ', text).strip()
            emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  
                                    u"\U0001F300-\U0001F5FF" 
                                    u"\U0001F680-\U0001F6FF"  
                                    u"\U0001F1E0-\U0001F1FF"  
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            
            return text

        df['x'] = df['x'].apply(clean_text)
        return df

    def tokenize_and_select(self, text, first_tokens=128, last_tokens=382):
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False, padding=False)
        input_ids = tokens['input_ids'].squeeze()

        if input_ids.size(0) < (first_tokens + last_tokens):
            print(f"Warning: The input sequence is shorter than {first_tokens + last_tokens} tokens.")
            return None, None

        selected_ids = torch.cat([input_ids[:first_tokens], input_ids[-last_tokens:]])
        attention_mask = torch.ones_like(selected_ids)

        # Padding if necessary
        if selected_ids.size(0) < (first_tokens + last_tokens):
            padding_length = (first_tokens + last_tokens) - selected_ids.size(0)
            selected_ids = torch.cat([selected_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        return selected_ids, attention_mask

    def infer(self, text):
        input_ids, attention_mask = self.tokenize_and_select(text)

        if input_ids is None or attention_mask is None:
            return None, None  # Skip inference if tokenization failed

        input_ids = input_ids.unsqueeze(0).to(self.device)  # Add batch dimension
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).cpu().item()
            pred_prob = torch.max(probs, dim=-1).values.cpu().item()

        return pred_label, pred_prob

    def run_inference(self, df):
        df = self.preprocess_data(df)
        preds = []
        probs = []

        for text in tqdm(df['x'], desc="Running inference", unit="text"):
            pred_label, pred_prob = self.infer(text)
            preds.append(pred_label)
            probs.append(pred_prob)

        df['pred_label'] = preds
        df['pred_prob'] = probs

        return df


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize HeadTailInference class
    inference = HeadTailInference(
        model_path="best_model.pth",  # Path to your trained model's state_dict
        tokenizer_name="google-bert/bert-base-multilingual-cased",  # The same tokenizer you used for training
        device=device
    )

    # Load your data
    df = pd.read_csv("path/to/inference_data.csv")  # Path to your inference data

    # Run inference
    df_with_predictions = inference.run_inference(df)

    # Save the resulting dataframe
    df_with_predictions.to_csv("inference_results.csv", index=False)
