import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class Inference:
    def __init__(self, model_path, tokenizer_name, device, pooling_strategy="mean"):
        self.model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.pooling_strategy = pooling_strategy
        self.model.to(self.device)
        self.model.eval()

    def preprocess_data(self, df):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
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

    def sliding_window_inference_CLS(self, text, max_len=510, stride=250):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding='max_length'
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        input_len = input_ids.shape[1]
        chunks = []
        attention_chunks = []

        for i in range(0, input_len, stride):
            end_index = min(i + max_len, input_len)
            chunk = input_ids[:, i:end_index]
            attn_chunk = attention_mask[:, i:end_index]

            if chunk.shape[1] < max_len:
                pad_length = max_len - chunk.shape[1]
                chunk = torch.cat([chunk, torch.zeros((1, pad_length), dtype=torch.long).to(self.device)], dim=1)
                attn_chunk = torch.cat([attn_chunk, torch.zeros((1, pad_length), dtype=torch.long).to(self.device)], dim=1)

            chunks.append(chunk)
            attention_chunks.append(attn_chunk)

        outputs = []

        for chunk, attn_chunk in zip(chunks, attention_chunks):
            with torch.no_grad():
                output = self.model(input_ids=chunk, attention_mask=attn_chunk)
                cls_embedding = output.last_hidden_state[:, 0, :]
                outputs.append(cls_embedding)

        aggregated_output = torch.cat(outputs, dim=0)

        if self.pooling_strategy == "mean":
            pooled_output = aggregated_output.mean(dim=0)
        elif self.pooling_strategy == "max":
            pooled_output = aggregated_output.max(dim=0).values
        elif self.pooling_strategy == "attention":
            pooled_output = self.attention_pooling(aggregated_output)

        return pooled_output

    def attention_pooling(self, embeddings):
        attention = nn.Sequential(
            nn.Linear(embeddings.size(-1), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        weights = attention(embeddings).softmax(dim=0)
        weighted_avg = (weights * embeddings).sum(dim=0)
        return weighted_avg

    def infer(self, text):
        pooled_output = self.sliding_window_inference_CLS(text)
        logits = self.model.classifier(pooled_output.unsqueeze(0))  # Apply the final classification layer
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

    # Initialize Inference class
    inference = Inference(
        model_path="best_model.pth",  # Path to your trained model's state_dict
        tokenizer_name="bert-base-multilingual-cased",  # The same tokenizer you used for training
        device=device,
        pooling_strategy="mean"  # Change this to "max" or "attention" as needed
    )

    # Load your data
    df = pd.read_csv("/kaggle/working/sample.csv")  # Path to your inference data

    # Run inference
    df_with_predictions = inference.run_inference(df)

    # Save the resulting dataframe
    df_with_predictions.to_csv("inference_results.csv", index=False)

