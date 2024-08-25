import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import re
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, tokenizer, path_to_data, device, pooling_strategy="mean"):
        self.model = model
        self.tokenizer = tokenizer
        self.path_to_data = path_to_data
        self.device = device
        self.pooling_strategy = pooling_strategy

    def load_data(self):
        return pd.read_csv(self.path_to_data)
    
    def preprocess_data(self):
        df = self.load_data()

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

    def sliding_window_inference_CLS(self, text, tokenizer, model, max_len=510, stride=250):
        tokens = tokenizer(
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
                output = model(input_ids=chunk, attention_mask=attn_chunk)
                cls_embedding = output.last_hidden_state[:, 0, :]
                outputs.append(cls_embedding)

        aggregated_output = torch.cat(outputs, dim=0)

        if self.pooling_strategy == "mean":
            pooled_output = aggregated_output.mean(dim=0)
        elif self.pooling_strategy == "max":
            pooled_output = aggregated_output.max(dim=0).values
        elif self.pooling_strategy == "attention":
            pooled_output = self.attention_pooling(aggregated_output)

        return pooled_output.cpu().numpy()

    def attention_pooling(self, embeddings):
        attention = nn.Sequential(
            nn.Linear(embeddings.size(-1), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        weights = attention(embeddings).softmax(dim=0)
        weighted_avg = (weights * embeddings).sum(dim=0)
        return weighted_avg

    def generate_embeddings(self):
        df = self.preprocess_data()
        pooled_embeddings = []

        for text in tqdm(df["x"], desc="Generating embeddings", unit="text"):
            pooled_embedding = self.sliding_window_inference_CLS(
                text, self.tokenizer, self.model
            )
            pooled_embeddings.append(pooled_embedding)

        embeddings_np = np.array(pooled_embeddings)
        return df, embeddings_np

    def tokenize(self):
        df = self.preprocess_data()
        input_ids = []
        attention_masks = []
        labels = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            encoded = self.tokenizer.encode_plus(
                row['x'],
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(row['y'])

        all_input_ids = torch.cat(input_ids, dim=0)
        all_attention_masks = torch.cat(attention_masks, dim=0)
        all_labels = torch.tensor(labels, dtype=torch.long)

        return all_input_ids, all_attention_masks, all_labels

    def create_dataloader(self, batch_size=16):
        all_input_ids, all_attention_masks, all_labels = self.tokenize()

        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
        val_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size)

        return train_loader, val_loader

    def accuracy(self, preds, labels):
        _, preds_max = torch.max(preds, 1)
        correct = (preds_max == labels).sum().item()
        return correct / labels.size(0)

    def train(self, num_epochs, lr=2e-5, patience=5, max_grad_norm=1.0):
        train_loader, val_loader = self.create_dataloader()
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_model_state = None
        counter = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            avg_train_loss = []
            self.model.train()
            for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                avg_train_loss.append(loss.item())

            avg_train_loss = np.mean(avg_train_loss)

            avg_val_loss = []
            total_correct = 0
            total_samples = 0
            self.model.eval()
            for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output.loss
                    avg_val_loss.append(loss.item())

                    logits = output.logits
                    total_correct += self.accuracy(logits, labels) * labels.size(0)
                    total_samples += labels.size(0)

            avg_val_loss = np.mean(avg_val_loss)
            val_accuracy = total_correct / total_samples
            val_accuracies.append(val_accuracy)

            if avg_val_loss < best_val_loss:
                counter = 0
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.3f} | Validation Loss: {avg_val_loss:.3f} | Validation Accuracy: {val_accuracy:.3f}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

        min_len = min(len(train_losses), len(val_losses), len(val_accuracies))
        train_losses = train_losses[:min_len]
        val_losses = val_losses[:min_len]
        val_accuracies = val_accuracies[:min_len]

        df = pd.DataFrame({"train": train_losses, "val": val_losses, "val_accuracy": val_accuracies})
        plt.plot(df.index + 1, df["train"], label="Train Loss")
        plt.plot(df.index + 1, df["val"], label="Validation Loss")
        plt.plot(df.index + 1, df["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Loss and Accuracy over Epochs")
        plt.legend()
        plt.savefig("metrics_curve.png")
        plt.show()

        if best_model_state is not None:
            torch.save(best_model_state, "best_model.pth")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=91)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(model=model, 
                      tokenizer=tokenizer, 
                      path_to_data="/kaggle/working/sample.csv", 
                      device=device,
                      pooling_strategy="attention")  # You can set this to "mean", "max", or "attention"

    trainer.train(num_epochs=3, lr=2e-5)