import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification
import re
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
import pickle

class Head_Tail_Training:
    def __init__(self, model, tokenizer, path_to_data, device):
        self.model = model
        self.tokenizer = tokenizer
        self.path_to_data = path_to_data
        self.device = device 

    def load_data(self):
        """
        Takes the file path with the parquet files, concatenates them, and removes documents with non-manual classification.
        """
        out_folder = self.path_to_data
        regex = ".parquet"

        folders = os.listdir(out_folder)
        dataframes = []
        for folder in folders:
            folder_path = os.path.join(out_folder, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                for file in tqdm(files, desc=f"Processing files in {folder}", unit="file"):
                    if file.endswith(regex):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_parquet(file_path)
                        dataframes.append(df)

        final_df = pd.concat(dataframes, axis=0) 
        final_df.reset_index(drop=True, inplace=True)
        df_filtered = final_df.loc[~final_df["text"].isna(), :]
        df_filtered = df_filtered[df_filtered["text"].apply(lambda x: len(str(x)) > 10)]
        df_filtered = df_filtered.loc[~df_filtered["CLASSIFICATION"].isna(), :]
        print(f"The number of total samples: {len(df_filtered)})")
        df = df_filtered[["text", "CLASSIFICATION"]]
        df = df.rename(columns={"text": "x"})
        le = LabelEncoder()
        df["y"] = le.fit_transform(df["CLASSIFICATION"])
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)

        return df
    
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
    
    def tokenize(self):
        def tokenize_and_select(text, tokenizer, first_tokens=128, last_tokens=382):
            tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
            input_ids = tokens['input_ids'].squeeze()

            selected_ids = torch.cat([input_ids[:first_tokens], input_ids[-last_tokens:]])

            attention_mask = torch.ones_like(selected_ids)

            if selected_ids.size(0) < (first_tokens + last_tokens):
                padding_length = (first_tokens + last_tokens) - selected_ids.size(0)
                selected_ids = torch.cat([selected_ids, torch.zeros(padding_length, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

            return selected_ids, attention_mask
        
        df = self.preprocess_data()
        df[['input_ids', 'attention_mask']] = df['x'].apply(lambda x: tokenize_and_select(x, self.tokenizer)).apply(pd.Series)

        all_input_ids = torch.stack(df['input_ids'].tolist())
        all_attention_masks = torch.stack(df['attention_mask'].tolist())
        all_labels = torch.tensor(df['y'].values)

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
            for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
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
            for j, (input_ids, attention_mask, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
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
        plt.plot(df.index + 1, df["train"], label="train loss")
        plt.plot(df.index + 1, df["val"], label="val loss")
        plt.plot(df.index + 1, df["val_accuracy"], label="val accuracy")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title("Loss and Accuracy over epochs")
        plt.legend()
        plt.savefig("metrics_curve.png")
        plt.show()
        
        if best_model_state is not None:
            torch.save(best_model_state, "best_model.pth")



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased', num_labels=91)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Head_Tail_Training(model, 
                                 tokenizer, 
                                 "/data-disk/scraping-output/p-drive-structured", 
                                 device)
    trainer.train(num_epochs=100, lr=2e-5)