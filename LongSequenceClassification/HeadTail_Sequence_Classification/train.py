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
from sklearn.metrics import f1_score, accuracy_score
import transformers

class Head_Tail_Training:
    def __init__(self, path, tokenizer, device):
        self.path = path
        self.model = None
        self.tokenizer = tokenizer
        self.device = device

    def load_data(self):

        df = pd.read_csv(self.path)
        return df.iloc[0:1000, :]

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

        df['x'] = df['review'].apply(clean_text)
        return df
    


    def labelencode(self):
        print(f"Beginning Label Encoding")
 
        df = self.preprocess_data()
        le = LabelEncoder()
        df["y"] = le.fit_transform(df["sentiment"])
        num_classes = len(le.classes_)

        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-uncased",
            num_labels=num_classes).to(self.device)
        
        return df

    
    def tokenize(self):
        print("beginning tokenization")
        def tokenize_and_select(text, tokenizer, max_length=512, first_tokens=128, last_tokens=382):
            tokens = tokenizer(text, add_special_tokens=False)  # Initially without special tokens
            input_ids = tokens['input_ids']

            # Adjusting to add CLS at the beginning and SEP at the end within the maximum length limit
            if len(input_ids) > (max_length - 2):  # Reserve spaces for CLS and SEP
                input_ids = input_ids[:first_tokens] + input_ids[-last_tokens:]
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
            else:
                input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

            attention_mask = [1] * len(input_ids)

            # Padding
            if len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length

            assert len(input_ids) == max_length, "Error in sequence length calculation."
            return torch.tensor(input_ids), torch.tensor(attention_mask)

        
        df = self.labelencode()
        df[['input_ids', 'attention_mask']] = df['x'].apply(lambda x: tokenize_and_select(x, self.tokenizer)).apply(pd.Series)

        all_input_ids = torch.stack(df['input_ids'].tolist())
        all_attention_masks = torch.stack(df['attention_mask'].tolist())
        all_labels = torch.tensor(df['y'].values)

        return all_input_ids, all_attention_masks, all_labels
    
    def create_dataloader(self, batch_size=32):
        print("Creating dataloaders")
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

    def f1_score(self, preds, labels):
        _, preds_max = torch.max(preds, 1)
        return f1_score(labels.cpu().numpy(), preds_max.cpu().numpy(), average='weighted')

    def train(self, num_epochs, lr=2e-5, patience=5, max_grad_norm=1.0):
        train_loader, val_loader = self.create_dataloader()
        self.model.to(self.device)

        scaler = torch.amp.GradScaler()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        train_losses = []
        val_losses = []
        val_f1_scores = []
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

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  
                    output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output.loss

                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()

                avg_train_loss.append(loss.item())

            avg_train_loss = np.mean(avg_train_loss)
            
            avg_val_loss = []
            all_preds = []
            all_labels = []
            self.model.eval()

            for j, (input_ids, attention_mask, labels) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  
                        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = output.loss

                    avg_val_loss.append(loss.item())
                    
                    logits = output.logits
                    all_preds.append(logits)
                    all_labels.append(labels)

            avg_val_loss = np.mean(avg_val_loss)
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            val_accuracy = self.accuracy(all_preds, all_labels)
            val_f1 = self.f1_score(all_preds, all_labels)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            
            if avg_val_loss < best_val_loss:
                counter = 0
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.3f} | Validation Loss: {avg_val_loss:.3f} | Validation Accuracy: {val_accuracy:.3f} | Validation F1 Score: {val_f1:.3f}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
        
        df = pd.DataFrame({
            "train": train_losses[:len(val_losses)],  
            "val": val_losses,
            "val_accuracy": val_accuracies[:len(val_losses)],
            "val_f1_score": val_f1_scores[:len(val_losses)]  
        })
        plt.plot(df.index + 1, df["train"], label="train loss")
        plt.plot(df.index + 1, df["val"], label="val loss")
        plt.plot(df.index + 1, df["val_accuracy"], label="val accuracy")
        plt.plot(df.index + 1, df["val_f1_score"], label="val f1 score")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title("Loss, Accuracy, and F1 Score over epochs")
        plt.legend()
        plt.savefig("metrics_curve.png")
        plt.show()
        
        if best_model_state is not None:
            torch.save(best_model_state, "best_model.pth")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    trainer = Head_Tail_Training(path=path,
                                 tokenizer=tokenizer,
                                 device=device)
    trainer.train(num_epochs=10, lr=2e-5)
