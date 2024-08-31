import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pickle
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

class Preparation:
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.vocab = None

    def load_data(self):
        dataframes = []
        for folder in tqdm(os.listdir(self.path), desc="Processing folders {folder}"):
            folder_path = os.path.join(self.path, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".parquet"):
                    file_name = os.path.join(folder_path, file)
                    df = pd.read_parquet(file_name)
                    dataframes.append(df)
        final_df = pd.concat(dataframes, axis=0)
        final_df = final_df.dropna(subset=["type__v", "subtype__v", "classification__v"])
        final_df = final_df.loc[~final_df["text"].isna()]
        
        return final_df

    def preprocess_data(self):
        df = self.load_data()

        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
            tokens = text.split()
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

        df['x'] = df['text'].apply(clean_text)
        return df
    
    def labelencode(self, max_count=2000, min_count=250):
        print(f"Beginning Label Encoding")
 
        df = self.preprocess_data()

        class_counts = df["classification__v"].value_counts()

        filtered_classes = class_counts[class_counts >= min_count].index
        df.loc[~df["classification__v"].isin(filtered_classes), "classification__v"] = "UNKNOWN"

        class_counts = df["classification__v"].value_counts()
        
        balanced_dfs = []

        for class_name, count in class_counts.items():
            class_df = df[df["classification__v"] == class_name]
            
            if count > max_count:
                class_df = class_df.sample(max_count, random_state=42)

            balanced_dfs.append(class_df)

        balanced_df = pd.concat(balanced_dfs)
 
        le = LabelEncoder()
        balanced_df["y"] = le.fit_transform(balanced_df["classification__v"])
        num_classes = len(le.classes_)
        print(f"num_classes = {num_classes}")

        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        return balanced_df

    def tokenize_text(self, df):
        tokenized_corpus = df['x'].apply(lambda text: text.split()).tolist()
        return tokenized_corpus

    def build_vocab(self, tokenized_corpus, min_freq=2):
        tokens = [token for sublist in tokenized_corpus for token in sublist]
        token_counts = Counter(tokens)
        
        vocab = {token: idx for idx, (token, count) in enumerate(token_counts.items()) if count >= min_freq}
        
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        
        self.vocab = vocab
        return vocab

    def tokens_to_indices(self, tokenized_corpus):
        if self.vocab is None:
            raise ValueError("Vocabulary is not built yet. Call build_vocab first.")
        return [[self.vocab.get(token, self.vocab['<UNK>']) for token in tokens] for tokens in tokenized_corpus]

    def create_embedding_model(self, tokenized_corpus, embed_size=100, window=5, min_count=1, workers=4):
        model = Word2Vec(sentences=tokenized_corpus, vector_size=embed_size, window=window, min_count=min_count, workers=workers)
        return model

    def prepare_data(self):
        df = self.labelencode()

        tokenized_corpus = self.tokenize_text(df)

        vocab = self.build_vocab(tokenized_corpus, min_freq=2)

        indexed_corpus = self.tokens_to_indices(tokenized_corpus)

        max_len = max(len(seq) for seq in indexed_corpus) 
        X_tensor = torch.zeros((len(indexed_corpus), max_len), dtype=torch.long)  
        
        for i, seq in enumerate(tqdm(indexed_corpus, desc="Processing sequences", unit="seq")):
            X_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)  

        y_tensor = torch.tensor(df['y'].values, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)

        return dataset, vocab

    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_heads, num_layers, dropout=0.1, max_len=512):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.pos_encoder = nn.Parameter(self._generate_positional_encoding(max_len, embed_size), requires_grad=False)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _generate_positional_encoding(self, max_len, embed_size):
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pos_encoding = torch.zeros(max_len, embed_size)
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        return pos_encoding
    
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

    path = '/data-disk/scraping-output/icon'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prep = Preparation(path, device)
    dataset, vocab = prep.prepare_data()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    vocab_size = len(vocab)
    embed_size = 128  
    num_classes = len(set(dataset.tensors[1].numpy()))  
    num_heads = 8  
    num_layers = 4 

    model = TransformerModel(vocab_size, embed_size, num_classes, num_heads, num_layers)

    num_epochs = 10
    lr = 2e-5
    weight_decay = 0.01
    patience = 3

    trained_model = train(
        model=model, 
        num_epochs=num_epochs, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        lr=lr, 
        weight_decay=weight_decay, 
        patience=patience, 
        device=device
    )
