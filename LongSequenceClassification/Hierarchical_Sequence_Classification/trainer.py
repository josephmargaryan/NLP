import pandas as pd
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader


def get_data(path, max_count=2000, min_count=250):
    dataframes = []
    for folder in tqdm(os.listdir(path), desc="Iterating through root"):
        folder_path = os.path.join(path, folder)
        for file in tqdm(os.listdir(folder_path), desc="Iterating through folder"):
            if file.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(folder_path, file))
                dataframes.append(df)
    df = pd.concat(dataframes, axis=0)
    df = df.dropna(subset=["type__v", "subtype__v", "classification__v"])
    df = df.loc[~df["text"].isna()]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
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

    df['x'] = df['text'].apply(clean_text)
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

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    train_df,  val_df = train_test_split(balanced_df, test_size=0.2)
    return train_df, val_df, num_classes



class HierarchicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, chunk_size=510, pooling_strategy="mean"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.pooling_strategy = pooling_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        if len(tokens) == 0:
            # Handling case where text tokenization might return an empty sequence
            tokens = torch.tensor([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id])

        chunks = [tokens[i:i + self.chunk_size] for i in range(0, len(tokens), self.chunk_size)]

        # Ensure that each chunk is of the correct length
        for i, chunk in enumerate(chunks):
            if len(chunk) < self.chunk_size:
                padding_length = self.chunk_size - len(chunk)
                chunks[i] = torch.cat((chunk, torch.full((padding_length,), self.tokenizer.pad_token_id)))

        # If no chunks were created, create a single chunk with CLS and SEP tokens
        if len(chunks) == 0:
            chunks = [torch.tensor([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id])]

        # Add [CLS] token at the beginning of each chunk
        chunks = [torch.cat((torch.tensor([self.tokenizer.cls_token_id]), chunk)) for chunk in chunks]
        
        # Now, stack the chunks
        input_ids = torch.stack(chunks)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        if self.pooling_strategy == "mean":
            input_ids = torch.mean(input_ids.float(), dim=0).long() 
            attention_mask = torch.mean(attention_mask.float(), dim=0).long()
        elif self.pooling_strategy == "max":
            input_ids = torch.max(input_ids, dim=0)[0]  
            attention_mask = torch.max(attention_mask, dim=0)[0]
        elif self.pooling_strategy == "self_attention":
            # Implement self-attention pooling if needed
            pass

        return input_ids, attention_mask, label


def create_dataloader(df, tokenizer, batch_size=16, max_len=512, chunk_size=510, pooling_strategy="mean", shuffle=True):
    dataset = HierarchicalDataset(
        texts=df["x"].tolist(),
        labels=df["y"].tolist(),
        tokenizer=tokenizer,
        max_len=max_len,
        chunk_size=chunk_size,
        pooling_strategy=pooling_strategy
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def train(model,
          num_epochs,
          train_loader,
          val_loader,
          lr,
          patience,
          device):
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    counter = 0
    best_model_state = None
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = []
        for i, (indices, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Backpropagating: Epoch {epoch+1}")):
            indices, attention_mask, labels = indices.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(indices, attention_mask=attention_mask, labels=labels)
                loss = output.loss  # Since you're passing labels during training, the model calculates loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            avg_train_loss.append(loss.item())
        avg_train_loss = np.mean(avg_train_loss)

        avg_val_loss = []  # Initialize as a list at the start of validation
        all_preds = []
        all_labels = []
        model.eval()
        for j, (indices, attention_mask, labels) in enumerate(tqdm(val_loader, desc=f"Forward pass Epoch {epoch+1}")):
            indices, attention_mask, labels = indices.to(device), attention_mask.to(device), labels.to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(indices, attention_mask=attention_mask)
                    logits = output.logits  # Extract the logits from the output
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    avg_val_loss.append(loss.item())  
                    all_preds.append(logits)
                    all_labels.append(labels)

        avg_val_loss = np.mean(avg_val_loss)  

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.argmax(dim=1).cpu().numpy())
        f1 = f1_score(all_labels.cpu().numpy(), all_preds.argmax(dim=1).cpu().numpy(), average='weighted')
        
        if avg_val_loss < best_val_loss:
            counter = 0
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
        
        scheduler.step()
        tqdm.write(f"Epoch {epoch+1} | Val loss: {avg_val_loss:.3f}")

        val_losses.append(avg_val_loss)  
        train_losses.append(avg_train_loss)
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)

        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        if best_model_state is not None:
            torch.save(best_model_state, "best_model.pth")

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
    print(f"Final Validation F1 Score: {val_f1_scores[-1]:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, val_df, num_classes = get_data("/data-disk/scraping-output/icon")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_classes).to(device)
    train_loader = create_dataloader(train_df, tokenizer, batch_size=16, pooling_strategy="mean", shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, batch_size=16, pooling_strategy="mean", shuffle=False)

    train(model,
          100,
          train_loader,
          val_loader,
          1e-4,
          3,
          device)

