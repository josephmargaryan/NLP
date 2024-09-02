import pandas as pd
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModel
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
    df = df[df["text"].apply(lambda x: len(str(x)) > 10)]

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
    def __init__(self, texts, labels, tokenizer, model, device, chunk_size=510, pooling_strategy="mean"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.pooling_strategy = pooling_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        if tokens.size(0) == 0:  # If no tokens are generated
            tokens = torch.tensor([self.tokenizer.unk_token_id])  # Use UNK token as placeholder

        # Split the tokens into chunks
        chunks = [tokens[i:i + (self.chunk_size - 2)] for i in range(0, len(tokens), self.chunk_size - 2)]
        
        padded_chunks = []
        for chunk in chunks:
            chunk = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),  # CLS token at the start
                chunk,
                torch.tensor([self.tokenizer.sep_token_id])  # SEP token at the end
            ])

            # Padding if needed
            padding_length = self.chunk_size - chunk.size(0)
            if padding_length > 0:
                chunk = torch.cat((chunk, torch.full((padding_length,), self.tokenizer.pad_token_id)))

            padded_chunks.append(chunk)

        input_ids = torch.stack(padded_chunks).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)

        # Pass each chunk through the model
        with torch.no_grad():
            cls_embeddings = []
            for i in range(input_ids.size(0)):
                outputs = self.model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
                cls_embeddings.append(outputs.last_hidden_state[:, 0, :])  # CLS token embedding

        cls_embeddings = torch.cat(cls_embeddings, dim=0)

        # Pooling
        if self.pooling_strategy == "mean":
            document_representation = torch.mean(cls_embeddings, dim=0)
        elif self.pooling_strategy == "max":
            document_representation = torch.max(cls_embeddings, dim=0)[0]
        elif self.pooling_strategy == "self_attention":
            attn_weights = torch.softmax(torch.mm(cls_embeddings, cls_embeddings.transpose(0, 1)), dim=-1)
            document_representation = torch.mm(attn_weights, cls_embeddings).mean(0)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return document_representation, label


def create_dataloader(df, tokenizer, model, device, batch_size=16, chunk_size=510, pooling_strategy="mean", shuffle=True):
    dataset = HierarchicalDataset(
        texts=df["x"].tolist(),
        labels=df["y"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_size=chunk_size,
        pooling_strategy=pooling_strategy
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class DocumentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(DocumentClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, document_representation):
        logits = self.classifier(document_representation)
        return logits

def train(model,
          num_epochs,
          train_loader,
          val_loader,
          lr,
          weight_decay,
          patience,
          device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3)
    scaler = torch.amp.GradScaler()
    train_losses = []
    val_losses = []
    val_accuracies = []  # Track val accuracies
    val_f1_scores = []  # Track val F1 scores
    best_model_state = None
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = []
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Backpropagating epoch {epoch+1}")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            avg_train_loss.append(loss.item())
        avg_train_loss = np.mean(avg_train_loss)
        train_losses.append(avg_train_loss)  # Save training loss

        avg_val_loss = []
        all_preds = []
        all_labels = []
        model.eval()
        for j, (x, y) in enumerate(tqdm(val_loader, desc=f"Forward pass {epoch+1}")):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(x)
                    loss = criterion(output, y)
                    avg_val_loss.append(loss.item())
                    all_preds.append(torch.argmax(output, dim=1))  # Convert probabilities to class labels
                    all_labels.append(y)

        avg_val_loss = np.mean(avg_val_loss)
        val_losses.append(avg_val_loss)  # Save validation loss

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        val_f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        val_accuracies.append(val_accuracy)  # Save validation accuracy
        val_f1_scores.append(val_f1)  # Save validation F1 score

        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        tqdm.write(f"Epoch {epoch+1} | val loss {avg_val_loss:.3f}")
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    if best_model_state is not None:
        torch.save(best_model_state, "best_model.pth")

    # Plotting metrics
    df = pd.DataFrame({
        "train": train_losses,  
        "val": val_losses,
        "val_accuracy": val_accuracies,
        "val_f1_score": val_f1_scores  
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



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare the dataset
    train_df, val_df, num_classes = get_data("/path/to/data")

    # Initialize the tokenizer and the plain BERT model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
    
    # Initialize the document classifier model
    hidden_size = bert_model.config.hidden_size  # Get the hidden size from the BERT model config
    classifier = DocumentClassifier(hidden_size=hidden_size, num_labels=num_classes).to(device)
    
    # Create dataloaders
    train_loader = create_dataloader(train_df, tokenizer, bert_model, device, batch_size=16, pooling_strategy="mean", shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, bert_model, device, batch_size=16, pooling_strategy="mean", shuffle=False)

    # Train the model
    train(classifier,
          num_epochs=100,
          train_loader=train_loader,
          val_loader=val_loader,
          lr=1e-4,
          weight_decay=1e-5,
          patience=3,
          device=device
          )


