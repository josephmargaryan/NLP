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
import seaborn as sns
import matplotlib.pyplot as plt


def get_data(path):
    df = pd.read_csv(path).head(100)


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

    df['x'] = df['review'].apply(clean_text)

    le = LabelEncoder()
    df["y"] = le.fit_transform(df["sentiment"])
    num_classes = len(le.classes_)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    train_df,  val_df = train_test_split(df, test_size=0.2)
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

def compute_cosine_similarity(representations):
    """
    Compute cosine similarity between all pairs of document representations.
    
    Parameters:
    - representations: A tensor of shape (N, hidden_size) where N is the number of documents.
    
    Returns:
    - similarity_matrix: A tensor of shape (N, N) containing the pairwise cosine similarities.
    """
    # Normalize the representations along the hidden_size dimension
    normalized_representations = F.normalize(representations, p=2, dim=1)
    
    # Compute the cosine similarity matrix (N x N)
    similarity_matrix = torch.mm(normalized_representations, normalized_representations.t())
    
    return similarity_matrix

def get_document_representations(dataloader, device):
    """
    Generate document representations using the HierarchicalDataset and a pretrained transformer model.
    
    Parameters:
    - dataloader: DataLoader for the dataset.
    - device: Device to run the model on (CPU or GPU).
    
    Returns:
    - representations: A tensor of shape (N, hidden_size) containing the document representations.
    """
    all_representations = []
    
    for document_representation, _ in tqdm(dataloader, desc="Generating document representations"):
        document_representation = document_representation.to(device)
        all_representations.append(document_representation)
    
    # Concatenate all document representations to form a tensor of shape (N, hidden_size)
    all_representations = torch.cat(all_representations, dim=0)
    
    return all_representations

# Assume other parts of your code are already in place

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare the dataset
    train_df, val_df, num_classes = get_data("/home/jmar/IMDB Dataset.csv")

    # Initialize the tokenizer and the plain BERT model
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    bert_model = AutoModel.from_pretrained("google/mobilebert-uncased").to(device)
    
    # Create dataloaders
    val_loader = create_dataloader(val_df, tokenizer, bert_model, device, batch_size=16, pooling_strategy="mean", shuffle=False)

    # Get document representations for the validation set
    document_representations = get_document_representations(val_loader, device)
    
    # Compute cosine similarity between all document representations
    document_list = val_df["x"].tolist()  
    similarity_matrix = compute_cosine_similarity(document_representations)
    
    # Print or save the similarity matrix
    print(similarity_matrix)
    torch.save(similarity_matrix, "similarity_matrix.pt")
    np.fill_diagonal(similarity_matrix.cpu().numpy(), -np.inf)  # Fill diagonal with -inf to ignore self-similarity
    max_sim_indices = np.unravel_index(np.argmax(similarity_matrix.cpu().numpy()), similarity_matrix.shape)

    doc1 = document_list[max_sim_indices[0]]
    doc2 = document_list[max_sim_indices[1]]
    print(f"The most similar documents are:\nDocument 1: {doc1}\nDocument 2: {doc2}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.cpu().numpy(), cmap="coolwarm", annot=False)
    plt.title("Document Cosine Similarity Matrix")
    plt.show()

