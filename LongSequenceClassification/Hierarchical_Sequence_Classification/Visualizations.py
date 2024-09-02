import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def process_document(text, tokenizer, model, chunk_size=510, pooling_strategy="mean"):
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
    
    if tokens.size(0) == 0:  # If no tokens are generated
        tokens = torch.tensor([tokenizer.unk_token_id])  # Use UNK token as placeholder

    # Split the tokens into chunks
    chunks = [tokens[i:i + (chunk_size - 2)] for i in range(0, len(tokens), chunk_size - 2)]
    
    padded_chunks = []
    for chunk in chunks:
        chunk = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),  # CLS token at the start
            chunk,
            torch.tensor([tokenizer.sep_token_id])  # SEP token at the end
        ])

        # Padding if needed
        padding_length = chunk_size - chunk.size(0)
        if padding_length > 0:
            chunk = torch.cat((chunk, torch.full((padding_length,), tokenizer.pad_token_id)))

        padded_chunks.append(chunk)

    input_ids = torch.stack(padded_chunks).to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    # Pass each chunk through the model
    with torch.no_grad():
        cls_embeddings = []
        for i in range(input_ids.size(0)):
            outputs = model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
            cls_embeddings.append(outputs.last_hidden_state[:, 0, :])  # CLS token embedding

    cls_embeddings = torch.cat(cls_embeddings, dim=0)

    # Pooling
    if pooling_strategy == "mean":
        document_representation = torch.mean(cls_embeddings, dim=0)
    elif pooling_strategy == "max":
        document_representation = torch.max(cls_embeddings, dim=0)[0]
    elif pooling_strategy == "self_attention":
        attn_weights = torch.softmax(torch.mm(cls_embeddings, cls_embeddings.transpose(0, 1)), dim=-1)
        document_representation = torch.mm(attn_weights, cls_embeddings).mean(0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    return document_representation.cpu().numpy()

def extract_embeddings(df, tokenizer, model, pooling_strategy="mean"):
    embeddings = []
    for text in tqdm(df["x"].tolist(), desc="Processing documents"):
        embedding = process_document(text, tokenizer, model, pooling_strategy=pooling_strategy)
        embeddings.append(embedding)
    return np.array(embeddings)

def apply_umap(embeddings, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    embedding_umap = reducer.fit_transform(embeddings)
    return embedding_umap

def plot_embeddings_2d(embedding_umap_2d, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_umap_2d[:, 0], 
        y=embedding_umap_2d[:, 1], 
        hue=labels, 
        palette="viridis",
        s=100,
        alpha=0.8
    )
    plt.title("2D UMAP of Document Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.show()

def plot_embeddings_3d(embedding_umap_3d, labels):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        embedding_umap_3d[:, 0], 
        embedding_umap_3d[:, 1], 
        embedding_umap_3d[:, 2], 
        c=labels, 
        cmap='viridis', 
        s=100,
        alpha=0.8
    )
    ax.set_title("3D UMAP of Document Embeddings")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":


    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract embeddings
    df_embeddings = extract_embeddings(df_short, tokenizer, model, pooling_strategy="mean")

    # Apply UMAP for 2D and 3D visualizations
    embedding_umap_2d = apply_umap(df_embeddings, n_components=2)
    embedding_umap_3d = apply_umap(df_embeddings, n_components=3)

    encoded_labels = df_short["y"]

    # Plot 2D and 3D embeddings
    plot_embeddings_2d(embedding_umap_2d, encoded_labels)
    plot_embeddings_3d(embedding_umap_3d, encoded_labels)

