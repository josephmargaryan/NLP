import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D

class HeadTailEmbeddingVisualizer:
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

    def generate_embeddings(self, df):
        df = self.preprocess_data(df)
        embeddings = []

        for text in tqdm(df['x'], desc="Generating embeddings", unit="text"):
            input_ids, attention_mask = self.tokenize_and_select(text)
            if input_ids is None:
                embeddings.append(None)
                continue

            input_ids = input_ids.unsqueeze(0).to(self.device)  # Add batch dimension
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = output.hidden_states[-1][:, 0, :]  # Extract the embedding from the [CLS] token
                embeddings.append(pooled_output.cpu().numpy().flatten())

        embeddings_np = np.array([e for e in embeddings if e is not None])
        return df, embeddings_np

    def reduce_dimensions(self, embeddings, n_components=2):
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def plot_embeddings(self, df, reduced_embeddings, n_components=2):
        if n_components == 2:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df['y'], palette="deep")
            plt.title("UMAP projection of Document Embeddings (2D)")
            plt.show()
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=df['y'], cmap="viridis")
            ax.set_title("UMAP projection of Document Embeddings (3D)")
            plt.colorbar(scatter)
            plt.show()

    def find_similar_documents(self, embeddings, target_index, top_n=5):
        target_embedding = embeddings[target_index].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, embeddings).flatten()
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
        return similar_indices, similarities[similar_indices]

    def visualize_and_find_similar(self, df, n_components=2, target_index=None, top_n=5):
        df, embeddings = self.generate_embeddings(df)
        reduced_embeddings = self.reduce_dimensions(embeddings, n_components=n_components)
        self.plot_embeddings(df, reduced_embeddings, n_components=n_components)
        
        if target_index is not None:
            similar_indices, similarities = self.find_similar_documents(embeddings, target_index, top_n)
            similar_docs = df.iloc[similar_indices]
            print(f"Similar documents to index {target_index} (similarity scores):")
            for i, (idx, sim) in enumerate(zip(similar_indices, similarities)):
                print(f"Rank {i+1}: Document index {idx}, Similarity: {sim:.4f}")
                print(f"Text: {similar_docs['x'].iloc[i]}\n")
            return similar_docs, similarities


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the HeadTailEmbeddingVisualizer class
    visualizer = HeadTailEmbeddingVisualizer(
        model_path="best_model.pth",  # Path to your trained model's state_dict
        tokenizer_name="google-bert/bert-base-multilingual-cased",  # The same tokenizer you used for training
        device=device
    )

    # Load your data
    df = pd.read_csv("path/to/data.csv")  # Path to your data with 'x' and 'y' columns

    # Visualize embeddings and find similar documents
    visualizer.visualize_and_find_similar(df, n_components=2, target_index=0, top_n=5)

