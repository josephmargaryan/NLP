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
    def __init__(self, model_path, tokenizer_name, device, path_to_data, path_to_le):
        self.model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=206, output_hidden_states=True)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.path_to_data = path_to_data
        self.model.to(self.device)
        self.model.eval()
        with open(path_to_le, "rb") as f:
            self.le = pickle.load(f)

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

        df = pd.concat(dataframes, axis=0) 
        print(f"Length before filtering {len(df)}")
        df.reset_index(drop=True, inplace=True)
        df = df.loc[~df["text"].isna(), :]
        df = df[df["text"].apply(lambda x: len(str(x)) > 10)]
        df = df.loc[~df["CLASSIFICATION"].isna(), :]
        print(f"Number of samples with manual classification {len(df)}")
        df["x"] = df["text"]
        try:
            df["y"] = self.le.transform(df["CLASSIFICATION"])
        except ValueError as e:
            print(f"Handling unseen labels: {e}")
            df["y"] = df["CLASSIFICATION"].apply(lambda x: self.le.transform([x])[0] if x in self.le.classes_ else -1)
    
        
        return df.iloc[0:31, :]
        
    def preprocess_data(self):
        df = self.load_data()
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

        # If the input sequence is shorter than 510 tokens, pad it to the required length
        if input_ids.size(0) <= first_tokens + last_tokens:
            padding_length = (first_tokens + last_tokens) - input_ids.size(0)
            selected_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones(input_ids.size(0), dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)])
        else:
            # Select the first 128 and last 382 tokens
            selected_ids = torch.cat([input_ids[:first_tokens], input_ids[-last_tokens:]])
            attention_mask = torch.ones_like(selected_ids)

        return selected_ids, attention_mask


    def generate_embeddings(self):
        df = self.preprocess_data()
        embeddings = []
        valid_indices = []

        for idx, text in enumerate(tqdm(df['x'], desc="Generating embeddings", unit="text")):
            input_ids, attention_mask = self.tokenize_and_select(text)
            if input_ids is None:
                continue  # Skip this text if it doesn't generate a valid embedding

            valid_indices.append(idx)
            input_ids = input_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = output.hidden_states  # Get all hidden states
                pooled_output = hidden_states[-1][:, 0, :]  # Use the last layer's [CLS] token embedding
                embeddings.append(pooled_output.cpu().numpy().flatten())

        embeddings_np = np.array(embeddings)
        filtered_df = df.iloc[valid_indices].reset_index(drop=True)  # Keep only rows with valid embeddings
        return filtered_df, embeddings_np


    def reduce_dimensions(self, embeddings, n_components=2):
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def plot_embeddings(self, reduced_embeddings, n_components=2):
        filtered_df, _ = self.generate_embeddings()
        
        # Use inverse_transform to get the label names
        label_names = self.le.inverse_transform(filtered_df['y'])

        if n_components == 2:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=label_names, palette="deep")
            plt.title("UMAP projection of Document Embeddings (2D)")
            plt.show()
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=label_names, cmap="viridis")
            ax.set_title("UMAP projection of Document Embeddings (3D)")
            plt.colorbar(scatter)
            plt.show()


    def find_similar_documents(self, embeddings, target_index, top_n=5):
        target_embedding = embeddings[target_index].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, embeddings).flatten()
        similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
        return similar_indices, similarities[similar_indices]

    def visualize_and_find_similar(self, n_components=2, target_index=None, top_n=5):
        df, embeddings = self.generate_embeddings()
        reduced_embeddings = self.reduce_dimensions(embeddings, n_components=n_components)
        self.plot_embeddings(reduced_embeddings, n_components=n_components)
        
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

    visualizer = HeadTailEmbeddingVisualizer(
        model_path="/home/jmar/Head_Tail_Method/best_model.pth", 
        tokenizer_name="google-bert/bert-base-multilingual-cased",  
        device=device,
        path_to_data="/data-disk/scraping-output/p-drive-structured",
        path_to_le="/home/jmar/Head_Tail_Method/label_encoder.pkl"
    )

    visualizer.visualize_and_find_similar(n_components=2, target_index=0, top_n=5)


