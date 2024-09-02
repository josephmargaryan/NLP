import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle

class DocumentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(DocumentClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, document_representation):
        logits = self.classifier(document_representation)
        return logits

class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, model, device, chunk_size=510, pooling_strategy="mean"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.pooling_strategy = pooling_strategy

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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

        # Pass each chunk through the transformer model to get CLS embeddings
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

        return document_representation
    
def create_dataloader(df, tokenizer, model, device, batch_size=16, chunk_size=510, pooling_strategy="mean", shuffle=False):
    dataset = HierarchicalDataset(
        texts=df["x"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        chunk_size=chunk_size,
        pooling_strategy=pooling_strategy
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def inference(transformer_model, classifier_model, tokenizer, df, device, label_encoder_path, chunk_size=510, pooling_strategy="mean", batch_size=16):
    """
    Perform inference on a new dataset using the transformer model to extract features and a trained classifier model.
    
    Parameters:
    - transformer_model: The pretrained transformer model to extract CLS embeddings.
    - classifier_model: The trained DocumentClassifier model.
    - tokenizer: The tokenizer used to preprocess the input text.
    - df: A DataFrame containing the text data in df["x"].
    - device: The device to run the model on (CPU or GPU).
    - label_encoder_path: Path to a label encoder's pickle file.
    - chunk_size: The maximum size of chunks for tokenization.
    - pooling_strategy: The strategy to pool the chunk representations ('mean', 'max', 'self_attention').
    - batch_size: The batch size for inference.
    
    Returns:
    - predictions: A list of predicted class labels.
    """
    transformer_model.to(device)
    classifier_model.to(device)
    
    transformer_model.eval()
    classifier_model.eval()
    
    # Load the label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Create DataLoader for inference
    dataloader = create_dataloader(
        df=df,
        tokenizer=tokenizer,
        model=transformer_model,  # Pass the pretrained transformer model here
        device=device,
        chunk_size=chunk_size,
        pooling_strategy=pooling_strategy,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_preds = []
    for batch in tqdm(dataloader, desc="Performing inference"):
        document_representations = batch.to(device)  # Get document representations (not raw inputs)
        with torch.no_grad():
            logits = classifier_model(document_representations)  # Pass document representations to classifier
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # Convert predicted indices back to original labels
    predicted_labels = label_encoder.inverse_transform(all_preds)
    
    return predicted_labels

# Assuming all the other classes and functions (DocumentClassifier, HierarchicalDataset, etc.) remain the same

if __name__ == "__main__":
    # Load your data (replace with your actual data loading code)
    # df = pd.read_csv('your_inference_data.csv')  # Replace with the correct path to your dataset
    
    # Initialize tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    transformer_model = AutoModel.from_pretrained('bert-base-uncased')
    classifier_model = DocumentClassifier(hidden_size=768, num_labels=2)  # Adjust num_labels according to your model
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the classifier model state dict and move to the correct device
    classifier_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    classifier_model.to(device)  # Ensure the classifier model is on the correct device
    
    # Perform inference
    predictions = inference(transformer_model, classifier_model, tokenizer, df_short, device=device, label_encoder_path="/home/jmar/ML_testing/self_attention/label_encoder.pkl")
    
    # Add predictions to the DataFrame
    df_short["predicted_labels"] = predictions
    
    # Save the results to a CSV file
    df_short.to_csv("inference_results.csv", index=False)
    
    print("Inference completed. Results saved to 'inference_results.csv'.")


