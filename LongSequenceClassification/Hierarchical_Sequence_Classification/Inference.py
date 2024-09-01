import torch 
import torch.nn as nn
import re 
import pickle
from transformers import AutoTokenizer, AutoModel


class DocumentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(DocumentClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, document_representation):
        logits = self.classifier(document_representation)
        return logits
    

def load_model(model_path, hidden_size, num_classes, device):
    """Load the saved classifier model."""
    model = DocumentClassifier(hidden_size=hidden_size, num_labels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_text(text, tokenizer, bert_model, device, chunk_size=510, pooling_strategy="mean"):
    """Tokenize and preprocess the text into document representation."""
    # Clean the text (same as during training)
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
    
    text = clean_text(text)
    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)

    # Split the tokens into chunks of max length (chunk_size - 2)
    chunks = [tokens[i:i + (chunk_size - 2)] for i in range(0, len(tokens), chunk_size - 2)]

    # Add CLS and SEP tokens and pad to chunk_size
    padded_chunks = []
    for chunk in chunks:
        chunk = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),
            chunk,
            torch.tensor([tokenizer.sep_token_id])
        ])

        # Padding if needed
        padding_length = chunk_size - chunk.size(0)
        if padding_length > 0:
            chunk = torch.cat((chunk, torch.full((padding_length,), tokenizer.pad_token_id)))

        padded_chunks.append(chunk)

    input_ids = torch.stack(padded_chunks).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # Pass each chunk through BERT separately
    with torch.no_grad():
        cls_embeddings = []
        for i in range(input_ids.size(0)):
            output = bert_model(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
            cls_embeddings.append(output.last_hidden_state[:, 0, :])  # Extract the [CLS] token embeddings

    cls_embeddings = torch.cat(cls_embeddings, dim=0)  # Shape: (num_chunks, hidden_size)

    # Apply the pooling strategy to combine chunk representations
    if pooling_strategy == "mean":
        document_representation = torch.mean(cls_embeddings, dim=0)  # Shape: (hidden_size)
    elif pooling_strategy == "max":
        document_representation = torch.max(cls_embeddings, dim=0)[0]  # Shape: (hidden_size)
    elif pooling_strategy == "self_attention":
        # Simple self-attention pooling implementation
        attn_weights = torch.softmax(torch.bmm(cls_embeddings.unsqueeze(0), cls_embeddings.unsqueeze(0).transpose(1, 2)).squeeze(0), dim=-1)
        document_representation = torch.bmm(attn_weights.unsqueeze(0), cls_embeddings.unsqueeze(0)).squeeze(0).mean(dim=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    return document_representation

def predict(texts, classifier_model, tokenizer, bert_model, device, label_encoder_path, chunk_size=510, pooling_strategy="mean"):
    """Predict the class for a list of texts."""
    # Load label encoder
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    predictions = []
    for text in texts:
        # Preprocess text
        document_representation = preprocess_text(text, tokenizer, bert_model, device, chunk_size, pooling_strategy)
        document_representation = document_representation.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            logits = classifier_model(document_representation)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        # Convert to label
        predicted_label = le.inverse_transform(pred)[0]
        predictions.append(predicted_label)
    
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the tokenizer and load the models
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
    hidden_size = 768  # This should match the hidden size of the BERT model you're using
    num_classes = 59  # Update this based on your actual number of classes
    classifier_model = load_model("best_model.pth", hidden_size, num_classes, device)
    
    # Example texts for prediction
    new_texts = ["Example text for prediction", "Another example text for classification"]
    
    # Predict labels for new texts
    predictions = predict(new_texts, classifier_model, tokenizer, bert_model, device, "label_encoder.pkl")
    
    for text, prediction in zip(new_texts, predictions):
        print(f"Text: {text}\nPredicted Label: {prediction}\n")

if __name__ == "__main__":
    main()

