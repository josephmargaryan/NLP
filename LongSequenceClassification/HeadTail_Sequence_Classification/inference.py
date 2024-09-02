import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pickle

class ModelInference:
    def __init__(self, model_path, tokenizer_name, label_encoder_path, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased").to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to inference mode
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def preprocess_and_tokenize(self, df, text_column='text', max_length=512, first_tokens=128, last_tokens=382):
        def clean_text(text):
            import re
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        df['clean_text'] = df[text_column].apply(clean_text)

        def tokenize_and_select(text):
            tokens = self.tokenizer(text, add_special_tokens=False)
            input_ids = tokens['input_ids']

            if len(input_ids) > (max_length - 2):
                input_ids = input_ids[:first_tokens] + input_ids[-last_tokens:]
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

            attention_mask = [1] * len(input_ids)

            if len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length

            return torch.tensor(input_ids), torch.tensor(attention_mask)

        df['tokens'] = df['clean_text'].apply(tokenize_and_select)
        df[['input_ids', 'attention_mask']] = pd.DataFrame(df['tokens'].tolist(), index=df.index)
        df.drop(columns=['tokens', 'clean_text'], inplace=True)
        return df

    def predict(self, df):
        input_ids = torch.stack(df['input_ids'].tolist()).to(self.device)
        attention_mask = torch.stack(df['attention_mask'].tolist()).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        predicted_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())
        return predicted_labels

# Example usage
if __name__ == "__main__":
    df_to_infer = pd.DataFrame({'text': ["This is a sample review text.", "Another sample review!"]})
    model_path = '/kaggle/working/best_model.pth'
    tokenizer_name = "google-bert/bert-base-uncased"
    label_encoder_path = '/kaggle/working/label_encoder.pkl'
    
    inferencer = ModelInference(model_path=model_path, tokenizer_name=tokenizer_name, label_encoder_path=label_encoder_path, device='cuda')
    df_to_infer = inferencer.preprocess_and_tokenize(df_to_infer)
    predictions = inferencer.predict(df_to_infer)
    print(predictions)
