import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class HeadTailInference:
    def __init__(self, model_path, tokenizer_name, device, le_path, path_to_data):
        self.model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=206)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.path_to_data = path_to_data
        self.model.to(self.device)
        self.model.eval()

        with open (le_path, "rb") as f:
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
        
        return df.iloc[0:3, :]
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
        
        if input_ids.size(0) <= (first_tokens + last_tokens):
            selected_ids = input_ids  
        else:
            selected_ids = torch.cat([input_ids[:first_tokens], input_ids[-last_tokens:]])

        attention_mask = torch.ones_like(selected_ids)

        if selected_ids.size(0) < (first_tokens + last_tokens):
            padding_length = (first_tokens + last_tokens) - selected_ids.size(0)
            selected_ids = torch.cat([selected_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        return selected_ids, attention_mask

    def infer(self, text):
        input_ids, attention_mask = self.tokenize_and_select(text)

        if input_ids is None or attention_mask is None:
            return None, None  

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).cpu().item()
            pred_prob = torch.max(probs, dim=-1).values.cpu().item()

        return pred_label, pred_prob

    def run_inference(self, batch_size=16):
        df = self.preprocess_data()
        preds = []
        probs = []
        
        all_input_ids = []
        all_attention_masks = []
        
        for text in tqdm(df['x'], desc="Tokenizing for inference", unit="text"):
            input_ids, attention_mask = self.tokenize_and_select(text)
            if input_ids is not None and attention_mask is not None:
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
        
        # Batch processing
        dataset = torch.utils.data.TensorDataset(torch.stack(all_input_ids), torch.stack(all_attention_masks))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for batch in tqdm(dataloader, desc="Running inference", unit="batch"):
            input_ids, attention_masks = [x.to(self.device) for x in batch]

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_masks)
                logits = output.logits
                probs_batch = F.softmax(logits, dim=-1)
                pred_labels_batch = torch.argmax(probs_batch, dim=-1).cpu().numpy()
                pred_probs_batch = torch.max(probs_batch, dim=-1).values.cpu().numpy()
                
                preds.extend(pred_labels_batch)
                probs.extend(pred_probs_batch)

        df['pred_label'] = self.le.inverse_transform(preds)
        df['pred_prob'] = probs

        return df


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference = HeadTailInference(
        model_path="/home/jmar/Head_Tail_Method/best_model.pth",  
        tokenizer_name="google-bert/bert-base-multilingual-cased", 
        device=device,
        le_path="/home/jmar/Head_Tail_Method/label_encoder.pkl",
        path_to_data="/data-disk/scraping-output/p-drive-structured"
    )

    df = inference.run_inference()

    # Run inference
    df_with_predictions = inference.run_inference(df)

    # Save the resulting dataframe
    df_with_predictions.to_csv("inference_results.csv", index=False)
