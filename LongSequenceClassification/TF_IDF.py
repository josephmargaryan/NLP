from xgboost import XGBClassifier 
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
import os 
from tqdm import tqdm 
import re 
import pickle
from sklearn.model_selection import train_test_split


"""params = {'eta': 0.15814222503371855, 
            'max_depth': 9, 
           'min_child_weight': 4, 
           'gamma': 0.25716412943434713, 
           'subsample': 0.6612824933317458, 
           'colsample_bytree': 0.910079245627798, 
           'lambda': 0.43406902868323993, 
           'alpha': 0.17335310745516538}

model = XGBClassifier(**params)"""

# Initialize vectorizer and label encoder
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 5))
le = LabelEncoder()

def get_data(icon_folder="/data-disk/scraping-output/icon", max_count=2500, min_count=250):
    dataframes = []
    
    # Iterate through the directories and files
    for folder in tqdm(os.listdir(icon_folder), desc="P-drive folders"):
        folder_path = os.path.join(icon_folder, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".parquet"):
                df = pd.read_parquet(file_path)
                dataframes.append(df)
    
    # Concatenate dataframes
    df = pd.concat(dataframes, axis=0)
    df = df.dropna(subset=["type__v", "subtype__v", "classification__v"])
    df = df.loc[~df["text"].isna()]
    df = df[df["text"].apply(lambda x: len(str(x)) > 10)]

    # Clean text function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove emojis
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

    # Apply text cleaning
    df['x'] = df['text'].apply(clean_text)

    # Filter based on class distribution
    class_counts = df["classification__v"].value_counts()
    filtered_classes = class_counts[class_counts >= min_count].index
    df.loc[~df["classification__v"].isin(filtered_classes), "classification__v"] = "UNKNOWN"
    class_counts = df["classification__v"].value_counts()

    # Balance classes
    balanced_dfs = []
    for class_name, count in class_counts.items():
        class_df = df[df["classification__v"] == class_name]
        if count > max_count:
            class_df = class_df.sample(max_count, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs)

    # Encode labels
    le = LabelEncoder()
    balanced_df["y"] = le.fit_transform(balanced_df["classification__v"])
    num_classes = len(le.classes_)

    print("Starting vectorizer and label encoder")
    
    # Save LabelEncoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Vectorize text
    y = balanced_df["y"]
    X = vec.fit_transform(balanced_df["x"]).toarray()

    # Save vectorizer
    with open("vec.pkl", "wb") as f:
        pickle.dump(vec, f)

    print("Finished vectorizer and label encoder")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Create a DataFrame for training data
    training_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    training_data['y_train'] = y_train

    # Create a DataFrame for testing data
    testing_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    testing_data['y_test'] = y_test

    # Save the training and testing data to CSV files
    training_data.to_csv("training_data.csv", index=False)
    testing_data.to_csv("testing_data.csv", index=False)

    print("Training and testing data saved successfully.")
    
    return num_classes

# Run the function
num_classes = get_data()
print(f"Number of classes: {num_classes}")
