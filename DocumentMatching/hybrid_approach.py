from fuzzywuzzy import fuzz

def preprocess_similarity(row):
    # Apply fuzzy matching to string features
    row['name_similarity'] = fuzz.ratio(row['name_1'], row['name_2']) / 100
    row['surname_similarity'] = fuzz.ratio(row['surname_1'], row['surname_2']) / 100
    row['occupation_similarity'] = fuzz.ratio(row['occupation_1'], row['occupation_2']) / 100
    row['phone_similarity'] = 1 if row['phone_1'] == row['phone_2'] else 0

    return row

# Apply preprocessing to the entire dataframe
df = df.apply(preprocess_similarity, axis=1)

# Features for ML: similarity scores for each feature
X = df[['name_similarity', 'surname_similarity', 'occupation_similarity', 'phone_similarity']]
y = df['is_match']  # Target variable indicating a match

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

# Evaluate the classification performance
print(classification_report(y_test, y_pred))
