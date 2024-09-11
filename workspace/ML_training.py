import pandas as pd
from name_fuzzy_matching import first_name_fuzzy_match, last_name_fuzzy_match
from speciality_fuzzy_matching import specialty_fuzzy_match
from address_fuzzy_matching import standardize_and_fuzzy_match
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def get_similarity_scores(df_csv, df_excel):
    """
    Computes similarity scores for first name, last name, specialty, and address,
    and returns a dataframe containing the scores.

    Params:
    - df_csv (DataFrame): The first dataframe containing the CSV data.
    - df_excel (DataFrame): The second dataframe containing the Excel data.

    Returns:
    - merged_df (DataFrame): DataFrame containing the similarity scores.
    """
    # Compute similarity scores
    first_name_similarity_df = first_name_fuzzy_match(df_csv, df_excel)
    last_name_similarity_df = last_name_fuzzy_match(df_csv, df_excel)
    specialty_similarity_df = specialty_fuzzy_match(df_csv, df_excel)
    address_similarity_df = standardize_and_fuzzy_match(df_csv, df_excel)

    # Extract the similarity scores and merge them into a single dataframe
    merged_df = pd.merge(
        first_name_similarity_df[["first_name_similarity"]],
        last_name_similarity_df[["last_name_similarity"]],
        left_index=True,
        right_index=True,
    )
    merged_df = pd.merge(
        merged_df,
        specialty_similarity_df[["specialty_similarity"]],
        left_index=True,
        right_index=True,
    )
    merged_df = pd.merge(
        merged_df,
        address_similarity_df[["address_similarity"]],
        left_index=True,
        right_index=True,
    )

    return merged_df


# Extract similarity scores for the matched data (target = 1)
matched_similarity_scores = get_similarity_scores(csv_matched, matched_excel)
matched_similarity_scores["y_true"] = 1  # Add target column for matched data

# Extract similarity scores for the unmatched data (target = 0)
unmatched_similarity_scores = get_similarity_scores(csv_unmatched, unmatched_excel)
unmatched_similarity_scores["y_true"] = 0  # Add target column for unmatched data

# Concatenate matched and unmatched data
final_df = pd.concat([matched_similarity_scores, unmatched_similarity_scores])

# Now, 'final_df' contains the similarity scores as features and 'y_true' as the target variable.
print(final_df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features and target
X = final_df.drop(columns=["y_true"])
y = final_df["y_true"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
