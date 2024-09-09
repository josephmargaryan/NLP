from name_fuzzy_matching import first_name_fuzzy_match, last_name_fuzzy_match
from speciality_fuzzy_matching import specialty_fuzzy_match
from address_embeddings import address_embedding_similarity
from address_fuzzy_matching import standardize_and_fuzzy_match
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched
import pandas as pd


def similarity_engine(df_csv, df_excel):
    """
    Computes multiple similarity metrics (first name, last name, specialty, address) and combines them
    into an overall similarity score, which represents the likelihood of a match.

    Params:
    - df_csv (DataFrame): CSV data containing records.
    - df_excel (DataFrame): Excel data containing records.

    Returns:
    - result_df (DataFrame): DataFrame containing similarity scores and an overall match score.
    """

    first_name_similarity_df = first_name_fuzzy_match(df_csv, df_excel)
    last_name_similarity_df = last_name_fuzzy_match(df_csv, df_excel)
    specialty_similarity_df = specialty_fuzzy_match(df_csv, df_excel)
    address_fuzzy_similarity_df = standardize_and_fuzzy_match(df_csv, df_excel)
    address_embedding_similarity_df = address_embedding_similarity(df_csv, df_excel)

    merged_df = pd.merge(
        first_name_similarity_df[["vid__v", "first_name_similarity"]],
        last_name_similarity_df[["vid__v", "last_name_similarity"]],
        on="vid__v",
    )

    merged_df = pd.merge(
        merged_df,
        specialty_similarity_df[["vid__v", "specialty_similarity"]],
        on="vid__v",
    )
    merged_df = pd.merge(
        merged_df,
        address_fuzzy_similarity_df[["vid__v", "address_similarity"]],
        on="vid__v",
    )
    merged_df = pd.merge(
        merged_df,
        address_embedding_similarity_df[["vid__v", "cosine_similarity"]],
        on="vid__v",
    )

    weights = {
        "first_name_similarity": 0.3,  # 30%
        "last_name_similarity": 0.2,  # 20%
        "specialty_similarity": 0.2,  # 20%
        "address_similarity": 0.15,  # 15%
        "cosine_similarity": 0.15,  # 15% We can play around wit hthe weights, assigning 100% for phone number maybe
    }

    merged_df["overall_similarity"] = (
        merged_df["first_name_similarity"] * weights["first_name_similarity"]
        + merged_df["last_name_similarity"] * weights["last_name_similarity"]
        + merged_df["specialty_similarity"] * weights["specialty_similarity"]
        + merged_df["address_similarity"] * weights["address_similarity"]
        + merged_df["cosine_similarity"] * weights["cosine_similarity"]
    )

    match_threshold = 70
    merged_df["is_match"] = merged_df["overall_similarity"] >= match_threshold

    return merged_df[["vid__v", "overall_similarity", "is_match"]]


if __name__ == "__main__":
    result_df = similarity_engine(csv_matched.head(), matched_excel.head())
    print(result_df.head())
