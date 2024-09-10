from name_fuzzy_matching import first_name_fuzzy_match, last_name_fuzzy_match
from speciality_fuzzy_matching import specialty_fuzzy_match
from address_embeddings import address_embedding_similarity
from address_fuzzy_matching import standardize_and_fuzzy_match
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched
from phonenumber_rulebased import phone_number_rule_based
from type_rulebased import type_rule_based
from country_rulebased import country_rule_based
from email_rulebased import email_rule_based
import pandas as pd


def similarity_engine(
    df_csv,
    df_excel,
    use_first_name=True,
    use_last_name=True,
    use_specialty=True,
    use_address_fuzzy=True,
    use_address_cosine=False,
    use_phone=True,
    use_country=True,
    use_email=True,
    match_threshold=70,
):
    """
    Computes configurable similarity metrics and combines them into an overall similarity score,
    which represents the likelihood of a match.

    Params:
    - df_csv (DataFrame): CSV data containing records.
    - df_excel (DataFrame): Excel data containing records.
    - use_first_name (bool): Whether to use first name similarity.
    - use_last_name (bool): Whether to use last name similarity.
    - use_specialty (bool): Whether to use specialty similarity.
    - use_address_fuzzy (bool): Whether to use fuzzy matching for address.
    - use_address_cosine (bool): Whether to use cosine similarity for address.
    - use_phone (bool): Whether to use phone number similarity.
    - use_country (bool): Whether to use country similarity.
    - use_email (bool): Whether to use email similarity.
    - match_threshold (float): Threshold for determining whether a match is valid.

    Returns:
    - result_df (DataFrame): DataFrame containing similarity scores and an overall match score.
    """
    # Initialize an empty list for results
    similarity_dfs = []
    weights = {}

    if use_first_name:
        first_name_similarity_df = first_name_fuzzy_match(df_csv, df_excel)
        similarity_dfs.append(
            first_name_similarity_df[["vid__v", "first_name_similarity"]]
        )
        weights["first_name_similarity"] = 0.2

    if use_last_name:
        last_name_similarity_df = last_name_fuzzy_match(df_csv, df_excel)
        similarity_dfs.append(
            last_name_similarity_df[["vid__v", "last_name_similarity"]]
        )
        weights["last_name_similarity"] = 0.15

    if use_specialty:
        specialty_similarity_df = specialty_fuzzy_match(df_csv, df_excel)
        similarity_dfs.append(
            specialty_similarity_df[["vid__v", "specialty_similarity"]]
        )
        weights["specialty_similarity"] = 0.15

    if use_address_fuzzy:
        address_fuzzy_similarity_df = standardize_and_fuzzy_match(df_csv, df_excel)
        similarity_dfs.append(
            address_fuzzy_similarity_df[["vid__v", "address_similarity"]]
        )
        weights["address_similarity"] = 0.1

    if use_address_cosine:
        address_embedding_similarity_df = address_embedding_similarity(df_csv, df_excel)
        similarity_dfs.append(
            address_embedding_similarity_df[["vid__v", "cosine_similarity"]]
        )
        weights["cosine_similarity"] = 0.1

    if use_phone:
        phone_similarity_df = phone_number_rule_based(df_csv, df_excel)
        similarity_dfs.append(phone_similarity_df[["vid__v", "phone_match"]])
        weights["phone_match"] = 0.2

    if use_country:
        country_similarity_df = country_rule_based(df_csv, df_excel)
        similarity_dfs.append(country_similarity_df[["vid__v", "country_match"]])
        weights["country_match"] = 0.05

    if use_email:
        email_similarity_df = email_rule_based(df_csv, df_excel)
        similarity_dfs.append(email_similarity_df[["vid__v", "email_match"]])
        weights["email_match"] = 0.05

    if similarity_dfs:
        merged_df = similarity_dfs[0]
        for df in similarity_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="vid__v")

        merged_df["overall_similarity"] = sum(
            merged_df[col] * weights[col] for col in weights
        )

        merged_df["is_match"] = merged_df["overall_similarity"] >= match_threshold

        return merged_df[["vid__v", "overall_similarity", "is_match"]]

    return pd.DataFrame(columns=["vid__v", "overall_similarity", "is_match"])


if __name__ == "__main__":
    # Use configurations with caution as it wil crash due to being computationally expensive
    result_df = similarity_engine(
        df_csv=csv_matched.head(),
        df_excel=matched_excel.head(),
        use_first_name=True,
        use_last_name=True,
        use_specialty=False,
        use_address_fuzzy=True,
        use_address_cosine=False,
        use_phone=True,
        use_country=False,
        use_email=False,
        match_threshold=70,
    )
    print(result_df.head())
