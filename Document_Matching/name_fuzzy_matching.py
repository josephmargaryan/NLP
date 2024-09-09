from fuzzywuzzy import fuzz
import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def first_name_fuzzy_match(df_csv, df_excel):
    """
    Applies fuzzy matching to the first name columns between two dataframes (CSV and Excel data).

    Params:
    - df_csv (DataFrame): The first dataframe containing the 'first_name__v' column.
    - df_excel (DataFrame): The second dataframe containing the 'FIRST_NAME' column.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing fuzzy matching scores between 'first_name__v' and 'FIRST_NAME'.
    """
    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "FIRST_NAME"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )

    merged_df["first_name_similarity"] = merged_df.apply(
        lambda row: fuzz.token_sort_ratio(
            str(row["FIRST_NAME"]), str(row["first_name__v"])
        ),
        axis=1,
    )

    return merged_df[["vid__v", "FIRST_NAME", "first_name__v", "first_name_similarity"]]


def last_name_fuzzy_match(df_csv, df_excel):
    """
    Applies fuzzy matching to the last name columns between two dataframes (CSV and Excel data).

    Params:
    - df_csv (DataFrame): The first dataframe containing the 'last_name__v' column.
    - df_excel (DataFrame): The second dataframe containing the 'LAST_NAME' column.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing fuzzy matching scores between 'last_name__v' and 'LAST_NAME'.
    """
    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "LAST_NAME"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )

    merged_df["last_name_similarity"] = merged_df.apply(
        lambda row: fuzz.token_sort_ratio(
            str(row["LAST_NAME"]), str(row["last_name__v"])
        ),
        axis=1,
    )

    return merged_df[["vid__v", "LAST_NAME", "last_name__v", "last_name_similarity"]]


if __name__ == "__main__":
    first_name_similarity_df = first_name_fuzzy_match(csv_matched, matched_excel)
    print(first_name_similarity_df.head())

    last_name_similarity_df = last_name_fuzzy_match(csv_matched, matched_excel)
    print(last_name_similarity_df.head())
