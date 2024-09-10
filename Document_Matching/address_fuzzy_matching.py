from fuzzywuzzy import fuzz
import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def standardize_and_fuzzy_match(df_csv, df_excel):
    """
    Standardizes address fields for both CSV and Excel dataframes,
    concatenates them, and applies fuzzy matching on the addresses.

    Params:
    - df_csv (DataFrame): CSV data containing address components.
    - df_excel (DataFrame): Excel data containing address components.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing fuzzy matching scores for addresses.
    """
    df_excel = df_excel.copy()
    df_excel["address_full"] = df_excel[
        ["ADDR1", "CITY", "ADMIN_AREA", "POSTALCODE", "COUNTRY"]
    ].apply(lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1)
    df_csv = df_csv.copy()
    df_csv["address_full"] = df_csv[
        [
            "address_line_1__v",
            "address_line_2__v",
            "locality__v",
            "postal_code__v",
            "country__v",
        ]
    ].apply(lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1)

    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "address_full"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )

    merged_df["address_similarity"] = merged_df.apply(
        lambda row: fuzz.token_sort_ratio(row["address_full_x"], row["address_full_y"]),
        axis=1,
    )
    return merged_df[
        ["vid__v", "address_full_x", "address_full_y", "address_similarity"]
    ]


if __name__ == "__main__":
    merged_df_result = standardize_and_fuzzy_match(
        csv_matched.head(5), matched_excel.head(5)
    )
    print(merged_df_result.head())
