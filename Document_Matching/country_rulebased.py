import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def country_rule_based(df_csv, df_excel):
    """
    Rule-based matching of countries between systems.

    Params:
    - df_csv (DataFrame): CSV data containing 'country__v'.
    - df_excel (DataFrame): Excel data containing 'COUNTRY'.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing country match score.
    """
    df_excel = df_excel.copy()
    df_csv = df_csv.copy()

    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "COUNTRY"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )
    merged_df["country_match"] = merged_df.apply(
        lambda row: 100 if row["country__v"] == row["COUNTRY"] else 0, axis=1
    )

    return merged_df[["vid__v", "country__v", "COUNTRY", "country_match"]]


if __name__ == "__main__":
    print(country_rule_based(csv_matched, matched_excel))
