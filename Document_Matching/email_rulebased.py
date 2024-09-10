from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched
import pandas as pd


def email_rule_based(df_csv, df_excel):
    """
    Rule-based matching of emails between systems.

    Params:
    - df_csv (DataFrame): CSV data containing 'email_1__v'.
    - df_excel (DataFrame): Excel data containing 'EMAIL'.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing email match score.
    """
    df_excel = df_excel.copy()
    df_csv = df_csv.copy()

    # Standardize email addresses by stripping whitespace and converting to lowercase
    df_excel["EMAIL"] = df_excel["EMAIL"].apply(lambda x: str(x).strip().lower())
    df_csv["email_1__v"] = df_csv["email_1__v"].apply(lambda x: str(x).strip().lower())

    # Merge and compute exact match
    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "EMAIL"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )
    merged_df["email_match"] = merged_df.apply(
        lambda row: 100 if row["email_1__v"] == row["EMAIL"] else 0, axis=1
    )

    return merged_df[["vid__v", "email_1__v", "EMAIL", "email_match"]]


if __name__ == "__main__":
    # Perform the email matching
    result = email_rule_based(csv_matched, matched_excel)
    print(result.head()["email_match"])
