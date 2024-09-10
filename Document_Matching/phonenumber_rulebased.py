import pandas as pd
import re
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def standardize_phone_number(phone):
    """
    Standardizes a phone number by removing non-numeric characters and ensuring consistent formatting.

    Params:
    - phone (str): The phone number to standardize.

    Returns:
    - str: The standardized phone number.
    """
    # Remove non-numeric characters
    phone = re.sub(r"\D", "", phone)

    # Handle leading country codes (for instance, replace '00' or '+' with nothing)
    if phone.startswith("00"):
        phone = phone[2:]
    elif phone.startswith("0"):
        phone = phone[1:]

    return phone


def phone_number_rule_based(df_csv, df_excel):
    """
    Rule-based matching of phone numbers between systems.

    Params:
    - df_csv (DataFrame): CSV data containing 'phone_1__v'.
    - df_excel (DataFrame): Excel data containing 'PHONE'.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing phone number match score.
    """
    df_excel = df_excel.copy()
    df_csv = df_csv.copy()

    # Standardize phone numbers using the custom standardization function
    df_excel["PHONE"] = df_excel["PHONE"].apply(
        lambda x: standardize_phone_number(str(x))
    )
    df_csv["phone_1__v"] = df_csv["phone_1__v"].apply(
        lambda x: standardize_phone_number(str(x))
    )

    # Merge and compute exact match
    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "PHONE"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )
    merged_df["phone_match"] = merged_df.apply(
        lambda row: 100 if row["phone_1__v"] == row["PHONE"] else 0, axis=1
    )

    return merged_df[["vid__v", "phone_1__v", "PHONE", "phone_match"]]


if __name__ == "__main__":
    print(
        phone_number_rule_based(csv_matched, matched_excel)[
            "phone_match"
        ].value_counts()
    )
