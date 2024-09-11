import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched

"""
Doesnt work as it yields a KEY VALUE ERROR
"""


def type_rule_based(df_csv, df_excel):
    """
    Rule-based matching of healthcare provider types between systems.

    Params:
    - df_csv (DataFrame): CSV data containing 'type'.
    - df_excel (DataFrame): Excel data containing 'HCP_TYPE_V__LABEL'.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing type match score.
    """
    # Define mappings for type labels
    type_mapping = {
        "DOCTOR": "DOCTOR",
        "PHARMACIST": "PHARMACIST",
        "NON_PRESCRIBING_HCP": "OTHER",
        "RESIDENT": "DOCTOR",
        "NURSE_MIDWIFE": "NURSE_MIDWIFE",
        "DENTIST": "OTHER",
        "OTHER": "OTHER",
    }

    df_excel = df_excel.copy()
    df_csv = df_csv.copy()

    df_excel["HCP_TYPE_V__LABEL_mapped"] = df_excel.loc[:, "HCP_TYPE_V__LABEL"].map(
        type_mapping
    )
    df_csv["type_mapped"] = df_csv["type"].map(type_mapping)

    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "HCP_TYPE_V__LABEL_mapped"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )
    merged_df["type_match"] = merged_df.apply(
        lambda row: 100 if row["type_mapped"] == row["HCP_TYPE_V__LABEL_mapped"] else 0,
        axis=1,
    )

    return merged_df[["vid__v", "type", "HCP_TYPE_V__LABEL", "type_match"]]


if __name__ == "__main__":
    type_rule_based(csv_matched, matched_excel)
