from fuzzywuzzy import fuzz
import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def specialty_fuzzy_match(df_csv, df_excel):
    """
    Applies fuzzy matching to the specialty columns between two dataframes (CSV and Excel data).

    Params:
    - df_csv (DataFrame): The first dataframe containing the 'specialty_1_label' column.
    - df_excel (DataFrame): The second dataframe containing the 'SPCLTY1' column.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing fuzzy matching scores between 'specialty_1_label' and 'SPCLTY1'.
    """
    merged_df = pd.merge(
        df_csv,
        df_excel[["VID", "SPCLTY1"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )

    merged_df["specialty_similarity"] = merged_df.apply(
        lambda row: fuzz.token_sort_ratio(
            str(row["specialty_1_label"]), str(row["SPCLTY1"])
        ),
        axis=1,
    )

    return merged_df[["vid__v", "specialty_1_label", "SPCLTY1", "specialty_similarity"]]


def specialty_fuzzy_match_single(record, df_target):
    """
    Applies fuzzy matching to the specialty of a single record against all rows in the target dataframe.

    Params:
    - record (Series): A single record containing the 'SPCLTY1' field.
    - df_target (DataFrame): Target dataframe containing the 'specialty_1_label' field.

    Returns:
    - result_df (DataFrame): DataFrame containing fuzzy matching scores for specialties.
    """
    specialty_record = str(record["SPCLTY1"])

    df_target = df_target.copy()
    df_target["specialty_similarity"] = df_target["specialty_1_label"].apply(
        lambda x: fuzz.token_sort_ratio(specialty_record, str(x))
    )

    return df_target[["vid__v", "specialty_1_label", "specialty_similarity"]]


if __name__ == "__main__":
    merged_df_result = specialty_fuzzy_match(csv_matched.head(5), matched_excel.head(5))
    print(merged_df_result.head())
