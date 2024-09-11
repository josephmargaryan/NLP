import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def prepare_data(path="/home/jmar/matching_project/data"):
    """
    Preprocesses the data by loading, cleaning, and standardizing CSV and Excel files from the specified directory.

    Params:
        - path (str): The directory path where all the CSV and Excel files are located.

    Preprocessing steps:
        - Removes columns with more than 90% NaN values.
        - Subsets the data to retain only records that have corresponding IDs in both data sources.
        - Splits the data into records with matches and no matches.
        - Sorts the data by IDs for easier comparison of matched records.

    Returns:
        - matched_excel (DataFrame): DataFrame containing matched records from the Excel files.
        - csv_matched (DataFrame): DataFrame containing matched records from the CSV files.
        - unmatched_excel (DataFrame): DataFrame containing unmatched records from the Excel files.
        - csv_unmatched (DataFrame): DataFrame containing unmatched records from the CSV files.
    """

    dataframes_excell = []
    dataframes = []
    for file in tqdm(os.listdir(path), desc="Iterating through folder"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file), low_memory=False)
            dataframes.append(df)
        if file.endswith(".xlsx"):
            xls = pd.read_excel(os.path.join(path, file))
            dataframes_excell.append(xls)

    df_excel = pd.concat(dataframes_excell, axis=0)
    df_csv = pd.concat(dataframes, axis=0)

    df_excel["MATCH_CATEGORY_HCP"] = (
        df_excel["MATCH_CATEGORY_HCP"].str.lower().str.strip()
    )
    df_excel["VID"] = df_excel["VID"].str.replace("v_", "")
    df_csv["vid__v"] = df_csv["vid__v"].astype("str")

    excel_matched = df_excel[df_excel["MATCH_CATEGORY_HCP"] == "match manual"]
    csv_matched = df_csv[df_csv["vid__v"].isin(excel_matched["VID"])]

    matched_excel = excel_matched[excel_matched["VID"].isin(csv_matched["vid__v"])]

    excel_unmatched = df_excel[df_excel["MATCH_CATEGORY_HCP"] == "unmatch manual"]
    csv_unmatched = df_csv[df_csv["vid__v"].isin(excel_unmatched["VID"])]

    unmatched_excel = excel_unmatched[
        excel_unmatched["VID"].isin(csv_unmatched["vid__v"])
    ]

    matched_excel = matched_excel.sort_values(by="VID").reset_index(drop=True)
    csv_matched = csv_matched.sort_values(by="vid__v").reset_index(drop=True)
    unmatched_excel = unmatched_excel.sort_values(by="VID").reset_index(drop=True)
    csv_unmatched = csv_unmatched.sort_values(by="vid__v").reset_index(drop=True)

    columns_with_nan = matched_excel.isna().sum()
    columns_above_3000_nan = columns_with_nan[columns_with_nan > 2905].index.tolist()

    csv_matched = csv_matched.drop(
        columns=[
            "national_id",
            "birth_year__v",
            "hcp.candidate_record__v",
            "data_privacy_opt_out__v",
            "email_3__v",
            "email_2__v",
            "specialty_5_label",
            "specialty_4_label",
            "specialty_3_label",
            "unique_name",
            "middle_name__v",
        ],
        axis=1,
    )

    matched_excel = matched_excel.drop(columns=columns_above_3000_nan, axis=1)

    column_to_move = "VID"
    first_column = matched_excel.pop(column_to_move)  # Remove the column and save it
    matched_excel.insert(0, column_to_move, first_column)

    matched_excel["y_true"] = 1
    csv_matched["y_true"] = 1
    unmatched_excel["y_true"] = 0
    csv_unmatched["y_true"] = 0

    return matched_excel, csv_matched, unmatched_excel, csv_unmatched


matched_excel, csv_matched, unmatched_excel, csv_unmatched = prepare_data()
