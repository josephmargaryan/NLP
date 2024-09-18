import pandas as pd
import os
import re
from tqdm import tqdm
import yaml


class Preprocessing:
    def __init__(self, config_path):
        # Load configuration from the YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.source_data_extension = self.config["source_data_extension"]
        self.target_data_extension = self.config["target_data_extension"]
        self.nan_threshold = self.config["nan_threshold"]
        self.source_id_column = self.config["source_id_column"]
        self.target_id_column = self.config["target_id_column"]
        self.data_path = self.config["data_path"]

    def load_data(self):
        dataframes_source = []
        dataframes_target = []

        for file in tqdm(os.listdir(self.data_path), desc="Iterating through folder"):
            if file.endswith(self.source_data_extension):
                df_source = self.load_file_by_extension(
                    os.path.join(self.data_path, file), self.source_data_extension
                )
                dataframes_source.append(df_source)

            elif file.endswith(self.target_data_extension):
                df_target = self.load_file_by_extension(
                    os.path.join(self.data_path, file), self.target_data_extension
                )
                dataframes_target.append(df_target)

        source_df = (
            pd.concat(dataframes_source, axis=0)
            if dataframes_source
            else pd.DataFrame()
        )
        target_df = (
            pd.concat(dataframes_target, axis=0)
            if dataframes_target
            else pd.DataFrame()
        )

        if "MATCH_CATEGORY_HCP" in source_df.columns:
            source_df["MATCH_CATEGORY_HCP"] = (
                source_df["MATCH_CATEGORY_HCP"].str.lower().str.strip()
            )
        if "MATCH_CATEGORY_HCP" in target_df.columns:
            target_df["MATCH_CATEGORY_HCP"] = (
                target_df["MATCH_CATEGORY_HCP"].str.lower().str.strip()
            )

        return source_df, target_df

    def load_file_by_extension(self, file_path, file_extension):
        """
        Helper function to load a file based on its extension
        """
        if file_extension == ".csv":
            return pd.read_csv(file_path, low_memory=False)
        elif file_extension == ".xlsx":
            return pd.read_excel(file_path)
        elif file_extension == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def rename_id_columns(self, df1, df2):
        """
        Renames the ID columns in two dataframes to a common name 'id'.
        """
        if self.source_id_column in df1.columns:
            df1 = df1.rename(columns={self.source_id_column: "id"})
        else:
            print(
                f"Warning: '{self.source_id_column}' column not found in the first dataframe."
            )

        if self.target_id_column in df2.columns:
            df2 = df2.rename(columns={self.target_id_column: "id"})
        else:
            print(
                f"Warning: '{self.target_id_column}' column not found in the second dataframe."
            )

        return df1, df2

    def remove_high_nan_columns_and_clean_id(self, df):
        """
        Removes columns that have a proportion of NaN values greater than the specified threshold
        and removes all rows where the 'id' column contains NaN values.
        """
        # Remove columns with a proportion of NaN values greater than the threshold
        nan_proportion = df.isna().mean()
        columns_to_drop = nan_proportion[
            nan_proportion > self.nan_threshold
        ].index.tolist()

        df = df.drop(columns=columns_to_drop)

        if "id" in df.columns:
            df = df.dropna(subset=["id"])
        else:
            print("Warning: 'id' column not found.")

        return df

    def clean_and_convert_id_column(self, df, id_column):
        """
        Cleans the ID column by handling both string and integer types.
        For strings: removes non-integer characters (like 'v_' prefix).
        For integers: converts them to strings for uniformity.

        Params:
        - df (DataFrame): The dataframe with the ID column to clean.
        - id_column (str): The name of the ID column to process.

        Returns:
        - df (DataFrame): The dataframe with the cleaned 'id' column.
        """
        if df[id_column].dtype == "object":
            # Remove non-integer characters from string IDs (e.g., 'v_' prefix)
            df[id_column] = df[id_column].str.replace(r"\D", "", regex=True)

        df[id_column] = df[id_column].astype(str)

        return df

    def standardize_address(self, df, address_columns):
        """
        Standardizes multiple address columns into one column 'address_full' by combining non-null values.

        Params:
        - df (DataFrame): The dataframe (source or target) to process.
        - address_columns (list): List of address-related columns to combine.

        Returns:
        - df (DataFrame): The dataframe with the new 'address_full' column.
        """
        # Combine non-null values from the specified address columns into 'address_full'
        df["address_full"] = df[address_columns].apply(
            lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1
        )

        return df

    def preprocess_email_column(self, df, email_column):
        """
        Preprocesses the email column by stripping whitespace and converting to lowercase.

        Params:
        - df (DataFrame): The dataframe (source or target) to process.
        - email_column (str): The name of the email column to clean.

        Returns:
        - df (DataFrame): The dataframe with the cleaned email column.
        """
        df["cleaned_email"] = df[email_column].apply(
            lambda x: str(x).strip().lower() if pd.notnull(x) else x
        )
        return df

    def standardize_phone_number(self, phone):
        """
        Standardizes a phone number by removing non-numeric characters and ensuring consistent formatting.

        It retains the country code if present, especially for cases like +20 (Egypt).

        If the phone is None or not a valid string, it returns None.

        Params:
        - phone (str): The phone number to standardize.

        Returns:
        - str: The standardized phone number or None if input is invalid.
        """
        if pd.isna(phone) or not isinstance(phone, str):
            return None

        phone = re.sub(r"[^\d+]", "", phone)

        if phone.startswith("+"):
            phone = phone

        if phone.startswith("0"):
            phone = phone[1:]

        return phone

    def standardize_phone_number_column(self, df, phone_column):
        """
        Standardizes the phone number column by applying the standardize_phone_number function.

        Params:
        - df (DataFrame): The dataframe (source or target) to process.
        - phone_column (str): The name of the phone number column to clean.

        Returns:
        - df (DataFrame): The dataframe with the standardized phone number column.
        """
        df["cleaned_phonenumber"] = df[phone_column].apply(
            self.standardize_phone_number
        )
        return df

    def process_data(self):
        """
        Main function to execute the entire preprocessing pipeline
        """
        source_df, target_df = self.load_data()

        source_df, target_df = self.rename_id_columns(source_df, target_df)

        source_df = self.remove_high_nan_columns_and_clean_id(source_df)
        target_df = self.remove_high_nan_columns_and_clean_id(target_df)

        source_df = self.clean_and_convert_id_column(source_df, "id")
        target_df = self.clean_and_convert_id_column(target_df, "id")

        if self.config.get("source_address_columns"):
            source_address_columns = self.config["source_address_columns"][0]
            source_df = self.standardize_address(source_df, source_address_columns)

        if self.config.get("target_address_columns"):
            target_address_columns = self.config["target_address_columns"]
            target_df = self.standardize_address(target_df, target_address_columns)

        if self.config.get("source_df_email"):
            source_df_email = self.config["source_df_email"]
            source_df = self.preprocess_email_column(source_df, source_df_email)

        if self.config.get("target_df_email"):
            target_df_email = self.config["target_df_email"]
            target_df = self.preprocess_email_column(target_df, target_df_email)

        if self.config.get("source_data_phonenumber"):
            source_phone_column = self.config["source_data_phonenumber"]
            source_df = self.standardize_phone_number_column(
                source_df, source_phone_column
            )

        if self.config.get("target_data_phonenumber"):
            target_phone_column = self.config["target_data_phonenumber"]
            target_df = self.standardize_phone_number_column(
                target_df, target_phone_column
            )

        # Merge source and target dataframes
        merged_df = pd.merge(source_df, target_df, on="id", how="inner")

        return source_df, target_df, merged_df


def get_dataframes(config_path):

    preprocessor = Preprocessing(config_path)
    source_df, target_df, merged_df = preprocessor.process_data()
    return source_df, target_df, merged_df


if __name__ == "__main__":
    source_df, target_df, merged_df = get_dataframes(
        "/home/jmar/matching_project/workspace2/yml_examples/preprocess.yml"
    )
