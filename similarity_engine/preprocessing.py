import os
import pandas as pd
import yaml
import re

class Preprocessing:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.source_filename = self.config["source_filename"]  # Full path
        self.target_filename = self.config["target_filename"]  # Full path
        self.nan_threshold = self.config["nan_threshold"]
        self.source_id_column = self.config["source_id_column"]
        self.target_id_column = self.config["target_id_column"]

    def load_data(self):
        # Load source and target files directly using their paths
        source_df = self.load_file_by_extension(self.source_filename)
        target_df = self.load_file_by_extension(self.target_filename)
        return source_df, target_df

    def load_file_by_extension(self, file_path):
        return pd.read_excel(file_path)

    def rename_id_columns(self, df1, df2):
        # Rename the ID columns in both DataFrames based on the configuration
        if self.source_id_column in df1.columns:
            df1 = df1.rename(columns={self.source_id_column: "id"})
        else:
            raise KeyError(f"Source ID column '{self.source_id_column}' not found in source file")

        if self.target_id_column in df2.columns:
            df2 = df2.rename(columns={self.target_id_column: "id"})
        else:
            raise KeyError(f"Target ID column '{self.target_id_column}' not found in target file")

        return df1, df2

    def remove_high_nan_columns_and_clean_id(self, df):
        nan_proportion = df.isna().mean()
        columns_to_drop = nan_proportion[nan_proportion > self.nan_threshold].index.tolist()
        df = df.drop(columns=columns_to_drop)
        df = df.dropna(subset=["id"]) if "id" in df.columns else df
        return df

    def clean_and_convert_id_column(self, df, id_column):
        if df[id_column].dtype == "object":
            df[id_column] = df[id_column].str.replace(r"\D", "", regex=True)
        df[id_column] = df[id_column].astype(str)
        return df

    def standardize_address(self, df, address_columns):
        if address_columns:  # Only process if columns are provided
            df["cleaned_address"] = df[address_columns].apply(
                lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1
            )
        return df

    def preprocess_email_column(self, df, email_column):
        if email_column and email_column in df.columns:
            df["cleaned_email"] = df[email_column].apply(
                lambda x: str(x).strip().lower() if pd.notnull(x) else x
            )
        return df

    def standardize_phone_number(self, phone, is_target=False):
        if pd.isna(phone):
            return None
        
        # Handle source phone numbers (as floats)
        if not is_target:
            return int(phone) if isinstance(phone, float) else None
        
        # Handle target phone numbers (as strings with prefix)
        if is_target and isinstance(phone, str):
            phone = re.sub(r"[^\d]", "", phone)  # Remove all non-digit characters
            if phone.startswith("353"):  # Assuming the international prefix is +353
                phone = phone[3:]  # Remove the international prefix
            return int(phone) if phone else None
        
        return None

    def standardize_phone_number_column(self, df, phone_column, is_target=False):
        if phone_column and phone_column in df.columns:
            df["cleaned_phonenumber"] = df[phone_column].apply(self.standardize_phone_number, is_target=is_target)
        return df

    def standardize_last_name(self, df, last_name_column):
        if last_name_column and last_name_column in df.columns:
            df["cleaned_last_name"] = df[last_name_column].apply(
                lambda x: x.split()[-1].lower() if isinstance(x, str) else x
            )
        return df

    def standardize_first_name(self, df, first_name_column):
        if first_name_column and first_name_column in df.columns:
            df["cleaned_first_name"] = df[first_name_column].str.lower().str.strip()
        return df

    def filter_transformed_columns(self, df, transformed_columns):
        # Return only the transformed columns, ensuring they exist
        columns_in_df = [col for col in transformed_columns if col in df.columns]
        return df[columns_in_df]

    def process_data(self):
        source_df, target_df = self.load_data()

        # Preprocess each DataFrame
        source_df, target_df = self.rename_id_columns(source_df, target_df)
        source_df = self.remove_high_nan_columns_and_clean_id(source_df)
        target_df = self.remove_high_nan_columns_and_clean_id(target_df)

        source_df = self.clean_and_convert_id_column(source_df, "id")
        target_df = self.clean_and_convert_id_column(target_df, "id")

        # Process and transform addresses, emails, phone numbers, and names
        if self.config.get("source_address_columns"):
            source_df = self.standardize_address(source_df, self.config["source_address_columns"])
        if self.config.get("target_address_columns"):
            target_df = self.standardize_address(target_df, self.config["target_address_columns"])

        if self.config.get("source_df_email"):
            source_df = self.preprocess_email_column(source_df, self.config["source_df_email"])
        if self.config.get("target_df_email"):
            target_df = self.preprocess_email_column(target_df, self.config["target_df_email"])

        if self.config.get("source_data_phonenumber"):
            source_df = self.standardize_phone_number_column(source_df, self.config["source_data_phonenumber"], is_target=False)
        if self.config.get("target_data_phonenumber"):
            target_df = self.standardize_phone_number_column(target_df, self.config["target_data_phonenumber"], is_target=True)

        if self.config.get("source_df_last_name"):
            source_df = self.standardize_last_name(source_df, self.config["source_df_last_name"])
        if self.config.get("target_df_last_name"):
            target_df = self.standardize_last_name(target_df, self.config["target_df_last_name"])

        if self.config.get("source_df_first_name"):
            source_df = self.standardize_first_name(source_df, self.config["source_df_first_name"])
        if self.config.get("target_df_first_name"):
            target_df = self.standardize_first_name(target_df, self.config["target_df_first_name"])

        # Define the transformed columns that we want to keep
        transformed_columns = ["id", "cleaned_address", "cleaned_email", "cleaned_phonenumber", "cleaned_last_name", "cleaned_first_name"]

        # Filter only the transformed columns
        source_df = self.filter_transformed_columns(source_df, transformed_columns)
        target_df = self.filter_transformed_columns(target_df, transformed_columns)

        # Merge the source and target dataframes on 'id'
        # merged_df = pd.merge(source_df, target_df, on="id", how="inner")

        return source_df, target_df


# Example Usage
if __name__ == "__main__":
    preprocessing = Preprocessing("/root/similarity_matching/yml_templates/preprocess.yml")
    source_df, target_df = preprocessing.process_data()

    # Save to CSV for checking the output
    source_df.to_csv("source_df.csv", index=False)
    target_df.to_csv("target_df.csv", index=False)
