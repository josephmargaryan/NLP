import pandas as pd
import re

class Preprocessing:
    def __init__(self, source_filename, target_filename, nan_threshold,
                 source_id_column, target_id_column, source_address_columns=None,
                 target_address_columns=None, source_email_column=None,
                 target_email_column=None, source_phone_column=None,
                 target_phone_column=None, source_last_name_column=None,
                 target_last_name_column=None, source_first_name_column=None,
                 target_first_name_column=None, source_type_column=None,
                 target_type_column=None, source_subtype_column=None,
                 target_subtype_column=None):

        # Store parameters
        self.source_filename = source_filename
        self.target_filename = target_filename
        self.nan_threshold = nan_threshold
        self.source_id_column = source_id_column
        self.target_id_column = target_id_column
        self.source_address_columns = source_address_columns or []
        self.target_address_columns = target_address_columns or []
        self.source_email_column = source_email_column
        self.target_email_column = target_email_column
        self.source_phone_column = source_phone_column
        self.target_phone_column = target_phone_column
        self.source_last_name_column = source_last_name_column
        self.target_last_name_column = target_last_name_column
        self.source_first_name_column = source_first_name_column
        self.target_first_name_column = target_first_name_column
        self.source_type_column = source_type_column
        self.target_type_column = target_type_column
        self.source_subtype_column = source_subtype_column
        self.target_subtype_column = target_subtype_column

    def load_data(self):
        source_df = pd.read_excel(self.source_filename)
        target_df = pd.read_excel(self.target_filename)
        return source_df, target_df

    def rename_id_columns(self, df1, df2):
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

    def clean_and_convert_id_column(self, df, id_column="id"):
        if df[id_column].dtype == "object":
            df[id_column] = df[id_column].str.replace(r"\D", "", regex=True)
        df[id_column] = df[id_column].astype(str)
        return df

    def standardize_address(self, df, address_columns):
        if address_columns:
            df["cleaned_address"] = df[address_columns].apply(
                lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1
            )
            df["cleaned_address"] = df["cleaned_address"].apply(self.post_process_address)
        return df

    def post_process_address(self, address):
        """Standardize common abbreviations in the address string."""
        replacements = {
            r'\bsq\b': 'square',
            r'\bst\b': 'street',
            r'\brd\b': 'road',
            r'\bave\b': 'avenue',
            r'\bblvd\b': 'boulevard'
        }
        for pattern, replacement in replacements.items():
            address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
        return address

    def preprocess_email_column(self, df, email_column):
        if email_column in df.columns:
            df["cleaned_email"] = df[email_column].apply(
                lambda x: str(x).strip().lower() if pd.notnull(x) else x
            )
        return df

    def standardize_phone_number(self, phone, is_target=False):
        if pd.isna(phone):
            return None
        if not is_target:
            return int(phone) if isinstance(phone, float) else None
        if isinstance(phone, str):
            phone = re.sub(r"[^\d]", "", phone)
            if phone.startswith("353"):
                phone = phone[3:]
            return int(phone) if phone else None
        return None

    def standardize_phone_number_column(self, df, phone_column, is_target=False):
        if phone_column in df.columns:
            df["cleaned_phonenumber"] = df[phone_column].apply(self.standardize_phone_number, is_target=is_target)
        return df

    def standardize_last_name(self, df, last_name_column):
        if last_name_column in df.columns:
            df["cleaned_last_name"] = df[last_name_column].apply(
                lambda x: x.split()[-1].lower() if isinstance(x, str) else x
            )
        return df

    def standardize_first_name(self, df, first_name_column):
        if first_name_column in df.columns:
            df["cleaned_first_name"] = df[first_name_column].str.lower().str.strip()
        return df

    def post_process_city(self, city):
        """Remove numbers from city names."""
        if isinstance(city, str):
            return re.sub(r'\d+', '', city).strip()
        return city

    def standardize_city(self, df, city_column):
        if city_column in df.columns:
            df["cleaned_city"] = df[city_column].apply(self.post_process_city)
        return df

    def clean_type_and_subtype(self, df, type_column, subtype_column):
        if type_column in df.columns:
            df["cleaned_type"] = df[type_column].str.lower().str.strip()
        if subtype_column in df.columns:
            df["cleaned_subtype"] = df[subtype_column].str.lower().str.strip()
        return df

    def filter_transformed_columns(self, df, transformed_columns):
        columns_in_df = [col for col in transformed_columns if col in df.columns]
        return df[columns_in_df]

    def process_data(self):
        source_df, target_df = self.load_data()
        source_df, target_df = self.rename_id_columns(source_df, target_df)
        source_df = self.remove_high_nan_columns_and_clean_id(source_df)
        target_df = self.remove_high_nan_columns_and_clean_id(target_df)

        source_df = self.clean_and_convert_id_column(source_df)
        target_df = self.clean_and_convert_id_column(target_df)

        source_df = self.standardize_address(source_df, self.source_address_columns)
        target_df = self.standardize_address(target_df, self.target_address_columns)

        source_df = self.preprocess_email_column(source_df, self.source_email_column)
        target_df = self.preprocess_email_column(target_df, self.target_email_column)

        source_df = self.standardize_phone_number_column(source_df, self.source_phone_column, is_target=False)
        target_df = self.standardize_phone_number_column(target_df, self.target_phone_column, is_target=True)

        source_df = self.standardize_last_name(source_df, self.source_last_name_column)
        target_df = self.standardize_last_name(target_df, self.target_last_name_column)

        source_df = self.standardize_first_name(source_df, self.source_first_name_column)
        target_df = self.standardize_first_name(target_df, self.target_first_name_column)

        source_df = self.standardize_city(source_df, "CITY")
        target_df = self.standardize_city(target_df, "address.hcp_address_city (HCP_ADDRESS_CITY)")

        source_df = self.clean_type_and_subtype(source_df, self.source_type_column, self.source_subtype_column)
        target_df = self.clean_type_and_subtype(target_df, self.target_type_column, self.target_subtype_column)

        transformed_columns = [
            "id", "cleaned_address", "cleaned_email", "cleaned_phonenumber",
            "cleaned_last_name", "cleaned_first_name", "cleaned_type", "cleaned_subtype"
        ]

        source_df = self.filter_transformed_columns(source_df, transformed_columns)
        target_df = self.filter_transformed_columns(target_df, transformed_columns)

        return source_df, target_df

# Example Usage
source_path = "/root/similarity_matching/data/IE_HCP_NMF_BASE.xlsx"
target_path = "/root/similarity_matching/data/VOD_IE_Extract_for_Base_20241023.xlsx"
preprocessor = Preprocessing(
    source_filename=source_path,
    target_filename=target_path,
    nan_threshold=0.99,
    source_id_column="RCRD_NMBR",
    target_id_column="hcp.hcp_vid (HCP_VID)",
    source_address_columns=["ADDR1", "CITY"],
    target_address_columns=["address.hcp_address_line1 (HCP_ADDRESS_LINE1)", "address.hcp_address_city (HCP_ADDRESS_CITY)"],
    source_email_column="EMAIL",
    target_email_column="hcp.hcp_email (HCP_EMAIL)",
    source_phone_column="PHONE",
    target_phone_column="address.hcp_address_phone (HCP_ADDRESS_PHONE)",
    source_last_name_column="LAST_NAME",
    target_last_name_column="hcp.last_name (LAST_NAME)",
    source_first_name_column="FIRST_NAME",
    target_first_name_column="hcp.first_name (FIRST_NAME)",
    source_type_column="HCP_TYPE_V__LABEL",
    target_type_column="hcp.hcp_type (HCP_TYPE)",
    source_subtype_column="SPCLTY1",
    target_subtype_column="hcp.specialty1 (SPECIALTY1)"
)

source_df, target_df = preprocessor.process_data()
source_df.to_csv("source_df.csv", index=False)
target_df.to_csv("target_df.csv", index=False)

