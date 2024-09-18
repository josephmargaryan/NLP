import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from tqdm import tqdm
import yaml
import numpy as np
import os


class SimilarityEngine:
    def __init__(self, config_path, output_folder="results"):
        # Load YAML config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Extract settings from config
        self.submodules = self.config.get("submodules", {})
        self.weights = self.config.get("weights", {})
        self.match_threshold = self.config.get("match_threshold", 70)
        self.model_name = self.config.get("model_name", "bert-base-nli-mean-tokens")
        self.columns = self.config.get("columns", {})
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Load model for embeddings if cosine similarity is needed
        self.model = (
            SentenceTransformer(self.model_name)
            if self.submodules.get("use_address_cosine", False)
            else None
        )

    def first_name_fuzzy_match(self, record, df_target):
        """
        Fuzzy matching for first names.
        """
        df_target = df_target.copy()
        first_name_record = record[self.columns["first_name"]["source"]]
        df_target["first_name_similarity"] = df_target[
            self.columns["first_name"]["target"]
        ].apply(
            lambda x: (
                fuzz.token_sort_ratio(str(first_name_record), str(x))
                if pd.notna(x)
                else 0
            )
        )
        return df_target[[self.columns["id"]["target"], "first_name_similarity"]]

    def last_name_fuzzy_match(self, record, df_target):
        """
        Fuzzy matching for last names.
        """
        df_target = df_target.copy()
        last_name_record = record[self.columns["last_name"]["source"]]
        df_target["last_name_similarity"] = df_target[
            self.columns["last_name"]["target"]
        ].apply(
            lambda x: (
                fuzz.token_sort_ratio(str(last_name_record), str(x))
                if pd.notna(x)
                else 0
            )
        )
        return df_target[[self.columns["id"]["target"], "last_name_similarity"]]

    def subtype_fuzzy_match(self, record, df_target):
        """
        Fuzzy matching for subtypes (specialty).
        """
        df_target = df_target.copy()
        subtype_record = record[self.columns["subtype"]["source"]]
        df_target["subtype_similarity"] = df_target[
            self.columns["subtype"]["target"]
        ].apply(
            lambda x: (
                fuzz.token_sort_ratio(str(subtype_record), str(x)) if pd.notna(x) else 0
            )
        )
        return df_target[[self.columns["id"]["target"], "subtype_similarity"]]

    def type_fuzzy_match(self, record, df_target):
        """
        Fuzzy matching for types (e.g., healthcare provider type).
        """
        df_target = df_target.copy()
        type_record = record[self.columns["type"]["source"]]
        df_target["type_similarity"] = df_target[self.columns["type"]["target"]].apply(
            lambda x: (
                fuzz.token_sort_ratio(str(type_record), str(x)) if pd.notna(x) else 0
            )
        )
        return df_target[[self.columns["id"]["target"], "type_similarity"]]

    def phone_number_rule_based(self, record, df_target):
        """
        Rule-based matching for phone numbers.
        Immediate 100% match if phone numbers match.
        """
        df_target = df_target.copy()
        phone_record = record[self.columns["phone"]["source"]]
        df_target["phone_match"] = df_target[self.columns["phone"]["target"]].apply(
            lambda x: 100 if str(x) == str(phone_record) else 0
        )
        return df_target[[self.columns["id"]["target"], "phone_match"]]

    def email_rule_based(self, record, df_target):
        """
        Rule-based matching for emails.
        Immediate 100% match if emails match.
        """
        df_target = df_target.copy()
        email_record = record[self.columns["email"]["source"]]
        df_target["email_match"] = df_target[self.columns["email"]["target"]].apply(
            lambda x: 100 if str(x) == str(email_record) else 0
        )
        return df_target[[self.columns["id"]["target"], "email_match"]]

    def address_fuzzy_match(self, record, df_target):
        """
        Fuzzy matching for addresses.
        """
        df_target = df_target.copy()
        address_record = record[self.columns["address"]["source"]]
        df_target["address_fuzzy_similarity"] = df_target[
            self.columns["address"]["target"]
        ].apply(
            lambda x: (
                fuzz.token_sort_ratio(str(address_record), str(x)) if pd.notna(x) else 0
            )
        )
        return df_target[[self.columns["id"]["target"], "address_fuzzy_similarity"]]

    def address_embedding_similarity(self, record, df_target):
        """
        Embedding-based cosine similarity for addresses.
        """
        if self.model is None:
            return df_target[[self.columns["id"]["target"]]].assign(
                address_embedding_similarity=0
            )

        df_target = df_target.copy()  # Make a copy to avoid SettingWithCopyWarning
        address_record = record[self.columns["address"]["source"]]
        address_embedding_record = (
            self.model.encode(str(address_record)) if pd.notna(address_record) else None
        )

        if address_embedding_record is None:
            df_target["address_embedding_similarity"] = 0
        else:
            df_target["address_embedding"] = df_target[
                self.columns["address"]["target"]
            ].apply(lambda x: self.model.encode(str(x)) if pd.notna(x) else None)
            df_target["address_embedding_similarity"] = df_target[
                "address_embedding"
            ].apply(
                lambda x: (
                    cosine_similarity([address_embedding_record], [x])[0][0] * 100
                    if x is not None
                    else 0
                )
            )
        return df_target[[self.columns["id"]["target"], "address_embedding_similarity"]]

    def apply_similarity(self, record, df_target):
        """
        Apply all similarity submodules for a single record against the target dataframe.
        """
        similarity_dfs = []

        if self.submodules.get("use_first_name", False):
            similarity_dfs.append(self.first_name_fuzzy_match(record, df_target))
        if self.submodules.get("use_last_name", False):
            similarity_dfs.append(self.last_name_fuzzy_match(record, df_target))
        if self.submodules.get("use_subtype", False):
            similarity_dfs.append(self.subtype_fuzzy_match(record, df_target))
        if self.submodules.get("use_type", False):  # Added type fuzzy match
            similarity_dfs.append(self.type_fuzzy_match(record, df_target))

        if self.submodules.get("use_phone", False):
            similarity_dfs.append(self.phone_number_rule_based(record, df_target))
        if self.submodules.get("use_email", False):
            similarity_dfs.append(self.email_rule_based(record, df_target))

        if self.submodules.get("use_address_fuzzy", False):
            similarity_dfs.append(self.address_fuzzy_match(record, df_target))
        if self.submodules.get("use_address_cosine", False):
            similarity_dfs.append(self.address_embedding_similarity(record, df_target))

        result_df = self.merge_similarity_dfs(similarity_dfs)

        result_df["overall_similarity"] = (
            result_df["first_name_similarity"]
            * self.weights.get("first_name_similarity", 0)
            + result_df["last_name_similarity"]
            * self.weights.get("last_name_similarity", 0)
            + result_df["address_fuzzy_similarity"]
            * self.weights.get("address_fuzzy_similarity", 0)
            + result_df["subtype_similarity"]
            * self.weights.get("subtype_similarity", 0)
            + result_df["type_similarity"]
            * self.weights.get("type_similarity", 0)  # Include type similarity
        )

        if self.submodules.get("use_address_cosine", False):
            result_df["overall_similarity"] += result_df[
                "address_embedding_similarity"
            ] * self.weights.get("address_embedding_similarity", 0)

        result_df["rule_based_match"] = (result_df["phone_match"] == 100) | (
            result_df["email_match"] == 100
        )

        result_df["source_id"] = record[self.columns["id"]["source"]]
        result_df["target_id"] = result_df[self.columns["id"]["target"]]

        result_df["match_label"] = result_df.apply(
            lambda row: (
                "Match"
                if row["overall_similarity"] >= self.match_threshold
                or row["rule_based_match"]
                else "No Match"
            ),
            axis=1,
        )

        return result_df

    def merge_similarity_dfs(self, similarity_dfs):
        """
        Merge all the individual similarity dataframes into one.
        """
        merged_df = similarity_dfs[0]
        for df in similarity_dfs[1:]:
            merged_df = pd.merge(
                merged_df, df, on=self.columns["id"]["target"], how="outer"
            )
        return merged_df

    def compare_single_record(self, record, df_target, output_filename="single_record_results.csv"):
        """
        Compare one record from source to all records in the target and save the results as a CSV file.
        """
        similarity_results = self.apply_similarity(record, df_target)

        # Save the results as a CSV file in the output folder
        if not similarity_results.empty:
            output_path = os.path.join(self.output_folder, output_filename)
            similarity_results.to_csv(output_path, index=False)

        return similarity_results

    def compare_multiple_records(
        self, df_source, df_target, output_filename="similarity_results.csv"
    ):
        """
        Compare multiple records from source to target and save the results as a CSV file.
        """
        results = []
        for _, record in tqdm(
            df_source.iterrows(), total=len(df_source), desc="Processing records"
        ):
            result = self.compare_single_record(record, df_target)
            if not result.empty:
                results.append(result)

        similarity_results = pd.concat(results, ignore_index=True)

        # Save the results as a CSV file in the output folder
        output_path = os.path.join(self.output_folder, output_filename)
        similarity_results.to_csv(output_path, index=False)

        return similarity_results


if __name__ == "__main__":
    """    engine = SimilarityEngine(config_path="/path/to/engine_config.yml")
        source = pd.read_csv("/path/to/Source_Data.csv")
        target = pd.read_csv("/path/to/Target_Data.csv")

        # Compare a single record
        record = source.iloc[0, :]
        result = engine.compare_single_record(record, target.copy())
        print(result)

        # Compare multiple records
        results = engine.compare_multiple_records(source.head(10).copy(), target.copy())
        print(results)"""
