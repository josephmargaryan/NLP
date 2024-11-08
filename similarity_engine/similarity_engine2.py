import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

class SimilarityMatcher:
    def __init__(self, source_df, target_df, source_id, target_id,
                 source_columns, target_columns, source_weights,
                 target_weights, source_rule_based, target_rule_based,
                 threshold, feature_threshold, output_file="results.parquet"):

        self.source_df = source_df
        self.target_df = target_df
        self.source_id = source_id
        self.target_id = target_id
        self.source_columns = source_columns
        self.target_columns = target_columns
        self.source_weights = source_weights
        self.target_weights = target_weights
        self.source_rule_based = source_rule_based
        self.target_rule_based = target_rule_based
        self.threshold = threshold
        self.feature_threshold = feature_threshold
        self.output_file = output_file

    def calculate_similarity(self, source_val, target_val, rule_based=False):
        """Calculate similarity between two values."""
        if pd.isna(source_val) or pd.isna(target_val):
            return None
        if rule_based:
            return 1.0 if source_val == target_val else 0.0
        return fuzz.ratio(str(source_val), str(target_val)) / 100

    def match_records(self):
        matches = []

        # Create a tqdm progress bar
        total_iterations = len(self.source_df) * len(self.target_df)
        with tqdm(total=total_iterations, desc="Matching records", unit="pair") as pbar:

            # Iterate through all rows in both source and target DataFrames
            for _, source_row in self.source_df.iterrows():
                matched_target_ids = set()

                for _, target_row in self.target_df.iterrows():
                    total_similarity = 0
                    total_weight = 0
                    similarity_scores = {}
                    feature_threshold_met = False  # Flag to ensure one feature exceeds feature threshold

                    for src_col, tgt_col, src_weight, tgt_weight in zip(self.source_columns, self.target_columns, self.source_weights, self.target_weights):
                        is_rule_based = src_col in self.source_rule_based or tgt_col in self.target_rule_based
                        similarity = self.calculate_similarity(source_row[src_col], target_row[tgt_col], rule_based=is_rule_based)

                        if similarity is not None:
                            weight = (src_weight + tgt_weight) / 2
                            similarity_scores[f"{src_col}_vs_{tgt_col}"] = similarity
                            total_similarity += similarity * weight
                            total_weight += weight

                            # Check if any feature's similarity score exceeds the feature threshold
                            if similarity >= self.feature_threshold:
                                feature_threshold_met = True

                    # Calculate overall similarity score and check both thresholds
                    overall_similarity = total_similarity / total_weight if total_weight > 0 else 0
                    if overall_similarity >= self.threshold and feature_threshold_met:
                        target_id = target_row[self.target_id]

                        # If a match is found, add all target rows with the same ID to matches
                        if target_id not in matched_target_ids:
                            matched_target_ids.add(target_id)
                            matching_rows = self.target_df[self.target_df[self.target_id] == target_id]

                            for _, matched_row in matching_rows.iterrows():
                                matches.append({
                                    "source_id": source_row[self.source_id],
                                    "target_id": matched_row[self.target_id],
                                    **similarity_scores,
                                    "overall_similarity": overall_similarity
                                })

                    pbar.update(1)

        # Save results to Parquet file
        match_df = pd.DataFrame(matches)
        table = pa.Table.from_pandas(match_df)
        pq.write_table(table, self.output_file)

# Sample usage with feature threshold parameter
matcher = SimilarityMatcher(
    source_df=source_df,
    target_df=filtered,
    source_id="id",
    target_id="id",
    source_columns=["cleaned_first_name",
                    "cleaned_last_name",
                    "cleaned_address",
                    "cleaned_type",
                    "cleaned_subtype"],
    target_columns=["cleaned_first_name",
                    "cleaned_last_name",
                    "cleaned_address",
                    "cleaned_type",
                    "cleaned_subtype"],
    source_weights=[0.3, 0.5, 0.1, 0.5, 0.5],
    target_weights=[0.3, 0.5, 0.1, 0.5, 0.5],
    source_rule_based=["cleaned_phonenumber",
                       "cleaned_email"],
    target_rule_based=["cleaned_phonenumber",
                       "cleaned_email"],
    threshold=0.6,
    feature_threshold=0.85,  # Ensures at least one feature is highly similar
    output_file="results.parquet"
)

matcher.match_records()

