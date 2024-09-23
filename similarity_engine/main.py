from similarity_engine import SimilarityEngine
from preprocessing import Preprocessing
from visualization_utils import SimilarityVisualizer
import pandas as pd


def main():
    print("Beginning preprocessing")
    print("\n" + "-" * 50)
    source_df, target_df, _ = Preprocessing(
        "/home/jmar/matching_project/similarity_engine/yml_examples/preprocess.yml"
    ).process_data()
    source_df = source_df.drop_duplicates(subset="id")
    target_df = target_df.drop_duplicates(subset="id")
    print("\n" + "-" * 50)
    print("Beginning similarity caclulations")
    print("\n" + "-" * 50)
    engine = SimilarityEngine(
        "/home/jmar/matching_project/similarity_engine/yml_examples/engine.yml"
    )
    scores = engine.compare_multiple_records(source_df.head(5000), target_df.head(5000))
    print("\n" + "-" * 50)
    print("We dont run the visualizations when comparing 25 mil records")
    print("\n" + "-" * 50)
    print("Computing accuracy")
    true_positives = scores.loc[scores["source_id"] == scores["target_id"]]
    value_counts = true_positives["match_label"].value_counts()
    true_matches = value_counts.get("Match", 0)
    total_records = value_counts.sum()
    accuracy = true_matches / total_records
    print(
        f"Accuracy: True Positives / All = {true_matches} / {total_records} = {accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
