from similarity_engine import SimilarityEngine
from preprocessing import Preprocessing
from visualization_utils import SimilarityVisualizer
import pandas as pd


def main():
    # Preprocess data and retrieve dataframes
    preprocessing = Preprocessing(config_path="/path/to/preprocess_config.yml")
    source_df, target_df, merged_df = preprocessing.process_data()

    # Filter data
    source_df_unique = source_df.drop_duplicates(subset="id")
    target_df_unique = target_df.drop_duplicates(subset="id")
    merged_df = pd.merge(source_df_unique, target_df_unique, on="id", how="inner")
    correct_matches = merged_df[merged_df["MATCH_CATEGORY_HCP"] == "match manual"]
    matching_ids = correct_matches["id"].unique()

    filtered_source_df = source_df[source_df["id"].isin(matching_ids)]
    filtered_target_df = target_df[target_df["id"].isin(matching_ids)]

    # Apply similarity engine for a single record and save results
    engine = SimilarityEngine(config_path="/path/to/engine_config.yml")
    """    single_record_results = engine.compare_single_record(
            filtered_source_df.iloc[0],
            filtered_target_df,
            output_filename="single_record_results.csv",
        )
    """
    # Apply similarity engine for multiple records and save results
    similarity_results = engine.compare_multiple_records(
        filtered_source_df.tail(50),
        filtered_target_df.iloc[0:500, :],
        output_filename="similarity_results.csv",
    )

    # Visualize the results
    visualizer = SimilarityVisualizer(
        similarity_results, output_folder="visualizations"
    )
    visualizer.plot_overall_similarity_hist()
    visualizer.plot_rule_based_vs_similarity()

    feature_columns = [
        "first_name_similarity",
        "last_name_similarity",
        "address_fuzzy_similarity",
        "subtype_similarity",
        "type_similarity",
    ]
    visualizer.plot_feature_histograms(feature_columns)
    visualizer.plot_similarity_trends(feature_columns)


if __name__ == "__main__":
    main()
