import matplotlib.pyplot as plt
import pandas as pd


import os
import matplotlib.pyplot as plt


class SimilarityVisualizer:
    def __init__(self, similarity_results, output_folder="visualizations"):
        """
        Initialize the visualizer with the similarity results dataframe and output folder.

        :param similarity_results: DataFrame containing the similarity results
        :param output_folder: Folder to save visualizations
        """
        self.similarity_results = similarity_results
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def plot_overall_similarity_hist(self):
        """
        Plots and saves a histogram of the overall similarity scores.
        """
        self.similarity_results["overall_similarity"].plot(
            kind="hist", bins=50, color="blue", alpha=0.7
        )
        plt.xlabel("Overall similarity score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Overall Similarity Scores")

        output_path = os.path.join(self.output_folder, "overall_similarity_hist.png")
        plt.savefig(output_path)
        plt.close()

    def plot_rule_based_vs_similarity(self):
        """
        Generates and saves a bar plot to compare the count of matches based on rule-based and overall similarity,
        with adjustable alpha transparency for better visualization.
        """
        grouped_data = (
            self.similarity_results.groupby(["rule_based_match", "match_label"])
            .size()
            .reset_index(name="count")
        )

        # Create a bar plot with transparency (alpha)
        plt.figure(figsize=(10, 6))
        for rule_based, color, alpha in zip(
            [True, False], ["blue", "orange"], [0.6, 0.4]
        ):
            subset = grouped_data[grouped_data["rule_based_match"] == rule_based]
            plt.bar(
                subset["match_label"],
                subset["count"],
                color=color,
                alpha=alpha,
                label=f"Rule Based Match: {rule_based}",
            )

        plt.xlabel("Match Label")
        plt.ylabel("Count")
        plt.title("Count of Matches Based on Rule-Based and Overall Similarity")
        plt.legend()

        output_path = os.path.join(self.output_folder, "rule_based_vs_similarity.png")
        plt.savefig(output_path)
        plt.close()

    def plot_feature_histograms(self, feature_columns):
        """
        Plots and saves histograms for individual similarity features.

        :param feature_columns: List of columns in the DataFrame to plot the distributions for
        """
        num_features = len(feature_columns)
        plt.figure(
            figsize=(15, 5 * num_features)
        )  # Adjust size depending on the number of features

        for i, feature in enumerate(feature_columns, 1):
            plt.subplot(num_features, 1, i)  # Create subplots for each feature
            plt.hist(
                self.similarity_results[feature].dropna(),
                bins=50,
                alpha=0.7,
                color="blue",
            )
            plt.title(f"Distribution of {feature}")
            plt.xlabel("Similarity Score")
            plt.ylabel("Frequency")

        output_path = os.path.join(self.output_folder, "feature_histograms.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_similarity_trends(
        self, feature_columns, overall_column="overall_similarity"
    ):
        """
        Plots and saves line charts to show the trends of individual similarity features along with the overall similarity.

        :param feature_columns: List of columns in the DataFrame to plot the trends for
        :param overall_column: Column name of the overall similarity score (default: 'overall_similarity')
        """
        num_features = len(feature_columns)
        plt.figure(
            figsize=(15, 5 * num_features)
        )  # Adjust size depending on the number of features

        for i, feature in enumerate(feature_columns, 1):
            plt.subplot(num_features, 1, i)  # Create subplots for each feature
            plt.plot(
                self.similarity_results[feature],
                label=f"{feature} Trend",
                color="blue",
                alpha=0.6,
            )
            plt.plot(
                self.similarity_results[overall_column],
                label="Overall Similarity",
                color="green",
                linestyle="--",
                alpha=0.8,
            )
            plt.title(f"{feature} and Overall Similarity Trends")
            plt.xlabel("Record Index")
            plt.ylabel("Similarity Score")
            plt.legend()

        output_path = os.path.join(self.output_folder, "similarity_trends.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


# Example usage
if __name__ == "__main__":
    """    # Assuming similarity_results has already been generated by the SimilarityEngine
        similarity_results = pd.read_csv("/path/to/similarity_engine_results")
        visualizer = SimilarityVisualizer(similarity_results)

        # Plot overall similarity histogram
        visualizer.plot_overall_similarity_hist()

        # Plot rule-based vs similarity counts
        visualizer.plot_rule_based_vs_similarity()

        # Plot histograms for selected feature columns
        feature_columns = [
            "first_name_similarity",
            "last_name_similarity",
            "address_fuzzy_similarity",
            "subtype_similarity",
            "type_similarity",
        ]
        visualizer.plot_feature_histograms(feature_columns)

        # Plot trends of selected feature columns along with overall similarity
        visualizer.plot_similarity_trends(feature_columns)"""
