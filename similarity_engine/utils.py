import matplotlib.pyplot as plt
import pandas as pd 

engine = SimilarityEngine(config_path="/path/to/engine_config.yml")

similarity_results = engine.compare_multiple_records(filtered_source_df.iloc[0:50, :], filtered_target_df.iloc[0:500, :])
similarity_results

similarity_results["overall_similarity"].plot(kind="hist")
plt.xlabel("Overall similarity score")
plt.show()

def plot_rule_based_vs_similarity(similarity_results):
    """
    Generates a bar plot to compare the count of matches based on rule-based and overall similarity,
    with adjustable alpha transparency for better visualization.
    
    :param similarity_results: DataFrame containing 'rule_based_match' and 'match_label' columns
    """
    # Group data by 'rule_based_match' and 'match_label' and count occurrences
    grouped_data = similarity_results.groupby(['rule_based_match', 'match_label']).size().reset_index(name='count')

    # Create a bar plot with transparency (alpha)
    plt.figure(figsize=(10, 6))
    for rule_based, color, alpha in zip([True, False], ['blue', 'orange'], [0.6, 0.4]):
        subset = grouped_data[grouped_data['rule_based_match'] == rule_based]
        plt.bar(subset['match_label'], subset['count'], color=color, alpha=alpha, label=f'Rule Based Match: {rule_based}')

    # Customize the plot
    plt.xlabel('Match Label')
    plt.ylabel('Count')
    plt.title('Count of Matches Based on Rule-Based and Overall Similarity')
    plt.legend()

    # Show the plot
    plt.show()

# Example usage with higher alpha values for clearer distinction
plot_rule_based_vs_similarity(similarity_results)


import matplotlib.pyplot as plt

def plot_feature_histograms(similarity_results, feature_columns):
    """
    Plots histograms for individual similarity features to help visualize their distributions.
    
    :param similarity_results: DataFrame containing the similarity results
    :param feature_columns: List of columns in the DataFrame to plot the distributions for
    """
    num_features = len(feature_columns)
    plt.figure(figsize=(15, 5 * num_features))  # Adjust size depending on the number of features
    
    # Plot histograms for each feature
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(num_features, 1, i)  # Create subplots for each feature
        plt.hist(similarity_results[feature].dropna(), bins=50, alpha=0.7, color='blue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Example usage
feature_columns = ['first_name_similarity', 'last_name_similarity', 'address_fuzzy_similarity']
plot_feature_histograms(similarity_results, feature_columns)


def plot_similarity_trends(similarity_results, feature_columns, overall_column='overall_similarity'):
    """
    Plots line charts to show the trends of individual similarity features along with the overall similarity.
    
    :param similarity_results: DataFrame containing the similarity results
    :param feature_columns: List of columns in the DataFrame to plot the trends for
    :param overall_column: Column name of the overall similarity score (default: 'overall_similarity')
    """
    num_features = len(feature_columns)
    plt.figure(figsize=(15, 5 * num_features))  # Adjust size depending on the number of features
    
    # Plot each feature's trend along with the overall similarity
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(num_features, 1, i)  # Create subplots for each feature
        plt.plot(similarity_results[feature], label=f'{feature} Trend', color='blue', alpha=0.6)
        plt.plot(similarity_results[overall_column], label=f'Overall Similarity', color='green', linestyle='--', alpha=0.8)
        plt.title(f'{feature} and Overall Similarity Trends')
        plt.xlabel('Record Index')
        plt.ylabel('Similarity Score')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
feature_columns = ['first_name_similarity', 'last_name_similarity', 'address_fuzzy_similarity']
plot_similarity_trends(similarity_results, feature_columns)


