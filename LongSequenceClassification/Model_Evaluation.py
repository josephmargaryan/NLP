import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Optional, Tuple


class ECEStatistics:
    def __init__(self, path: str):
        self.path = path
        self.df_filtered = self.preprocess()

    def preprocess(self) -> pd.DataFrame:
        """
        Takes the file path with the parquet files, concatenates them, and removes documents with non-manual classification.
        """
        out_folder = self.path
        regex = ".parquet"

        folders = os.listdir(out_folder)
        dataframes = []
        for folder in folders:
            folder_path = os.path.join(out_folder, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                for file in files:
                    if file.endswith(regex):
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_parquet(file_path)
                        dataframes.append(df)

        final_df = pd.concat(dataframes, axis=0)
        final_df.reset_index(drop=True, inplace=True)
        print(f"The number of total samples: {len(final_df)}")
        df_filtered = final_df.loc[~final_df["CLASSIFICATION"].isna(), :]
        print(f"The amount of samples manually classified {len(df_filtered)}")
        df_filtered = df_filtered.loc[~df_filtered["classes"].isna(), :]
        df_filtered = df_filtered.loc[~df_filtered["conf"].isna(), :]
        df_filtered = df_filtered[df_filtered['classes'].apply(lambda x: x[0] != 'UNKNOWN')]
        df_filtered = df_filtered[df_filtered["text"].apply(lambda x: len(str(x)) > 10)]
        print(f"The number predictions by the model: {len(df_filtered)}")

        results = df_filtered.apply(
        lambda row: pd.Series(self.get_first_non_unknown(row['classes'], row['conf'])),
        axis=1) 

        results.columns = ['selected_class', 'selected_conf']
        df_filtered = pd.concat([df_filtered, results], axis=1)

        self.y_true = df_filtered["CLASSIFICATION"]
        self.y_hat = df_filtered["selected_class"]
        self.y_conf = df_filtered["selected_conf"]

        return df_filtered

    @staticmethod
    def get_first_non_unknown(classes: List[str], conf: List[float]) -> Tuple[Optional[str], Optional[float]]:
        for i, cls in enumerate(classes):
            if cls != "UNKNOWN":
                return cls.split('#')[-1], conf[i]
        return None, None  # If all values are UNKNOWN

    def create_ece(self, num_bins: Optional[int] = 10):
        """
        Calculate and plot the Expected Calibration Error (ECE) for a set of predictions.
        """
        true_labels = self.y_true
        predicted_labels = self.y_hat
        predicted_confidences = self.y_conf

        classes = sorted(set(true_labels))
        bins = np.linspace(0, 1, num_bins + 1)

        total_bin_counts = np.zeros(num_bins)
        total_bin_correct_counts = np.zeros(num_bins)
        total_bin_confidences = np.zeros(num_bins)

        for true, pred, conf in zip(true_labels, predicted_labels, predicted_confidences):
            bin_index = np.digitize(conf, bins) - 1
            bin_index = max(0, min(bin_index, num_bins - 1))  

            total_bin_counts[bin_index] += 1
            total_bin_confidences[bin_index] += conf
            if true == pred:
                total_bin_correct_counts[bin_index] += 1

        bin_accuracies = total_bin_correct_counts / total_bin_counts
        bin_confidences = total_bin_confidences / total_bin_counts

        bin_accuracies = np.nan_to_num(bin_accuracies)
        bin_confidences = np.nan_to_num(bin_confidences)

        ece = np.sum(total_bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(total_bin_counts)

        plt.figure(figsize=(12, 8))
        plt.plot(bin_confidences, bin_accuracies, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Perfect Calibration')
        plt.xlabel('Confidence Score')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig("ECE_Overall.png")

        print(f'Overall ECE: {ece:.4f}')
        return ece

    def calculate_accuracy(self) -> float:
        """
        Calculate accuracy from true and predicted labels.
        """
        assert len(self.y_true) == len(self.y_hat), "y_true and y_hat must have the same length"

        correct_predictions = sum(yt == yh for yt, yh in zip(self.y_true, self.y_hat))
        accuracy = correct_predictions / len(self.y_true)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def correct_predictions_by_class(self) -> None:
        """
        Count and print correct predictions by class.
        """
        assert len(self.y_true) == len(self.y_hat), "y_true and y_hat must have the same length"

        correct_counts = defaultdict(int)

        for yt, yh in zip(self.y_true, self.y_hat):
            if yt == yh:
                correct_counts[yt] += 1

        sorted_correct_counts = sorted(correct_counts.items(), key=lambda item: item[1], reverse=True)

        print("Correct predictions by class:")
        for cls, count in sorted_correct_counts:
            print(f"Class '{cls}': {count} correct predictions")

        classes = [cls for cls, _ in sorted_correct_counts]
        counts = [count for _, count in sorted_correct_counts]


        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Number of Correct Predictions')
        plt.title('Correct Predictions by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Barplot.png")

    def plot_confidence_distribution(self) -> None:
        """
        Plot the distribution of model predicted probabilities.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(self.y_conf, bins=10, edgecolor='k', alpha=0.7)
        plt.title('Distribution of Model Predicted Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig("Conf_Distribution.png")


if __name__=="__main__":
    path = "/data-disk/scraping-output/p-drive-structured"
    ece_stats = ECEStatistics(path)
    ece_stats.create_ece(num_bins=10)
    ece_stats.calculate_accuracy()
    ece_stats.correct_predictions_by_class()
    ece_stats.plot_confidence_distribution()
