from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from preprocess import matched_excel, csv_matched, unmatched_excel, csv_unmatched


def address_embedding_similarity(
    df_csv, df_excel, model_name="bert-base-nli-mean-tokens", limit_rows=None
):
    """
    Standardizes and concatenates address-related columns from two dataframes, generates embeddings for the addresses,
    and computes cosine similarity between the embeddings.

    Params:
    - df_csv (DataFrame): The first dataframe (e.g., CSV data) containing address components.
    - df_excel (DataFrame): The second dataframe (e.g., Excel data) containing address components.
    - model_name (str): The name of the pretrained model to use for embedding generation (default: 'bert-base-nli-mean-tokens').
    - limit_rows (int): If provided, limits the operation to the first `n` rows for testing.

    Returns:
    - merged_df (DataFrame): Merged DataFrame containing cosine similarity scores between address embeddings.
    """
    model = SentenceTransformer(model_name)

    if limit_rows:
        df_csv = df_csv.head(limit_rows).copy()
        df_excel = df_excel.head(limit_rows).copy()
    df_excel = df_excel.copy()
    df_excel["address_full"] = df_excel[
        ["ADDR1", "CITY", "ADMIN_AREA", "POSTALCODE", "COUNTRY"]
    ].apply(lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1)
    df_csv = df_csv.copy()
    df_csv["address_full"] = df_csv[
        [
            "address_line_1__v",
            "address_line_2__v",
            "locality__v",
            "postal_code__v",
            "country__v",
        ]
    ].apply(lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1)

    df_excel["address_embedding"] = df_excel["address_full"].apply(
        lambda x: model.encode(x)
    )
    df_csv["address_embedding"] = df_csv["address_full"].apply(
        lambda x: model.encode(x)
    )

    merged_df = pd.merge(
        df_csv[["vid__v", "address_full", "address_embedding"]],
        df_excel[["VID", "address_full", "address_embedding"]],
        left_on="vid__v",
        right_on="VID",
        how="inner",
    )

    merged_df["cosine_similarity"] = merged_df.apply(
        lambda row: cosine_similarity(
            [row["address_embedding_x"]], [row["address_embedding_y"]]
        )[0][0],
        axis=1,
    )

    return merged_df[
        ["vid__v", "address_full_x", "address_full_y", "cosine_similarity"]
    ]


def address_embedding_similarity_single(
    record, df_target, model_name="bert-base-nli-mean-tokens"
):
    """
    Standardizes and concatenates address-related columns from a single record and the target dataframe,
    generates embeddings for the addresses, and computes cosine similarity between the embeddings.

    Params:
    - record (Series): A single record containing address components.
    - df_target (DataFrame): The target dataframe containing address components.
    - model_name (str): The name of the pretrained model to use for embedding generation.

    Returns:
    - result_df (DataFrame): DataFrame containing cosine similarity scores between the record and target dataframe.
    """
    model = SentenceTransformer(model_name)

    # Prepare the address for the single record
    address_full_record = " ".join(
        [
            str(record[col])
            for col in ["ADDR1", "CITY", "ADMIN_AREA", "POSTALCODE", "COUNTRY"]
            if pd.notnull(record[col])
        ]
    )
    address_embedding_record = model.encode(address_full_record)

    # Prepare addresses for the target dataframe
    df_target = df_target.copy()
    df_target["address_full"] = df_target[
        [
            "address_line_1__v",
            "address_line_2__v",
            "locality__v",
            "postal_code__v",
            "country__v",
        ]
    ].apply(lambda row: " ".join([str(x) for x in row if pd.notnull(x)]), axis=1)
    df_target["address_embedding"] = df_target["address_full"].apply(
        lambda x: model.encode(x)
    )

    # Compute cosine similarity
    df_target["cosine_similarity"] = df_target["address_embedding"].apply(
        lambda x: cosine_similarity([address_embedding_record], [x])[0][0]
    )

    return df_target[["vid__v", "address_full", "cosine_similarity"]]


if __name__ == "__main__":
    merged_df_result = address_embedding_similarity(
        csv_matched.head(5), matched_excel.head(5), limit_rows=5
    )
    print(merged_df_result.head())
