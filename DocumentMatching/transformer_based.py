df['combined_text'] = df['name'] + " " + df['address'] + " " + df['other_text_features']

from sentence_transformers import SentenceTransformer

# Load pre-trained transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for the combined textual metadata
df['embeddings'] = df['combined_text'].apply(lambda x: model.encode(x))

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all pairs of documents
cosine_sim = cosine_similarity(df['embeddings'].tolist())

# Example: Get similarity score between document i and document j
similarity_score = cosine_sim[i, j]

df['is_match'] = cosine_sim > 0.85

# Combine cosine similarity from embeddings with other features
df['total_score'] = (0.7 * cosine_sim) + (0.3 * other_similarity_features)
df['is_match'] = df['total_score'] > threshold
