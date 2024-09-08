from fuzzywuzzy import fuzz

def calculate_similarity(row):
    # Example of weighted similarity
    name_similarity = fuzz.ratio(row['name_1'], row['name_2']) / 100
    surname_similarity = fuzz.ratio(row['surname_1'], row['surname_2']) / 100
    occupation_similarity = fuzz.ratio(row['occupation_1'], row['occupation_2']) / 100
    phone_similarity = 1 if row['phone_1'] == row['phone_2'] else 0

    # Weighted total similarity
    total_similarity = (
        0.3 * name_similarity +
        0.2 * surname_similarity +
        0.2 * occupation_similarity +
        0.3 * phone_similarity
    )
    return total_similarity

# Apply to dataframe
df['similarity_score'] = df.apply(calculate_similarity, axis=1)
df['is_match'] = df['similarity_score'] > 0.7


