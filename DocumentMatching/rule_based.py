weights = {
    'phone_number': 0.6,
    'address': 0.4,
    'name': 0.2
}

from fuzzywuzzy import fuzz

def calculate_similarity(row):
    phone_similarity = 1 if row['phone_number_1'] == row['phone_number_2'] else 0
    address_similarity = fuzz.ratio(row['address_1'], row['address_2']) / 100
    name_similarity = fuzz.ratio(row['name_1'], row['name_2']) / 100

    total_similarity = (
        weights['phone_number'] * phone_similarity +
        weights['address'] * address_similarity +
        weights['name'] * name_similarity
    )
    return total_similarity

df['similarity_score'] = df.apply(calculate_similarity, axis=1)
df['is_match'] = df['similarity_score'] > 0.7

