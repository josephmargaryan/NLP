def feature_engineering(row):
    row['phone_similarity'] = 1 if row['phone_number_1'] == row['phone_number_2'] else 0
    row['address_similarity'] = fuzz.ratio(row['address_1'], row['address_2']) / 100
    row['name_similarity'] = fuzz.ratio(row['name_1'], row['name_2']) / 100
    return row

df = df.apply(feature_engineering, axis=1)

X = df[['phone_similarity', 'address_similarity', 'name_similarity']]
y = df['match']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
