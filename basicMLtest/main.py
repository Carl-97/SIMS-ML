import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


def contains_digits_and_letters(entry):
    return any(char.isdigit() for char in entry) and any(char.isalpha() for char in entry)


def extract_features(column_data):
    text_entries = [item for item in column_data if isinstance(item, str)]
    text_count = len(text_entries)
    capitalized_count = sum(1 for item in text_entries if item.istitle())
    low_numbers_count = sum(1 for item in column_data if isinstance(item, (int, float)) and 0 <= item <= 100)
    large_numbers_count = sum(1 for item in column_data if isinstance(item, (int, float)) and item > 1000)
    seven_digit_numbers_count = sum(1 for item in column_data if isinstance(item, (int, float)) and 1e6 <= item < 1e7)
    alphanumeric_count = sum(1 for item in text_entries if contains_digits_and_letters(item))

    return [text_count, capitalized_count, low_numbers_count, large_numbers_count, seven_digit_numbers_count, alphanumeric_count]


def load_data(filename, header=True):
    return pd.read_excel(filename, engine='openpyxl', header=None if not header else 0)


def train_and_save_model(train_file, model_file):
    train_data = load_data(train_file)
    train_features = [extract_features(train_data[col].dropna()) for col in train_data.columns]
    train_labels = train_data.columns.tolist()

    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(train_features, train_labels)
    joblib.dump(clf, model_file)


def predict_labels(model, test_features):
    return model.predict(test_features)


def collect_feedback_and_retrain(model_file, test_file, train_file):
    clf_loaded = joblib.load(model_file)
    test_data = load_data(test_file, header=False)
    test_features = [extract_features(test_data[col].dropna()) for col in test_data.columns]

    predicted_labels = predict_labels(clf_loaded, test_features)
    corrected_labels = []

    # Collect feedback
    for idx, label in enumerate(predicted_labels):
        print(f"Column {idx + 1} is predicted as: {label}")
        correction = input("Is this correct? (yes/no) If no, please provide the correct label: ").strip()
        corrected_labels.append(label if correction.lower() == 'yes' else correction)

    # Combine test data with original training data for retraining
    train_data = load_data(train_file)
    train_features = [extract_features(train_data[col].dropna()) for col in train_data.columns]
    train_labels = train_data.columns.tolist() + corrected_labels

    # Add more trees to the existing forest
    n_additional_trees = 50
    clf = RandomForestClassifier(n_estimators=n_additional_trees, warm_start=True)
    clf.classes_ = clf_loaded.classes_
    clf.n_classes_ = clf_loaded.n_classes_
    clf.n_features_in_ = clf_loaded.n_features_in_
    clf.n_outputs_ = clf_loaded.n_outputs_
    clf.estimators_ = clf_loaded.estimators_

    clf.fit(train_features + test_features, train_labels)

    # Save the updated model
    joblib.dump(clf, model_file)


def predict_labels_test(model_file, test_file):
    clf_loaded = joblib.load(model_file)
    test_data = pd.read_excel(test_file, header=None, engine='openpyxl')
    test_features = [extract_features(test_data[col].dropna()) for col in test_data.columns]
    return clf_loaded.predict(test_features)


# Execute
train_and_save_model('data/formatted_training.xlsx', 'models/trained_model.pkl')
collect_feedback_and_retrain('models/trained_model.pkl', 'data/ifm_electronic_training.xlsx', 'data/formatted_training.xlsx')

predicted_labels = predict_labels_test('models/trained_model.pkl', 'data/quality_secured_articles.xlsx')
for idx, label in enumerate(predicted_labels):
    print(f"Column {idx + 1} is likely: {label}")
print('Finished, Thanks for your feedback!')
