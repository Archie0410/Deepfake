import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# 1. Load Dataset
def load_data(path='data/smart_adr_dataset.csv'):
    return pd.read_csv(path)

# 2. Preprocess Data
def preprocess(df):
    df = df.copy()

    # Label encode the target column
    target_encoder = LabelEncoder()
    df['adr_severity_encoded'] = target_encoder.fit_transform(df['adr_severity'])

    # Categorical features to encode
    categorical_cols = [
        'gender', 'drug', 'genomics', 'past_diseases',
        'reason_for_drug', 'allergies', 'addiction',
        'ayurvedic_medicine', 'hereditary_disease', 'age_group'
    ]

    encoders = {'adr_severity': target_encoder}  # Store target encoder

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['adr_severity', 'adr_severity_encoded'])
    y = df['adr_severity_encoded']
    return X, y, encoders

# 3. Train Model
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# 4. Save Model
def save_model(model, path='models/xgb_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

# 5. Save Encoders
def save_encoders(encoders, path='models/encoders.pkl'):
    joblib.dump(encoders, path)

# 6. Main Pipeline
def main():
    print("🔄 Loading data...")
    df = load_data()

    print("🧹 Preprocessing...")
    X, y, encoders = preprocess(df)

    print("🧠 Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📈 Training...")
    model = train_model(X_train, y_train)

    print("💾 Saving model...")
    save_model(model)

    print("💾 Saving encoders...")
    save_encoders(encoders)

    print("✅ Evaluating...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Entry point
if __name__ == "__main__":
    main()
