
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load collected data
data_path = os.path.join("data", "gestures.csv")
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nGesture samples:")
print(df.iloc[:, -1].value_counts())

# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

if y.nunique() < 2:
    raise ValueError("At least two gesture classes are required.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, os.path.join("model", "gesture_model.pkl"))

print("\nModel saved to model/gesture_model.pkl")
