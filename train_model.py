import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (IMPORTANT: separator = ;)
df = pd.read_csv("data/student-mat.csv", sep=";")

# Select features
X = df[["studytime", "failures", "absences"]]

# Target variable (pass or fail)
y = df["G3"] >= 10

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully")