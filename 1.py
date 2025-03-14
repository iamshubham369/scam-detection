# Step 1: Import Required Libraries
print("Importing Libraries... ✅")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ignore unnecessary warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Step 2: Load Dataset
print("Loading Dataset... ✅")
df = pd.read_csv("phishing_dataset.csv")  # Ensure dataset.csv is in the same directory

# Step 3: Data Preprocessing
print("Checking for missing values... ✅")
print(df.isnull().sum())  # Display missing values

# Drop rows with missing values if any
df.dropna(inplace=True)

# Step 4: Feature Selection
print("Preparing Features and Labels... ✅")
X = df.drop(columns=["index", "Result"])  # Remove unnecessary columns
y = df["Result"]

# Step 5: Split Dataset into Training & Testing Sets
print("Splitting Dataset into Train and Test... ✅")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

# Step 6: Train Initial Model
print("Training Random Forest Model... ✅")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate Model Performance
print("Evaluating Model... ✅")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Hyperparameter Tuning
print("Performing Hyperparameter Tuning... ✅ (This may take a few minutes)")
param_grid = {
    'n_estimators': [50, 100],  # Reduced number of estimators
    'max_depth': [None, 10],  # Smaller depth range
    'min_samples_split': [2, 5]  # Fewer values to test
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

print("Best Parameters Found: ✅", grid_search.best_params_)

# Step 9: Train Model with Best Parameters
print("Training Optimized Model... ✅")
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 10: Final Evaluation
print("Final Model Evaluation... ✅")
y_final_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_final_pred)
print(f"Final Model Accuracy: {final_accuracy:.2f}")

print("Final Classification Report:\n", classification_report(y_test, y_final_pred))
print("Final Confusion Matrix:\n", confusion_matrix(y_test, y_final_pred))

# Step 11: Save Model (Optional)
import joblib
print("Saving Model... ✅")
joblib.dump(best_model, "phishing_model.pkl")
print("Model Saved as phishing_model.pkl ✅")
# Function to make predictions on new data
def predict_url(features):
    model = joblib.load("phishing_model.pkl")  # Load the saved model
    prediction = model.predict([features])  # Make a prediction
    return "Phishing Website 🚨" if prediction[0] == 1 else "Legit Website ✅"

# Example test case (Replace with real feature values)
sample_features = [1, -1, 1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1]  
result = predict_url(sample_features)
print("Prediction for the given URL:", result)
