import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Load dataset
df = pd.read_csv('crop_data.csv')

# Print column names for verification
print("Dataset Columns:", df.columns.tolist())

# Drop columns with high missing values
columns_to_drop = ['ph', 'rainfall', 'N', 'P', 'K']
df = df.drop(columns=columns_to_drop, errors='ignore')  # Ignore if columns don't exist

# Encode target variable
df['Status'] = df['Status'].map({'ON': 1, 'OFF': 0})

# Define features
features = ['Soil Moisture', 'Temperature', ' Soil Humidity', 'Time', 
            'Air temperature (C)', 'Wind speed (Km/h)', 'Air humidity (%)', 
            'Wind gust (Km/h)', 'Pressure (KPa)']

# Verify that all features exist in the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise KeyError(f"The following features are not in the dataset: {missing_features}")

# Select features and target
X = df[features]
y = df['Status']

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create directory for saving model files if it doesn't exist
os.makedirs('app/model', exist_ok=True)

# Save model, imputer, and scaler
joblib.dump(model, 'app/model/model.pkl')
joblib.dump(imputer, 'app/model/imputer.pkl')
joblib.dump(scaler, 'app/model/scaler.pkl')

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")