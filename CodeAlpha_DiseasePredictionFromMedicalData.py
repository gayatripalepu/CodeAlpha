import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
heart_data = pd.read_csv(dataset_url, names=column_names)

# Preprocess the data
heart_data = heart_data.replace('?', np.nan)
heart_data = heart_data.dropna()

input_features = heart_data.drop("target", axis=1)
target_labels = heart_data["target"]

# Convert categorical columns to numeric
input_features["ca"] = input_features["ca"].astype(float)
input_features["thal"] = input_features["thal"].astype(float)
target_labels = target_labels.apply(lambda x: 1 if x > 0 else 0)  # Convert target to binary

# Standardize the features
data_scaler = StandardScaler()
features_standardized = data_scaler.fit_transform(input_features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features_standardized, target_labels, test_size=0.2, random_state=42)

# Build the model
heart_disease_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
heart_disease_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
training_history = heart_disease_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = heart_disease_model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {accuracy}')

# Make predictions
predictions = (heart_disease_model.predict(X_test) > 0.5).astype("int32")

# Print the classification report
print(classification_report(y_test, predictions))
