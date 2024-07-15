import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Generate data
features, labels = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Convert to DataFrame 
column_names = [f'attribute_{i}' for i in range(features.shape[1])]
dataset = pd.DataFrame(features, columns=column_names)
dataset['creditworthiness'] = labels

# Splitting the dataset into training and sets
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('creditworthiness', axis=1), dataset['creditworthiness'], test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
credit_model = LogisticRegression()
credit_model.fit(X_train_scaled, y_train)

# predicting
predictions = credit_model.predict(X_test_scaled)

# Model Evaluation
model_accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Accuracy: {model_accuracy}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)