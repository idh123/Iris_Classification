import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Nischala\Downloads\Iris.csv")
print(dataset.head(10))

# Checking for missing values
print("Missing values per column:")
print(dataset.isnull().sum())  

# Plot histogram for Sepal Length
plt.figure(figsize=(8, 6))
plt.hist(dataset['SepalLengthCm'], bins=20, color='lightcoral', edgecolor='black')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Count')
plt.title('Sepal Length Distribution')
plt.show()

# Plot histogram for Petal Length
plt.figure(figsize=(8, 6))
plt.hist(dataset['PetalLengthCm'], bins=20, color='lightblue', edgecolor='black')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Count')
plt.title('Petal Length Distribution')
plt.show()

# Splitting dataset into features and target variable
features = dataset.drop(columns=['Species'])
target = dataset['Species']

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(X_train)}")

# Standardizing the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Evaluate Logistic Regression Model
log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy * 100:.2f}%")

# Making sample predictions using Logistic Regression
sample_predictions = log_model.predict([[0, 5.1, 3.5, 1.4, 0.2],
                                        [1, 6.7, 3.0, 5.2, 2.3]])
print("Predictions:", sample_predictions)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree Model
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
