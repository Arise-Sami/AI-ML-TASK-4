"""
Project 2: Flower Classifier
Task 4: Prepare the Data (Split into Train/Test)

Load the Iris dataset, separate features from labels,
then split into 70% training and 30% testing sets.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
iris = load_iris()

# Features: sepal length, sepal width, petal length, petal width
X = iris.data

# Labels: 0=setosa, 1=versicolor, 2=virginica
y = iris.target

# Show the full dataset as a pandas DataFrame (optional but useful)
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[label] for label in y]

print("=" * 60)
print("IRIS DATASET OVERVIEW")
print("=" * 60)
print(f"Total samples: {len(df)}")
print(f"Features: {', '.join(feature_names)}")
print(f"Species: {', '.join(target_names)}")
print()
print("First 5 rows:")
print(df.head())
print()
print("Samples per species:")
print(df["species"].value_counts().sort_index())
print()

# Split: 70% train, 30% test (test set is held out and never used during training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Feature columns:  {X_train.shape[1]}")
print()

# Quick check: label distribution in each set
train_labels = pd.Series(y_train).map(lambda i: target_names[i])
test_labels = pd.Series(y_test).map(lambda i: target_names[i])

print("Training set species counts:")
print(train_labels.value_counts().sort_index())
print()
print("Test set species counts:")
print(test_labels.value_counts().sort_index())
