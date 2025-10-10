import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iris.csv")

df.columns = df.columns.str.strip()

feature_cols = [col for col in df.columns if col not in ["Id", "Species"]]


X = df[feature_cols].values
y = df["Species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def predict_one(x, k=5):
    distances = np.linalg.norm(X_train - x, axis=1) 
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

y_pred = [predict_one(x) for x in X_test]

accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

new_instance = np.array([5.1, 3.5, 1.4, 0.2])
prediction = predict_one(new_instance)
print("Prediction for new instance:", prediction)
