import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

print("Features shape:", X.shape)
print("Labels distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

#loading without scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy without scaling: {acc:.4f}")


#Load with scaling
knn = KNeighborsClassifier(n_neighbors=5)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)

acc_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"Accuracy with scaling: {acc_scaled:.4f}")

#Choose the optimal k by plotting accuracy
k_range = range(1, 21)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(k_range, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN accuracy for different k values')
plt.show()
optimal_k = k_range[np.argmax(accuracies)]
print(f"Optimal k: {optimal_k} with accuracy: {max(accuracies):.4f}")


#Confusion matrix and class-wise performance
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)
y_pred_optimal = knn.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred_optimal, target_names=['Malignant', 'Benign']))


#Feature importance via training on subsets of features
features = X.columns
feature_accuracies = []

for feature in features:
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_scaled[:, [X.columns.get_loc(feature)]], y_train)
    y_pred_feat = knn.predict(X_test_scaled[:, [X.columns.get_loc(feature)]])
    acc = accuracy_score(y_test, y_pred_feat)
    feature_accuracies.append((feature, acc))

feature_accuracies.sort(key=lambda x: x[1], reverse=True)

print("Feature accuracies:")
for feat, acc in feature_accuracies[:10]:  # Top 10 features
    print(f"{feat}: {acc:.4f}")

#Effect of train-test split ratio
for test_size in [0.4, 0.2, 0.1]:  # Corresponds to train sizes 60%, 80%, 90%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Train-Test split ratio {int((1-test_size)*100)}-{int(test_size*100)}: Accuracy = {acc:.4f}")
