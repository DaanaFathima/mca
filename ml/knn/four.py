import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Iris.csv")
print(df.head())
df.info()
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

df_numeric = df.drop(columns=["Id", "Species"], errors="ignore")

plt.figure(figsize=(10,6))
sns.boxplot(data=df_numeric)
plt.title("Before Outlier Treatment")
plt.show()

Q1 = df['SepalWidthCm'].quantile(0.25)
Q3 = df['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Lower bound: {lower}, Upper bound: {upper}")

df_capped = df.copy()
df_capped['SepalWidthCm'] = df_capped['SepalWidthCm'].clip(lower, upper)

plt.figure(figsize=(10,6))
sns.boxplot(data=df_capped.drop(columns=["Id", "Species"], errors="ignore"))
plt.title("After Outlier Treatment (SepalWidthCm Capped)")
plt.show()

X = df.drop(columns=["Id", "Species"], errors="ignore")
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(y_train.value_counts())

k=5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

new_instance=pd.DataFrame({
    'SepalLengthCm': [38],
    'SepalWidthCm': [4.9],
    'PetalLengthCm': [3.1],
    'PetalWidthCm': [1.5]
})

prediction = knn.predict(new_instance)
print("Predicted species for new instance:", prediction[0])

