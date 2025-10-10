
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix, andrews_curves, parallel_coordinates

sns.set(style="whitegrid")

df = pd.read_csv("Iris.csv")
print("Columns:", df.columns)

df = df.drop(columns=["Id"])


print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)  # rows, cols
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nUnique Species:", df["Species"].unique())
print("\nCount per Species:\n", df["Species"].value_counts())

print("\nStatistical Summary:")
print(df.describe())


print("Info")
print(df.info())

print("\nCorrelation Matrix:")
print(df.drop(columns=["Species"]).corr())





df.hist(figsize=(10,8), bins=15, edgecolor="black")
plt.suptitle("Histograms of Iris Features", fontsize=16)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df.drop(columns=["Species"]), orient="h")
plt.title("Boxplots of Iris Features")
plt.show()


plt.figure(figsize=(10,6))
for sp in df["Species"].unique():
    subset = df[df["Species"] == sp]
    sns.kdeplot(subset["PetalLengthCm"], label=sp, shade=True)
plt.title("Distribution of Petal Length by Species")
plt.xlabel("Petal Length (cm)")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", style="Species")
plt.title("Sepal Length vs Sepal Width by Species")
plt.show()

sns.pairplot(df, hue="Species", diag_kind="kde")
plt.suptitle("Pairplot of Features by Species", y=1.02)
plt.show()


scatter_matrix(df.drop(columns=["Species"]), figsize=(10,10), diagonal="kde", alpha=0.7)
plt.suptitle("Scatter Matrix of Iris Features", y=1.02)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], 
            s=df["PetalLengthCm"]*50, c=pd.factorize(df["Species"])[0], alpha=0.6)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Bubble Chart (Bubble size = Petal Length)")
plt.show()

plt.figure(figsize=(10,6))
sns.kdeplot(df["SepalLengthCm"], shade=True, label="Sepal Length")
sns.kdeplot(df["SepalWidthCm"], shade=True, label="Sepal Width")
sns.kdeplot(df["PetalLengthCm"], shade=True, label="Petal Length")
sns.kdeplot(df["PetalWidthCm"], shade=True, label="Petal Width")
plt.legend()
plt.title("Density Plot of Iris Features")
plt.show()

plt.figure(figsize=(10,6))
parallel_coordinates(df, "Species", colormap=plt.cm.Set1)
plt.title("Parallel Coordinates of Iris Features")
plt.show()


means = df.groupby("Species").mean()
means.T.plot(kind="bar", figsize=(10,6))
plt.title("Deviation Chart (Mean Feature Values per Species)")
plt.ylabel("Mean Value")
plt.show()

plt.figure(figsize=(10,6))
andrews_curves(df, "Species")
plt.title("Andrews Curves of Iris Dataset")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns=["Species"]).corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Iris Features")
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(x="Species", y="PetalLengthCm", data=df)
plt.title("Violin Plot of Petal Length by Species")
plt.show()