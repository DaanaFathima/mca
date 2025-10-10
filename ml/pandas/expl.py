import pandas as pd
df = pd.read_csv("/home/user/Desktop/ml/pandas/Iris.csv")

print("\n1. DataFrame() ")
print(df.head())

print("\n2. head()")
print(df.head(3))

print("\n3. tail()")
print(df.tail(3))

print("\n4. count() ")
print(df.count())

print("\n5. sum() ")
print(df.sum(numeric_only=True))

print("\n6. mean() ")
print(df.mean(numeric_only=True))

print("\n7. median() ")
print(df.median(numeric_only=True))

print("\n8. mode() ")
print(df.mode().head(1))

print("\n9. std() ")
print(df.std(numeric_only=True))

print("\n10. min()")
print(df.min(numeric_only=True))

print("\n11. max() ")
print(df.max(numeric_only=True))

print("\n13. prod()")
print(df.prod(numeric_only=True))


print("\n16. info() ")
print(df.info())

print("\n17. shape ")
print(df.shape)

print("\n18. describe() ")
print(df.describe())

# Absolute values (first 5 rows)
print(df.select_dtypes(include='number').abs().head())

# Cumulative sum (first 5 rows)
print(df.select_dtypes(include='number').cumsum().head())

# Cumulative product (first 5 rows)
print(df.select_dtypes(include='number').cumprod().head())
