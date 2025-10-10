import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


age = [18, 22, 30, 45, 65, 80]
accident_no = [38, 36, 24, 20, 18, 28]


X = np.array(age).reshape(-1, 1) 
y = np.array(accident_no)        


model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Age of Driver')
plt.ylabel('Number of Accidents')
plt.title('Simple Linear Regression: Age vs Number of Accidents')
plt.legend()
plt.show()

    
n=int(input("Ente the car age:"))
predict_ac=model.predict([[n]])
print(f"Predicted number of accident: {predict_ac[0]:.2f}")
