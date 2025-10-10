import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

car_age=[5,7,8,7,2,17,2,9,4,11,12,9,6]
car_speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]

x=np.array(car_age).reshape(-1,1)
y=np.array(car_speed)

model=LinearRegression()
model.fit(x,y)

y_predict=model.predict(x)

plt.scatter(x,y,color="blue",label="Actual Data")
plt.plot(x,y_predict,color="red",linewidth="2",label="Regression Line")

plt.title("Linear Regression")

plt.xlabel('Age')
plt.ylabel('Speed')

plt.legend()
plt.show()

n=int(input("Enter Car Age"))
predicted_speed=model.predict([[n]])
print(f"Predicted Car Speed:{predicted_speed[0]:2f}")
