# Random Forest Regression tutorial from Machine Learning A-Z - SuperDataScience
# Input by Ryan L Buchanan 24SEP20

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)
# Train the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predict a new result using specific instance of Years of Experience
regressor.predict([[6.5]])

# Visualize the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
ax = plt.gca()
ax.set_facecolor('grey')
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $US")
plt.show()