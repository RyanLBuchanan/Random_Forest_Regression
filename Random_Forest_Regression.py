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


# Predict a new result using specific instance of Years of Experience


# Visualize the Random Forest Regression results (higher resolution)