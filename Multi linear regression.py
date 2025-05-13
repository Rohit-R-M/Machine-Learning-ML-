# Assignment on Multi Regression: Build an application where it can predict price of a house using multiple variable Linear Regression (use USA Housing dataset from Kaggle). Display all co-efficients and MSE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

#Loading dataset

df = pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/USA_Housing.csv")

display(df.head())

sns.pairplot(df)

#Preparing X and Y

X=df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y=df['Price']

#Splitting X and Y into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Building model

model = LinearRegression()

#Model Training

model.fit(X_train, y_train)

#Model Testing

y_pred=model.predict(X_test)

#Calculating MSE and Accuracy

mse =mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error:{mse}")
print(f"R-squared score: {r2}")

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

