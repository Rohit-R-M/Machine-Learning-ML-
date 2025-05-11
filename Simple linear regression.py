#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Loading the data

# In[ ]:


df = pd.read_csv('C:/Users/3yearb1/Desktop/ML_datasets/Salary_dataset.csv')


# In[ ]:


display(df.head())


# In[8]:


df.info()


# In[15]:


df.describe()


# In[16]:


sns.regplot(x=df['YearsExperience'],y=df['Salary'])
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.show()


# Preparing X and Y

# In[37]:


X=df['YearsExperience'].values.reshape(-1, 1)
y=df['Salary']


# Splitting X and Y into training and testing sets

# In[39]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Model Building

# In[40]:


model = LinearRegression()


# Model Training

# In[ ]:


model.fit(X_train, y_train)


# Testing Model

# In[44]:


y_pred=model.predict(X_test)


# Calculating MSE and Accuracy

# In[45]:


mse =mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error:{mse}")
print(f"R-squared score: {r2}")


# In[47]:


plt.scatter(X_test, y_test, color='blue', label='Actual') 
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.legend()
plt.show()


# In[48]:


print(f"intercept: {model.intercept_}")
print(f"Cofficient:{model.coef_[0]}" )

