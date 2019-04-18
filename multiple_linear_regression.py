# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import OLS

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categories='auto')
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = OLS(endog=y, exog=X_opt).fit()
X_opt = X[:, [0, 1, 3, 4]]
regressor_OLS = OLS(endog=y, exog=X_opt).fit()
X_opt = X[:, [0, 1, 3]]
regressor_OLS = OLS(endog=y, exog=X_opt).fit()
X_opt = X[:, [0, 1]]
regressor_OLS = OLS(endog=y, exog=X_opt).fit()
X_opt = X[:, [0, 1]]
regressor_OLS = OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
