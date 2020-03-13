# Import the Libraries:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the csv file: "insurance.csv"

a = pd.read_csv("insurance.csv")

# some statistics about the data

a.describe()

# Converting the columns with text data into numeric data

a["southwest"]=a["region"]
a["southwest"].replace(["southwest","southeast","northwest","northeast"],[1,0,0,0],inplace=True)
a["southeast"]=a["region"]
a["southeast"].replace(["southwest","southeast","northwest","northeast"],[0,1,0,0],inplace=True)
a["northwest"]=a["region"]
a["northwest"].replace(["southwest","southeast","northwest","northeast"],[0,0,1,0],inplace=True)
a["northeast"]=a["region"]
a["northeast"].replace(["southwest","southeast","northwest","northeast"],[0,0,0,1],inplace=True)
a["female"]=a["sex"]
a["female"].replace(["female","male"],[1,0],inplace=True)
a["male"]=a["sex"]
a["male"].replace(["female","male"],[0,1],inplace=True)
a["yes"]=a["smoker"]
a["yes"].replace(["yes","no"],[1,0],inplace=True)
a["no"]=a["smoker"]
a["no"].replace(["yes","no"],[0,1],inplace=True)

a
a.head(10)

# Refering to the customers features data by "X", and refer to the label feature (expenses) by "y"

a.columns
y=a['expenses']
X=a[['age','female','male','bmi','children','southwest','southeast','northwest','northeast']]

#  Load the train_test_split function

from sklearn.model_selection import train_test_split

# Split the data into:
a training data set, and
a test data set. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=38)

# Import LinearRegression from sklearn.linear_model and create an instance of a LinearRegression() model named

from sklearn.linear_model import LinearRegression

# Fit the model to the training data

pq = LinearRegression()
pq.fit(X_train,y_train)

# Print the linear model's intercept and coefficients

print(pq.intercept_, pq.coef_)

# Use the trained model to predict the test data set

prediction=pq.predict(X_test)

 # Calculate:
the Mean Absolute Error, 
Mean Squared Error, and 
the Root Mean Squared Error.

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Show a histogram of the difference between the actual and predicted value of the test data set.

plt.hist(y_test-prediction)

# Use your own implementation of the Batch Gradient Descent to find the intercept and coeficients of the linear regression model

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
X = 3 * np.random.rand(1000, 1)
y = 5 + 4 * X + np.random.randn(1000, 1)
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.title("Generated Data")
plt.show()
X_b = np.c_[np.ones((1000, 1)), X]
alpha = 0.1
n_iterations = 1000
m = 100
W = np.random.randn(2,1)
for iteration in range(n_iterations):
    gradients = 1/m * X_b.T.dot(X_b.dot(W) - y)
    W = W - alpha * gradients
print(W)
def plot_gradient_descent(W, alpha):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            	y_predict = X_test_b.dot(W)
            	style = "b-" if iteration > 0 else "r--"
            	plt.plot(X_test, y_predict, style)
        gradients = 1/m * X_b.T.dot(X_b.dot(W) - y)
        W = W - alpha * gradients
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\alpha = {}$".format(alpha), fontsize=16)
    
# Demonstrate the effect of using different Learning Rate Parameter

np.random.seed(42)
W = np.random.randn(2,1)
X_test = np.array([[0], [2]])
X_test_b = np.c_[np.ones((2, 1)), X_test]
plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(W, alpha=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(W, alpha=0.1)
plt.subplot(133); plot_gradient_descent(W, alpha=0.9)
plt.show()

# Substitute the Stochastic Gradient Descent for the BGD

m = len(X_b)
np.random.seed(42)
alpha = 0.1
W = np.random.randn(2,1)
for i in range(1000):
       if i < 7:
             y_predict = X_test_b.dot(W)
             style = "b-" if i > 0 else "r--"
             if i == 6: style = "y-" 
             plt.plot(X_test, y_predict, style) 
       random_index = np.random.randint(m)
       xi = X_b[random_index:random_index+1]
       yi = y[random_index:random_index+1]
       gradients = 2 * xi.T.dot(xi.dot(W) - yi)
       W = W - alpha * gradients      # end of for loop
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18) 
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15]) 
plt.show()

# Perform Data Normalization

x_b = np.c_[np.ones((1000, 1)), X]
W = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 
print(W)

# Use the Normal Equation to solve the linear regression problem

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

END
