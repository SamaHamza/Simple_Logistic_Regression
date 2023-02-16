
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import expit # This is a special kind of the logistic activation function

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)

# Splitting the dataset into X and y values
X = np.random.normal(size=n_samples)
y = (X > 0).astype(float)
X[X > 0] *= 4
X += 0.3 * np.random.normal(size=n_samples)
X = X[:, np.newaxis]

# Building the logistic regressiton model:
clf = LogisticRegression(C=1e5)

# Here we train the model:
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(6, 5))
plt.clf()

#plt.scatter(X.ravel(), y, color="red", zorder=20) # return a flatten array that represent the X values
X_test = np.linspace(-5, 10, 300) # Return evenly spaced numbers over a specified interval

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color="navy", linewidth=3)

ols = LinearRegression()
ols.fit(X, y)
plt.plot( X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1,color="yellow") #Linear regression model fitting


plt.axhline(0.5, color="green") # Threshold line

plt.ylabel("y")
plt.xlabel("X")
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-0.25, 1.25)
plt.xlim(-4, 10)
plt.legend(
    ( "Logistic Regression Model","Linear Regression Model"),
    loc="lower right",
    fontsize="medium",
)
plt.tight_layout()
plt.show()
