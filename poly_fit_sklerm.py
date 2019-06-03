from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

lin_reg = LinearRegression()
poly_reg = PolynomialFeatures(degree=5)

X_poly = poly_reg.fit_transform(X)
#print(X_poly)
poly_reg.fit(X_poly,Y)
lin_reg.fit(X_poly,Y)

#print(dataset)
#print(X)
#print(Y)

plt.scatter(X,Y)
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)))

plt.show()
