import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold

df = pd.read_csv("trainingData.csv", names=['number', 'latitude', 'longitude', 'x_1', 'x_2', 'y_1'])
X1=df.iloc[:,3]
X2=df.iloc[: ,4]
X=np.column_stack((X1,X2))
y=df.iloc[:,5]

fig = plt.figure()
ax = fig.add_subplot(111 , projection ='3d') 
ax.scatter(X1 ,X2 ,y)
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Y')
plt.title("Original Data")
plt.show()

poly_X = PolynomialFeatures(10).fit_transform(X)

Xtest = []
grid=np.linspace(0,700)
for i in range(0,60,3) :
  for j in range(0,700,5) :
    Xtest.append([i,j])
Xtest = np.array(Xtest)
poly_Xtest = PolynomialFeatures(10).fit_transform(Xtest)

reg = linear_model.LinearRegression().fit(X,y)

pred_y_graph = reg.predict(Xtest)
pred_y_data = reg.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111 , projection ='3d') 
ax.plot_trisurf(Xtest[:,0],Xtest[:,1], pred_y_graph)
ax.scatter(X1 ,X2 ,y, color='blue')
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Y')
plt.title("Prediction linear")
plt.show()

print("Linear:")
print("interecept = ", reg.intercept_ )
print("coefficients = ")
print(reg.coef_, "\n")

print("explained variance score ",explained_variance_score(y, pred_y_data))
print("max error ",max_error(y, pred_y_data))
print("R2 score ", r2_score(y, pred_y_data))
print("MSE ", mean_squared_error(y, pred_y_data))
print("\n\n")

lml = linear_model.Lasso(alpha = 0.1).fit(X,y)

pred_y_graph = lml.predict(Xtest)
pred_y_data = lml.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111 , projection ='3d') 
ax.plot_trisurf(Xtest[:,0],Xtest[:,1], pred_y_graph)
ax.scatter(X1 ,X2 ,y, color='blue')
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Y')
plt.title("Prediction lasso")
plt.show()
print("Lasso:")
print("interecept = ", reg.intercept_ )
print("coefficients = ")
print(reg.coef_, "\n")

print("explained variance score ",explained_variance_score(y, pred_y_data))
print("max error ",max_error(y, pred_y_data))
print("R2 score ", r2_score(y, pred_y_data))
print("MSE ", mean_squared_error(y, pred_y_data))
print("\n\n")


lmr = linear_model.Ridge(alpha = 0.001).fit(X,y)
pred_y_graph = lmr.predict(Xtest)
pred_y_data = lmr.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111 , projection ='3d') 
ax.plot_trisurf(Xtest[:,0],Xtest[:,1], pred_y_graph)
ax.scatter(X1 ,X2 ,y, color='blue')
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Y')
plt.title("Prediction ridge")
plt.show()
print("Ridge:")
print("interecept = ", reg.intercept_ )
print("coefficients = ")
print(reg.coef_, "\n")

print("explained variance score ",explained_variance_score(y, pred_y_data))
print("max error ",max_error(y, pred_y_data))
print("R2 score ", r2_score(y, pred_y_data))
print("MSE ", mean_squared_error(y, pred_y_data))
print("\n\n")

lmr = linear_model.LassoLars(alpha=0.1, normalize=False).fit(X,y)
pred_y_graph = lmr.predict(Xtest)
pred_y_data = lmr.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111 , projection ='3d') 
ax.plot_trisurf(Xtest[:,0],Xtest[:,1], pred_y_graph)
ax.scatter(X1 ,X2 ,y, color='blue')
ax.set_xlabel("X_1")
ax.set_ylabel("X_2")
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Y')
plt.title("Prediction Lasso LARS")
plt.show()
print("Lasso LARS:")
print("interecept = ", reg.intercept_ )
print("coefficients = ")
print(reg.coef_, "\n")

print("explained variance score ",explained_variance_score(y, pred_y_data))
print("max error ",max_error(y, pred_y_data))
print("R2 score ", r2_score(y, pred_y_data))
print("MSE ", mean_squared_error(y, pred_y_data))
print("\n\n")

# lmr = linear_model.Ridge(alpha = 100).fit(poly_X,y)
# pred_y_graph = lmr.predict(poly_Xtest)
# pred_y_data = lmr.predict(poly_X)

# fig = plt.figure()
# ax = fig.add_subplot(111 , projection ='3d') 
# ax.plot_trisurf(Xtest[:,0],Xtest[:,1], pred_y_graph)
# ax.scatter(X1 ,X2 ,y, color='blue')
# ax.set_xlabel("X_1")
# ax.set_ylabel("X_2")
# ax.zaxis.set_rotate_label(False) 
# ax.set_zlabel('Y')
# plt.title("Prediction ridge")
# plt.show()
# print("Ridge:")
# print("interecept = ", reg.intercept_ )
# print("coefficients = ")
# print(reg.coef_, "\n")

# print("explained variance score ",explained_variance_score(y, pred_y_data))
# print("max error ",max_error(y, pred_y_data))
# print("R2 score ", r2_score(y, pred_y_data))
# print("\n\n")
 
  
C3 = [0.1, 1.0, 10, 20, 25, 30, 40, 50, 75, 100, 1000, 10000]
mean_error=[]; std_error=[]

for x in C3: 
  lml = linear_model.Lasso(alpha = 1/x)
  kf = KFold(n_splits=5)
  temp=[]
  
  for train, test in kf.split(X):
    lml.fit(X[train], y[train])
    ypred = lml.predict(X[test])
    temp.append(r2_score(y[test],ypred))
    
  mean_error.append(np.array(temp).mean())
  std_error.append(np.array(temp).std())

C_display = [100, 50, 30, 15]
for x in C_display:
  plt.errorbar(C3,mean_error,yerr=std_error)
  plt.xlabel("C"); plt.ylabel("R2 score")
  plt.xlim((0,x))
  plt.grid()
  title = "Lasso R2 score vs C"
  plt.title(title)
  plt.show()

C4 = [0.0001, 0.001, 0.01, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 10, 100]
mean_error=[]; std_error=[]

for x in C4: 
  lmr = linear_model.Ridge(alpha = 1/x)
  kf = KFold(n_splits=5)
  temp=[]
  
  for train, test in kf.split(X):
    lmr.fit(X[train], y[train])
    ypred = lmr.predict(X[test])
    temp.append(r2_score(y[test],ypred))
    
  mean_error.append(np.array(temp).mean())
  std_error.append(np.array(temp).std())

C_display_2 = [10, 1, 0.5, 0.2, 0.1]
for x in C_display_2:
  plt.errorbar(C4,mean_error,yerr=std_error)
  plt.xlabel("C"); plt.ylabel("R2 score")
  plt.xlim((0,x))
  plt.grid()
  title = "Ridge R2 score vs C"
  plt.title(title)
  plt.show()