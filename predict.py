import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

# Get CSV Data 
df = pd.read_csv (r'AAPL.CSV')
Dates = df['Date'].values
Prices = df['Adj Close'].values


#Help Functions
def get_performance (model_pred):
  #Function returns standard performance metrics
  print('  Mean Absolute Error:', metrics.mean_absolute_error(y_test, model_pred).round(4))  
  print('  Mean Squared Error:', metrics.mean_squared_error(y_test, model_pred).round(4))  
  print('  Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, model_pred)).round(4))
  print('---------------------------------')
  
def get_plot (model_pred):
  plt.scatter(model_pred, y_test, color="gray")
  plt.plot(y_test, y_test, color='red', linewidth=2)
  plt.show()

def plotPrediction(model_pred):
  plt.plot(model_pred, color="blue", linewidth=2)
  plt.show()


# Preprocess Data 

window_size=32
num_samples=len(Dates)-window_size

x_data = []
for x in range(num_samples):
  x_data.append(Prices[x : x + window_size])

y_data = []
for x in range(num_samples):
  y_data.append(Prices[x + window_size])

split_fraction=0.8
ind_split=int(split_fraction*num_samples)

x_train = x_data[:ind_split]
y_train = y_data[:ind_split]
x_test = x_data[ind_split:]
y_test = y_data[ind_split:]

# Train Models --------------------------

# 1. Predict Linear Regression
model_lr = LinearRegression()
model_lr.fit(x_train, y_train) # Trains on Data
y_pred_lr = model_lr.predict(x_test)

print("\nLinear regression")
get_performance(y_pred_lr)
get_plot(y_pred_lr)
plotPrediction(y_pred_lr)

# 2. Predict Ridge Regression
model_rr = Ridge()
model_rr.fit(x_train, y_train)
y_pred_rr = model_rr.predict(x_test)

print("Ridge Regression")
get_performance(y_pred_rr)
get_plot(y_pred_rr)
plotPrediction(y_pred_rr)

# 3. Predict Gradient Boosting
model_gb = GradientBoostingRegressor()
model_gb.fit(x_train, y_train)
y_pred_gb = model_gb.predict(x_test)

print("Gradient Boosting")
get_performance(y_pred_gb)
get_plot(y_pred_gb)
plotPrediction(y_pred_gb)

# 4. K Nearest Neighbor
model_knn = KNeighborsRegressor(n_neighbors=2)
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_test)

print("K Nearest Neighbor")
get_performance(y_pred_knn)
get_plot(y_pred_knn)
plotPrediction(y_pred_knn)