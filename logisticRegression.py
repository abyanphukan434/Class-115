import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('tempData.csv')
temperature_list = df['Temperature'].tolist()

melted_list = df['Melted'].tolist()

fig = px.scatter(x = temperature_list, y = melted_list)

fig.show()

temperature_array = np.array(temperature_list)

melted_array = np.array(melted_list)

m, c = np.polyfit(temperature_array, melted_array, 1)

y = []

for x in temperature_array:
  y_value = m * x + c
  y.append(y_value)

fig = px.scatter(x = temperature_array, y = melted_array)

fig.update_layout(shapes = [
                            dict(
                                type = 'line',
                                y0 = min(y), y1 = max(y),
                                x0 = min(temperature_array), x1 = max(temperature_array)
                            )
])

fig.show()

X = np.reshape(temperature_list, (len(temperature_list), 1))

Y = np.reshape(melted_list, (len(melted_list), 1))

lr = LogisticRegression()

lr.fit(X, Y)

plt.figure()

plt.scatter(X.ravel(), Y, color = 'black', zorder = 20)

def model(x):
  return 1/(1 + np.exp(-x))

X_test = np.linspace(0, 5000, 10000)

chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color = 'red', linewidth = 3)

plt.axhline(y = 0, color = 'k', linestyle = '-')

plt.axhline(y = 1, color = 'k', linestyle = '-')

plt.axhline(y = 0.5, color = 'b', linestyle = '--')

plt.axvline(x = X_test[6843], color = 'b', linestyle = '--')

plt.ylabel('y')

plt.xlabel('X')

plt.xlim(3400, 3450)

plt.show()

temp = float(input('Enter the temperature:'))

chances = model(temp * lr.coef_ + lr.intercept_).ravel()[0]

if chances <= 0.01:
  print('Tungsten will not be melted.')
elif chances >= 1:
  print('Tungsten will be melted.')
elif chances < 0.5:
  print('Tungsten might not get melted.')
else:
  print('Tungsten may get melted.')