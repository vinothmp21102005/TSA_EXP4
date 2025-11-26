# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
###  Date: 23.09.2025
###  Reg No:212223240182

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/train.csv')

# Choose one column as pseudo time series
X = data['ram']   # you can replace with 'battery_power' or another numeric column

# Declare required variables and set figure size
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

# Plot original series
plt.plot(X)
plt.title('Original Data (RAM)')
plt.show()

# Plot ACF and PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for simulated ARMA(1,1)
plot_acf(ARMA_1)
plt.title("Simulated ARMA(1,1) ACF")
plt.show()

plot_pacf(ARMA_1)
plt.title("Simulated ARMA(1,1) PACF")
plt.show()

# Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for simulated ARMA(2,2)
plot_acf(ARMA_2)
plt.title("Simulated ARMA(2,2) ACF")
plt.show()

plot_pacf(ARMA_2)
plt.title("Simulated ARMA(2,2) PACF")
plt.show()

```

### OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="758" height="420" alt="image" src="https://github.com/user-attachments/assets/1ddb1432-2621-4529-8a7f-1f3c932b8e1f" />


Partial Autocorrelation

<img width="771" height="410" alt="image" src="https://github.com/user-attachments/assets/e801e47e-6465-45e3-b169-7a88f208da50" />


Autocorrelation

<img width="789" height="414" alt="image" src="https://github.com/user-attachments/assets/61ff630d-285d-44e8-a5af-a9ae7920ce56" />


SIMULATED ARMA(2,2) PROCESS:

<img width="762" height="422" alt="image" src="https://github.com/user-attachments/assets/5bd96b95-cf49-4026-bb5a-1762373a54f5" />

Autocorrelation

<img width="784" height="406" alt="image" src="https://github.com/user-attachments/assets/4537e034-fcf7-474f-8427-b699d8f40bfb" />

Partial Autocorrelation

<img width="777" height="441" alt="image" src="https://github.com/user-attachments/assets/4ab1a5df-9f6c-4655-ac07-8d858e4f04f7" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
