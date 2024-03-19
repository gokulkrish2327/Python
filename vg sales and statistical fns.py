import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('vgsales.csv')
print(data.head(3))

# independent variable X and dependent variable (y)
X = data[['Year']]
y = data['Global_Sales']

X = sm.add_constant(X)
# Example: Replace missing values with the mean
X.fillna(X.mean(), inplace=True)
# Example: Replace infinite values with a large number
X.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)

# 1. fit the regression model and estimate regression parameters using OLS
model = sm.OLS(y, X).fit()
print(model.summary())

#regression parameters
coefficients = model.params
print("Coefficients:", coefficients)

# 2.Testing the  individual coefficient 
hypotheses = 'Year = 0'  
t_test_result = model.t_test(hypotheses)
print(t_test_result)


"""To test the overall performance of a regression model in a Python dataset,
you can use various metrics that evaluate how well the model fits the data and makes predictions.
Some commonly used metrics include:
R-square
adjusted R-square
MSE or RMSE
MAE
F-statistic and p-value"""

# 4 finding R-square
rsquared = model.rsquared
print("R-squared:", rsquared)

# adj R- sqr
adjusted_r_squared = model.rsquared_adj
print("Adjusted R-squared:", adjusted_r_squared)

# MSE and RMSE
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# MAE
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)

# F-stats and p value
f_statistic = model.fvalue
p_value = model.f_pvalue
print("F-statistic:", f_statistic)
print("p-value:", p_value)

#5. confidence interval for regression coeff.
confidence_intervals = model.conf_int()
print(confidence_intervals)

# 6 checking linearity
plt.scatter(X['Year'], y)
plt.xlabel('Year')
plt.ylabel('Global Sales')
plt.title('Scatter Plot: Year vs. Global Sales')
plt.show()

# 7 checking normality
residuals = model.resid
plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Normality(histogram)')
plt.show()

# 8 checking outliers
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Outliers')
plt.show()

#9. residual plots(q-q plot)
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

#10. model validation

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores  # Convert to positive values
avg_cv_mse = cv_mse_scores.mean()

print("Cross-Validation Mean Squared Error:", avg_cv_mse)
