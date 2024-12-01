import pandas as pd


train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_salary_train.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_salary_test.csv")

train_data.head()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axis = plt.subplots(2, 1, figsize = (5, 7))
total_data = pd.concat([train_data, test_data])

sns.regplot(ax = axis[0], data = total_data, x = "YearsExperience", y = "Salary")
sns.heatmap(total_data[["Salary", "YearsExperience"]].corr(), annot = True, fmt = ".2f", ax = axis[1], cbar = False)

plt.tight_layout()

plt.show()



X_train = train_data.drop(["Salary"], axis = 1)
y_train = train_data["Salary"]
X_test = test_data.drop(["Salary"], axis = 1)
y_test = test_data["Salary"]


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


print(f"Intercept (a): {model.intercept_}")
print(f"Coefficients (b): {model.coef_}")

y_pred = model.predict(X_test)
y_pred


To compare the predicted value of the original, we can easily perform a comparative plot as follows:

fig, axis = plt.subplots(1, 2, figsize = (5, 3.5))
total_data = pd.concat([train_data, test_data])

# We use the parameters adjusted in the training to draw the regression line in the plots
regression_equation = lambda x: 26354.43069701219 + 9277.78307971 * x

sns.scatterplot(ax = axis[0], data = test_data, x = "YearsExperience", y = "Salary")
sns.lineplot(ax = axis[0], x = test_data["YearsExperience"], y = regression_equation(test_data["YearsExperience"]))
sns.scatterplot(ax = axis[1], x = test_data["YearsExperience"], y = y_pred)
sns.lineplot(ax = axis[1], x = test_data["YearsExperience"], y = regression_equation(test_data["YearsExperience"])).set(ylabel = None)

plt.tight_layout()

plt.show()

To calculate the effectiveness of the model we will use the mean squared error (MSE) and the coefficient of determination (
), one of the most popular metrics


from sklearn.metrics import mean_squared_error, r2_score

print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")

The lower the RMSE value, the better the model. A perfect model (a hypothetical model that can always predict the exact expected value) would have a value for this metric of 0. We observe that there is a slippage of 37 million, so we could understand that it is very bad. If we rely on the 
 value, we observe that it is 95%, a very high value, and then 95% of the data is explained by the model, so it is satisfactory.
 
 
 Multiple linear regression
 
 
 To exemplify the implementation of a simple multiple regression model, we will use a data set with a few instances that has been previously treated with a full EDA.
 
 
 import pandas as pd
import matplotlib.pyplot as plt 

train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_weight-height_train.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/clean_weight-height_test.csv")

train_data.head()

For this problem, we want to calculate the weight (weight) as a function of the height (height) and gender (gender) of the person. Therefore, weight will be the dependent variable (target variable), and height and gender will be the independent variables (predictor variables). Since this is a continuous numerical prediction, we have to solve this with a multiple logistic regression model.




 
