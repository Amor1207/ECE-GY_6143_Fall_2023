# -*- coding: utf-8 -*-
"""“2-advertising-hw.ipynb”的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t2R7EKSezOOl3hpGkQyfm0NLQqfhc9LH

# Assignment: Linear regression on the Advertising data

*Fraida Fund*

Submit answers to the questions in PrairieLearn as you work through this notebook.

To illustrate principles of linear regression, we are going to use some data from the textbook “An Introduction to Statistical Learning withApplications in R” (Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani) (available via NYU Library).

The dataset is described as follows:

> Suppose that we are statistical consultants hired by a client to provide advice on how to improve sales of a particular product. The `Advertising` data set consists of the sales of that product in 200 different markets, along with advertising budgets for the product in each of those markets for three different media: TV, radio, and newspaper.
>
> …
>
> It is not possible for our client to directly increase sales of the product. On the other hand, they can control the advertising expenditure in each of the three media. Therefore, if we determine that there is an association between advertising and sales, then we can instruct our client to adjust advertising budgets, thereby indirectly increasing sales. In other words, our goal is to develop an accurate model that can be used to predict sales on the basis of the three media budgets.

Sales are reported in thousands of units, and TV, radio, and newspaper budgets, are reported in thousands of dollars.

For this assignment, you will fit a linear regression model to a small dataset. You will iteratively improve your linear regression model by examining the residuals at each stage, in order to identify problems with the model.

Make sure to include your name and net ID in a text cell at the top of the notebook.
"""

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

"""### 0. Read in and pre-process data

In this section, you will read in the “Advertising” data, and make sure it is loaded correctly. Visually inspect the data using a pairplot, and note any meaningful observations. In particular, comment on which features appear to be correlated with product sales, and which features appear to be correlated with one another. Then, split the data into training data (70%) and test data (30%).

**The code in this section is provided for you**.

#### Read in data
"""

!wget 'https://www.statlearning.com/s/Advertising.csv' -O 'Advertising.csv'

df  = pd.read_csv('Advertising.csv', index_col=0)
df.head()

"""Note that in this dataset, the first column in the data file is the row label; that’s why we use `index_col=0` in the `read_csv` command. If we would omit that argument, then we would have an additional (unnamed) column in the dataset, containing the row number.

(You can try removing the `index_col` argument and re-running the cell above, to see the effect and to understand why we used this argument.)

#### Visually inspect the data
"""

sns.pairplot(df);

"""The most important panels here are on the bottom row, where `sales` is on the vertical axis and the advertising budgets are on the horizontal axes.

Looking at this row, it appears that TV ad spending and radio ad spending are likely to be useful predictive features for `sales`; for newspaper ad spending, it is not clear from the pairplot whether there is a relationship.

#### Split up data

We will use 70% of the data for training and the remaining 30% to evaluate the regression model on data *not* used for training.
"""

train, test = train_test_split(df, test_size=0.3, random_state=9)

"""We will set the `random_state` to a constant so that every time you run this notebook, exactly the same data points will be assigned to test vs. training sets. This is helpful in the debugging stage."""

train.info()

test.info()

"""### 1. Fit simple linear regression models

Use the training data to fit a simple linear regression to predict product sales, for each of three features: TV ad budget, radio ad budget, and newspaper ad budget. In other words, you will fit *three* regression models, with each model being trained on one feature. For each of the three regression models, create a plot of the training data and the regression line, with product sales ($y$) on the vertical axis and the feature on which the model was trained ($x$) on the horizontal axis.

Also, for each regression model, print the intercept and coefficients, and compute the MSE and R2 on the training data, and MSE and R2 on the test data.

Comment on the results. Which type of ad spending seems to be associated with the largest increase in product sales? Which regression model is most effective at predicting product sales?

**The code in this section is provided for you**. However, you will need to add comments, observations, and answers to the questions.

#### Fit a simple linear regression
"""

reg_tv    = LinearRegression().fit(train[['TV']], train['sales'])
reg_radio = LinearRegression().fit(train[['radio']], train['sales'])
reg_news  = LinearRegression().fit(train[['newspaper']], train['sales'])

"""#### Look at coefficients"""

print("TV       : ", reg_tv.coef_[0], reg_tv.intercept_)
print("Radio    : ", reg_radio.coef_[0], reg_radio.intercept_)
print("Newspaper: ", reg_news.coef_[0], reg_news.intercept_)

"""#### Plot data and regression line"""

fig = plt.figure(figsize=(12,3))

plt.subplot(1,3,1)
sns.scatterplot(data=train, x="TV", y="sales");
sns.lineplot(data=train, x="TV", y=reg_tv.predict(train[['TV']]), color='red');

plt.subplot(1,3,2)
sns.scatterplot(data=train, x="radio", y="sales");
sns.lineplot(data=train, x="radio", y=reg_radio.predict(train[['radio']]), color='red');

plt.subplot(1,3,3)
sns.scatterplot(data=train, x="newspaper", y="sales");
sns.lineplot(data=train, x="newspaper", y=reg_news.predict(train[['newspaper']]), color='red');

"""#### Compute R2, MSE for simple regression"""

y_pred_tr_tv    = reg_tv.predict(train[['TV']])
y_pred_tr_radio = reg_radio.predict(train[['radio']])
y_pred_tr_news  = reg_news.predict(train[['newspaper']])

r2_tr_tv    = metrics.r2_score(train['sales'], y_pred_tr_tv)
r2_tr_radio = metrics.r2_score(train['sales'], y_pred_tr_radio)
r2_tr_news  = metrics.r2_score(train['sales'], y_pred_tr_news)
print("TV       : ", r2_tr_tv)
print("Radio    : ", r2_tr_radio)
print("Newspaper: ", r2_tr_news)

mse_tr_tv    = metrics.mean_squared_error(train['sales'], y_pred_tr_tv)
mse_tr_radio = metrics.mean_squared_error(train['sales'], y_pred_tr_radio)
mse_tr_news  = metrics.mean_squared_error(train['sales'], y_pred_tr_news)
print("TV       : ", mse_tr_tv)
print("Radio    : ", mse_tr_radio)
print("Newspaper: ", mse_tr_news)

y_pred_ts_tv    = reg_tv.predict(test[['TV']])
y_pred_ts_radio = reg_radio.predict(test[['radio']])
y_pred_ts_news  = reg_news.predict(test[['newspaper']])

r2_ts_tv    = metrics.r2_score(test['sales'], y_pred_ts_tv)
r2_ts_radio = metrics.r2_score(test['sales'], y_pred_ts_radio)
r2_ts_news  = metrics.r2_score(test['sales'], y_pred_ts_news)
print("TV       : ", r2_ts_tv)
print("Radio    : ", r2_ts_radio)
print("Newspaper: ", r2_ts_news)

mse_ts_tv    = metrics.mean_squared_error(test['sales'], y_pred_ts_tv)
mse_ts_radio = metrics.mean_squared_error(test['sales'], y_pred_ts_radio)
mse_ts_news  = metrics.mean_squared_error(test['sales'], y_pred_ts_news)
print("TV       : ", mse_ts_tv)
print("Radio    : ", mse_ts_radio)
print("Newspaper: ", mse_ts_news)

"""### 2. Explore the residuals for the single linear regression models

We know that computing MSE or R2 is not sufficient to diagnose a problem with a linear regression.

Create some additional plots as described below to help you identify any problems with the regression. Use training data for all of the items below.

For each of the three regression models, you will compute the residuals ($y - \hat{y}$). Then, you’ll create three plots - each with three subplots, one for each regression model - as follows:

**Plot 1**: Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis) *and* that all three subplots use the same scale. Label each axes, and each plot. What would you expect this plot to look like for a model that explains the data well?

**Plot 2**: Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Use the same vertical scale for all three subplots, and the same horizontal scale for all three subplots (but the vertical scale and the horizontal scale will not be the same as one another!). Comment on your observations. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales?

**Plot 3**: For each of the three regression models AND each of the three features, create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis. This plot will include nine subplots in total, for every combination of regression model and feature. Use the same vertical scale for all subplots (but the horizontal scale will depend on the feature!) Make sure to clearly label each axis, and also label each subplot with a title that indicates which regression model it uses. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features?

**The code in this section is not provided for you**. You will need to write code, as well as comments, observations, and answers to the questions.

------------------------------------------------------------------------

Note that in general, to earn full credit, plots must:

-   Be readable (especially text size).
-   Have a label on each axis.
-   Have an appropriate range for each axis. When there are multiple subplots, if the goal is to compare similar things in different subplots, in most cases it is appropriate for them all to use the same range.
-   If there are multiple subplots, or multiple data series in the same plot, it must be made clear which is which.

**Plot 1**: Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis) *and* that all three subplots use the same scale. Label each axes, and each plot. What would you expect this plot to look like for a model that explains the data well?
"""

# Calculating predicted sales for each regression model again
train['pred_sales_tv'] = reg_tv.predict(train[['TV']])
train['pred_sales_radio'] = reg_radio.predict(train[['radio']])
train['pred_sales_news'] = reg_news.predict(train[['newspaper']])

# Plotting again
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# TV Model
axes[0].scatter(train['sales'], train['pred_sales_tv'], alpha=0.6)
axes[0].plot([0, 30], [0, 30], '--', color='red')
axes[0].set_title('TV Model')
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')

# Radio Model
axes[1].scatter(train['sales'], train['pred_sales_radio'], alpha=0.6)
axes[1].plot([0, 30], [0, 30], '--', color='red')
axes[1].set_title('Radio Model')
axes[1].set_xlabel('Actual Sales')
axes[1].set_ylabel('Predicted Sales')

# Newspaper Model
axes[2].scatter(train['sales'], train['pred_sales_news'], alpha=0.6)
axes[2].plot([0, 30], [0, 30], '--', color='red')
axes[2].set_title('Newspaper Model')
axes[2].set_xlabel('Actual Sales')
axes[2].set_ylabel('Predicted Sales')

plt.tight_layout()
plt.show()

train

"""**Plot 2**: Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Use the same vertical scale for all three subplots, and the same horizontal scale for all three subplots (but the vertical scale and the horizontal scale will not be the same as one another!). Comment on your observations. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales?"""

# Calculating residuals for each regression model
train['residual_tv'] = train['sales'] - train['pred_sales_tv']
train['residual_radio'] = train['sales'] - train['pred_sales_radio']
train['residual_news'] = train['sales'] - train['pred_sales_news']

# Plotting residuals
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# TV Model
axes[0].scatter(train['sales'], train['residual_tv'], alpha=0.6)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('TV Model')
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Residuals')

# Radio Model
axes[1].scatter(train['sales'], train['residual_radio'], alpha=0.6)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Radio Model')
axes[1].set_xlabel('Actual Sales')
axes[1].set_ylabel('Residuals')

# Newspaper Model
axes[2].scatter(train['sales'], train['residual_news'], alpha=0.6)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_title('Newspaper Model')
axes[2].set_xlabel('Actual Sales')
axes[2].set_ylabel('Residuals')

plt.tight_layout()
plt.show()

train

"""**Plot 3**: For each of the three regression models AND each of the three features, create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis. This plot will include nine subplots in total, for every combination of regression model and feature. Use the same vertical scale for all subplots (but the horizontal scale will depend on the feature!) Make sure to clearly label each axis, and also label each subplot with a title that indicates which regression model it uses. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features?"""

# Plotting residuals against each feature
fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey=True)

features = ['TV', 'radio', 'newspaper']
models = ['tv', 'radio', 'news']

for i, feature in enumerate(features):
    for j, model in enumerate(models):
        residuals_column = f"residual_{model}"
        axes[i, j].scatter(train[feature], train[residuals_column], alpha=0.6)
        axes[i, j].axhline(0, color='red', linestyle='--')
        axes[i, j].set_title(f'{model.capitalize()} Model vs {feature}')
        axes[i, j].set_xlabel(feature)
        axes[i, j].set_ylabel('Residuals')

plt.tight_layout()
plt.show()

"""### 3. Try a multiple linear regression

Next, fit a multiple linear regression to predict product sales, using all three features to train a single model: TV ad budget, radio ad budget, and newspaper ad budget.

Print the intercept and coefficients, and compute the MSE and R2 on the training data, and MSE and R2 on the test data. Comment on the results. Make sure to explain any differences between the coefficients of the multiple regression model, and the coefficients of the three simple linear regression models. If they are different, why?

**The code in the first part of this section is provided for you**. However, you will need to add comments, observations, and answers to the questions.

Also repeat the analysis of part (3) for this regression model. Use training data for all of these items:

**Plot 1**: Create a scatter plot of predicted sales ($\hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Make sure both axes use the same scale (the range of the vertical axis should be the same as the range of the horizontal axis). Label each axes. Does this model explain the data more effectively than the simple linear regressions from the previous section?

**Plot 2**: Create a scatter plot with the residuals ($y - \hat{y}$) on the vertical axis, and actual sales ($y$) on the horizontal axis. Comment on your observations. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to actual sales?

**Plot 3**: For each of the three features, plot the residuals ($y - \hat{y}$) on the vertical axis, and the feature ($x$) on the horizontal axis. Make sure to clearly label each axis. Is there a pattern in the residuals (and if so, what might it indicate), or do they appear to have no pattern with respect to each of the three features?

------------------------------------------------------------------------

Note that in general, to earn full credit, plots must:

-   Be readable (especially text size).
-   Have a label on each axis.
-   Have an appropriate range for each axis. When there are multiple subplots, if the goal is to compare similar things in different subplots, in most cases it is appropriate for them all to use the same range.
-   If there are multiple subplots, or multiple data series in the same plot, it must be made clear which is which.

#### Fit a multiple linear regression
"""

reg_multi = LinearRegression().fit(train[['TV', 'radio', 'newspaper']], train['sales'])

"""#### Look at coefficients"""

print("Coefficients (TV, radio, newspaper):", reg_multi.coef_)
print("Intercept: ", reg_multi.intercept_)

"""#### Compute R2, MSE for multiple regression"""

y_pred_tr_multi = reg_multi.predict(train[['TV', 'radio', 'newspaper']])

r2_tr_multi  = metrics.r2_score(train['sales'], y_pred_tr_multi)
mse_tr_multi = metrics.mean_squared_error(train['sales'], y_pred_tr_multi)

print("Multiple regression R2 of Train:  ", r2_tr_multi)
print("Multiple regression MSE of Train: ", mse_tr_multi)

y_pred_ts_multi = reg_multi.predict(test[['TV', 'radio', 'newspaper']])

r2_ts_multi  = metrics.r2_score(test['sales'], y_pred_ts_multi)
mse_ts_multi = metrics.mean_squared_error(test['sales'], y_pred_ts_multi)

print("Multiple regression R2 of Test:  ", r2_ts_multi)
print("Multiple regression MSE of Test: ", mse_ts_multi)

### Plot 1 ###

# Calculating predicted sales for multiple regression model
train['pred_sales_multiple'] = reg_multi.predict(train[['TV','radio','newspaper']])

# Plotting again
fig, ax = plt.subplots(figsize=(6, 6))

# Multiple Model
ax.scatter(train['sales'], train['pred_sales_multiple'], alpha=0.6)
ax.plot([0, 30], [0, 30], '--', color='red')
ax.set_title('Multiple Model')
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')

plt.tight_layout()
plt.show()

### Plot 2 ###

# Calculating residuals for multiple regression model
train['residual_multiple'] = train['sales'] - train['pred_sales_multiple']

# Plotting again
fig, ax = plt.subplots(figsize=(6, 6))

# Multiple Model
ax.scatter(train['sales'], train['residual_multiple'], alpha=0.6)
ax.axhline(0, color='red', linestyle='--')

# Set y-axis limits to ensure the red dashed line is in the middle
max_residual = abs(train['residual_multiple']).max()
ax.set_ylim(-max_residual, max_residual)

ax.set_title('Multiple Model')
ax.set_xlabel('Residuals')
ax.set_ylabel('Predicted Sales')

plt.tight_layout()
plt.show()

### Plot 3 ###

# Plotting residuals
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False,sharey=True)

# Set y-axis limits to ensure the red dashed line is in the middle
max_residual = abs(train['residual_multiple']).max()


# TV Model
axes[0].scatter(train['TV'], train['residual_multiple'], alpha=0.6)
axes[0].set_ylim(-max_residual, max_residual)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('TV Model')
axes[0].set_xlabel('TV')
axes[0].set_ylabel('Residuals')

# Radio Model
axes[1].scatter(train['radio'], train['residual_multiple'], alpha=0.6)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Radio Model')
axes[1].set_xlabel('Radio')
axes[1].set_ylabel('Residuals')

# Newspaper Model
axes[2].scatter(train['newspaper'], train['residual_multiple'], alpha=0.6)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_title('Newspaper Model')
axes[2].set_xlabel('Newspaper')
axes[2].set_ylabel('Residuals')

plt.tight_layout()
plt.show()

train[['TV', 'radio', 'newspaper']].corr()

"""### 4. Linear regression with interaction terms

Our multiple linear regression includes additive effects of all three types of advertising media. However, it does not include *interaction* effects, in which combining different types of advertising media together results in a bigger boost in sales than just the additive effect of the individual media.

The pattern in the residuals plots from parts (1) through (3) suggest that a model including an interaction effect may explain sales data better than a model including additive effects. Add four columns to each data frame (`train` and `test`):

-   `newspaper` $\times$ `radio` (name this column `newspaper_radio`)
-   `TV` $\times$ `radio` (name this column `TV_radio`)
-   `newspaper` $\times$ `TV` (name this column `newspaper_TV`)
-   `newspaper` $\times$ `radio` $\times$ `TV` (name this column `newspaper_radio_TV`)

Note: you can use the `assign` function in `pandas` ([documentation here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html)) to create a new column and assign a value to it using operations on other columns.

Then, train a linear regression model on all seven features: the three types of ad budgets, and the four interaction effects. Repeat the analysis of part (3) for the model including interaction effects. Are the interaction effects helpful for explaining the effect of ads on product sales? Are there any patterns evident in the residual plots that suggest further opportunities for improving the model?

**The code in this section is not provided for you**. You will need to write code, in addition to comments, observations, and answers to the questions.

------------------------------------------------------------------------

Note that in general, to earn full credit, plots must:

-   Be readable (especially text size).
-   Have a label on each axis.
-   Have an appropriate range for each axis. When there are multiple subplots, if the goal is to compare similar things in different subplots, in most cases it is appropriate for them all to use the same range.
-   If there are multiple subplots, or multiple data series in the same plot, it must be made clear which is which.
"""

