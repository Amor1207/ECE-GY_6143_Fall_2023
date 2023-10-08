from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pandas as pd

df = pd.read_excel('brooklyn-bridge-automated-counts.xlsx')
df['hour'] = df['hour_beginning'].dt.hour
df['date'] = df['hour_beginning'].dt.date
df['day_name'] = df['hour_beginning'].dt.day_name()
df['day_no'] = df['hour_beginning'].dt.dayofweek
df['temperature'] = df['temperature'].fillna(method="ffill")
df['precipitation'] = df['precipitation'].fillna(method="ffill")
df['weather_summary'] = df['weather_summary'].fillna(method="ffill")
df['is_weekend'] = df['day_no'].isin([5, 6]).astype('int')
df['is_holiday'] = df['events'].notnull().astype('int')

X = np.array(df[['temperature', 'precipitation', 'hour', 'is_weekend', 'is_holiday']])
y = np.array(df['Pedestrians'])

kf = KFold(n_splits=nfold, shuffle=False)
kf.get_n_splits(X_tr)

# Iterate through the k-folds
for isplit, idx in enumerate(kf.split(X_tr)):
    idx_tr, idx_val = idx

    # Iterate through polynomial degrees
    for degree in range(1, nd):
        # Create polynomial features up to the desired degree
        poly = PolynomialFeatures(degree=degree)
        X_trans = poly.fit_transform(X)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X_trans, y, test_size=0.3, shuffle=False)
        # get "transformed" training and validation data
        x_train_dtest = X_tr[idx_tr]
        y_train_kfold = y_tr[idx_tr]
        x_val_dtest = X_tr[idx_val]
        y_val_kfold = y_tr[idx_val]

        # Train the model
        reg_dtest = LinearRegression().fit(x_train_dtest, y_train_kfold)

        # Compute R^2 for training and validation data
        r2_train[degree - 1, isplit] = r2_score(y_train_kfold, reg_dtest.predict(x_train_dtest))
        r2_val[degree - 1, isplit] = r2_score(y_val_kfold, reg_dtest.predict(x_val_dtest))

# Calculate the mean r2 score for validation
r2_mean = np.mean(r2_val, axis=1)
d_opt = np.argmax(r2_mean) + 1  # Adjusting for the 0-based index