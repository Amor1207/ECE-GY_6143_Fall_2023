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

poly = PolynomialFeatures(degree=5)
X_trans = poly.fit_transform(X)

X_trans.shape

X_trans_names = poly.get_feature_names()

X_tr, X_ts, y_tr, y_ts = train_test_split(X_trans, y, test_size = 0.3, shuffle=False)

X_tr.shape

nd = 6
nfold = 10
r2_train = np.zeros((nd, nfold))
r2_val = np.zeros((nd, nfold))

kf = KFold(n_splits=nfold, shuffle=False)
num_poly = [1, 6, 21, 56, 126, 252]
for isplit, (idx_tr, idx_val) in enumerate(kf.split(X_tr)):

    for degree in range(nd):
        X_train_fold = X_tr[idx_tr, :num_poly[degree]]
        X_val_fold = X_tr[idx_val, :num_poly[degree]]

        # Train the model
        reg_dtest = LinearRegression().fit(X_train_fold, y_tr[idx_tr])

        # Compute R^2 for training and validation data
        r2_train[degree, isplit] = r2_score(y_tr[idx_tr], reg_dtest.predict(X_train_fold))
        r2_val[degree, isplit] = r2_score(y_tr[idx_val], reg_dtest.predict(X_val_fold))

# Calculate the mean r2 score for validation
r2_mean = np.mean(r2_val, axis=1)
d_opt = np.argmax(r2_mean)   # Adjusting for the 0-based index