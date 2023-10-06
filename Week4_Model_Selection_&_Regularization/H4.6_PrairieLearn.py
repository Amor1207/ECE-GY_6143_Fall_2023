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

poly = PolynomialFeatures(degree=2)
X_trans = poly.fit_transform(X)

poly = PolynomialFeatures(degree=5)
X_trans = poly.fit_transform(X)

X_trans_names = poly.get_feature_names_out()

X_tr, X_ts, y_tr, y_ts = train_test_split(X_trans, y, test_size = 0.3, shuffle=False)


# use 10-fold cross validation (with `sklearn`'s `KFold`) to evaluate each `degree` from 0 to 5 (including 5)
# in an `sklearn` `LinearRegression` model, using `r2_score` for the metric.
# In your cross validation, you will save the validation R2 for each degree in an array called `r2_val`,
# and save the training R2 in an array called `r2_train`.

nd = 6
nfold = 10

r2_train = np.zeros((nd,nfold))
r2_val = np.zeros((nd,nfold))

kf = KFold(n_splits=nfold, shuffle=False)
kf.get_n_splits(X_tr)

# Convert data to sparse matrix
X_tr_sparse = csr_matrix(X_tr)

for isplit, idx in enumerate(kf.split(X_tr)):
    Itr, Its = idx
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=20)  # Select top 10 features
    X_tr_selected = selector.fit_transform(X_tr[Itr], y_tr[Itr])
    X_ts_selected = selector.transform(X_tr[Its])

    for degree in range(1, nd):  # Start loop from degree=1
        poly = PolynomialFeatures(degree=degree, interaction_only=True)  # Only interaction terms
        X_tr_poly = poly.fit_transform(X_tr_selected)
        X_ts_poly = poly.transform(X_ts_selected)
        reg = LinearRegression().fit(X_tr_poly, y_tr[Itr])
        r2_train[degree, isplit] = r2_score(y_tr[Itr], reg.predict(X_tr_poly))
        r2_val[degree, isplit] = r2_score(y_tr[Its], reg.predict(X_ts_poly))

# Calculate the mean r2 score for validation
r2_mean = np.mean(r2_val, axis=1)
d_opt = np.argmax(r2_mean) + 1  # Adjusting for the 0-based index