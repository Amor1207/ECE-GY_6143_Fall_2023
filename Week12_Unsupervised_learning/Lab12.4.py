import numpy as np
import random

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dat = np.genfromtxt('data.csv',delimiter=',')
X = dat[:, :-1]
y = dat[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Use the `sklearn` implementation of `KFold` to iterate over candidate models. In your `KFold`, use 5 splits, and don't shuffle the data (you already shuffled it when dividing into training and test.)
# Use the `sklearn` implementation of `PCA` to transform the data. Pass `random_state = 42` to `PCA` so that your result will match the auto-grader's.
# Use the `sklearn` implementation of `SVC` to classify the data using the first `n_comp` principal components.  Pass `random_state = 42` to `SVC` so that your result will match the auto-grader's.

kf = KFold(n_splits=5, shuffle=False)
average_accuracy = np.zeros(X_train.shape[1])
average_validation = np.zeros(X_train.shape[1])
pca = PCA(n_components=X_train.shape[1], random_state=42)
for n_comp in range(1,X_train.shape[1]+1):
    # pca = PCA(n_components=n_comp, random_state=42)
    fold_acc = np.zeros(kf.get_n_splits())
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        pca.fit(X_train_fold)
        X_train_fold_pca = pca.transform(X_train_fold)[:,:n_comp]
        X_test_fold_pca = pca.transform(X_test_fold)[:,:n_comp]
        svc = SVC(random_state=42)
        svc.fit(X_train_fold_pca, y_train_fold)
        y_pred = svc.predict(X_test_fold_pca)
        fold_acc[i] = accuracy_score(y_test_fold, y_pred)
    average_accuracy[n_comp-1] = fold_acc.mean()
    average_validation[n_comp-1] = fold_acc.std()/np.sqrt(kf.get_n_splits()-1)


# Compute the mean validation accuracy and the standard error of the mean validation accuracy across the folds.


# compute the optimal value of n_comp, and save this in n_pca_opt.

n_pca_opt = np.argmax(average_accuracy)+1

# compute the optimal n_comp according to the one-SE rule, and save this in n_pca_one_se.

max_avg_accuracy = np.max(average_accuracy)
max_se = average_validation[np.argmax(average_accuracy)]

# Calculate the threshold for the one-SE rule
threshold = max_avg_accuracy - max_se

# Apply the one-SE rule to find the simplest model within one standard error of the best model
n_pca_one_se = np.where(average_accuracy >= threshold)[0][0] + 1  # Adding 1 because array indices start from 0

print(n_pca_opt,n_pca_one_se)