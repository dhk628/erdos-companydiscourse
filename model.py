import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pandas.api.types import CategoricalDtype

df = pd.read_parquet('Data/Vectorized/costco_2021_reviews_filtered_vectorized_master_en_float32_nodup.parquet')

rating_counts = df['rating'].value_counts()
print('Number of ratings:')
print(rating_counts)

X = np.array(df['vector'].tolist())
y = np.array(df['rating'].tolist())

# Train-test split
X_0, X_final_test, y_0, y_final_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        random_state=123,
                                                        stratify=y,
                                                        test_size=0.2)

# K-fold cross-validation
kfold = StratifiedKFold(n_splits=5,
                        shuffle=True,
                        random_state=0)

i = 0
mses = np.zeros((3, 5))
# Regression
for i, (train_index, test_index) in enumerate(kfold.split(X_0, y_0)):
    # Get the kfold training data
    X_train = X_0[train_index, :]
    y_train = y_0[train_index]

    # Get the validation data
    X_test = X_0[test_index, :]
    y_test = y_0[test_index]

    # Under-sample training data
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # # Baseline
    # predict_baseline = y_resampled.mean() * np.ones(len(y_test))
    #
    # # Linear regression
    # linreg = LinearRegression()
    # linreg.fit(X_resampled, y_resampled)
    #
    # predict_linreg = linreg.predict(X_test)
    #
    # # KNN regression
    # knnreg = KNeighborsRegressor(n_neighbors=5)
    # knnreg.fit(X_resampled, y_resampled)
    #
    # predict_knnreg = knnreg.predict(X_test)
    #
    # mses[0, i] = mean_squared_error(y_test, predict_baseline)
    # mses[1, i] = mean_squared_error(y_test, predict_linreg)
    # mses[2, i] = mean_squared_error(y_test, predict_knnreg)

    # Ordinal regression
    y_resampled_ord = pd.Series(y_resampled).astype(CategoricalDtype(ordered=True))
    y_test_ord = pd.Series(y_test).astype(CategoricalDtype(ordered=True))
    ordreg = OrderedModel(y_resampled_ord,
                          X_resampled,
                          distr='logit')
    ordreg.fit(method='bfgs')

    predict_ordreg = ordreg.predict(X_test)

    print(mean_squared_error(y_test, predict_ordreg))



# Classification
for train_index, test_index in kfold.split(X_0, y_0):
    # Get the kfold training data
    X_train = X_0[train_index, :]
    y_train = y_0[train_index]

    # Get the validation data
    X_test = X_0[test_index, :]
    y_test = y_0[test_index]

    # Under-sample training data
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # Logistic regression, one-vs-rest
    logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
    logreg.fit(X_resampled, y_resampled)

    predict_logreg = logreg.predict(X_test)
    cm_logreg = confusion_matrix(y_test, predict_logreg)
    print('Confusion matrix for one-vs-all logistic regression:')
    print(cm_logreg)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_resampled, y_resampled)

    predict_knn = knn.predict(X_test)
    cm_knn = confusion_matrix(y_test, predict_knn)
    print('Confusion matrix for KNN:')
    print(cm_knn)


# # Logistic Regression, One-vs-All
# logreg = LogisticRegression(multi_class='ovr', solver='liblinear')
# logreg.fit(vectors_train, ratings_train)
#
# ratings_pred_logreg = logreg.predict(vectors_test)
# cm_logreg = confusion_matrix(ratings_test, ratings_pred_logreg)
# print('Confusion matrix for one-vs-all logistic regression:')
# print(cm_logreg)
#
# # Logistic Regression, Multinomial
# logreg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
# logreg_multi.fit(vectors_train, ratings_train)
#
# ratings_pred_logreg_multi = logreg_multi.predict(vectors_test)
# cm_logreg_multi = confusion_matrix(ratings_test, ratings_pred_logreg_multi)
# print('Confusion matrix for multinomial logistic regression:')
# print(cm_logreg_multi)
#
# # Decision Tree
# depth = 10
# dtree = DecisionTreeClassifier(max_depth=depth).fit(vectors_train, ratings_train)
# ratings_pred_dtree = dtree.predict(vectors_test)
#
# # creating a confusion matrix
# cm_dtree = confusion_matrix(ratings_test, ratings_pred_dtree)
# print('Confusion matrix for decision tree (depth ' + str(depth) + '):')
# print(cm_dtree)
#
# # SVM
# svm_linear = SVC(kernel='linear', C=1).fit(vectors_train, ratings_train)
# ratings_pred_svm = svm_linear.predict(vectors_test)

# # Plotting
# class_names = {1,2,3,4,5}
# disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_logreg_multi, display_labels=class_names)
# disp.plot()
# plt.show()


# # PCA
X_reduced = PCA(n_components=2).fit_transform(X_0)
colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'k'}

fig = plt.figure(1, figsize=(8, 6))

plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=[colors[r] for r in y_0],
    s=1,
)
#
# # n = len(data1_train)
# # plt.scatter(
# #     X_reduced[:n, 0],
# #     X_reduced[:n, 1],
# #     c='k',
# #     s=1,
# # )
# #
# # plt.scatter(
# #     X_reduced[4*n:5*n, 0],
# #     X_reduced[4*n:5*n, 1],
# #     c='r',
# #     s=1,
# # )
#
plt.show()
